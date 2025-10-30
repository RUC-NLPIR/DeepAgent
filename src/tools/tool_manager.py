import os
import json
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import requests


class ToolManager:
    """
    Central manager to unify tool retrieval and execution across datasets.
    Exposes:
      - await ToolManager.create(args)
      - retrieve_tools(query, top_k, executable_tools=None): List[dict]
      - await call_tool(adapted_tool_call, seq, caller_args): str | dict
    """

    def __init__(self, args):
        self.args = args
        self.retriever = None
        self.caller = None
        self.initial_obs_list = None
        self._tool_docs_cache = None
        self.url_to_snippet = {}
        self.search_cache = {}
        self.url_cache = {}
        # Runtime clients/resources
        self.vqa_client = None
        self.semaphore = None
        self.file_processor = None
        self.webshop_url_list = []
        self.aux_client = None
        self.aux_model_name = None
        # Remote retriever API base (optional). If set, use remote retrieval instead of local index/model
        try:
            self.tool_retriever_api_base = getattr(args, 'tool_retriever_api_base', None) or os.getenv('TOOL_RETRIEVER_API_BASE', None)
        except Exception:
            self.tool_retriever_api_base = None

    @classmethod
    async def create(cls, args, webshop_url_id=0):
        self = cls(args)
        await self._initialize(webshop_url_id)
        return self

    async def _initialize(self, webshop_url_id=0):
        args = self.args
        # Initialize retriever if needed
        if getattr(args, 'enable_tool_search', False):
            # Default: use remote retriever unless overridden per-dataset
            self.retriever = None

        # Initialize caller (environment/backends)
        self.caller = None
        # Configure GAIA/HLE resource directories
        try:
            if getattr(args, 'gaia_file_dir', None):
                from tools.file_process import FileProcessor
                self.file_processor = FileProcessor()
                self.file_processor.set_base_dir(args.gaia_file_dir)
                self.gaia_file_dir = args.gaia_file_dir
        except Exception:
            pass
        try:
            if getattr(args, 'hle_image_dir', None):
                # Stash for later use in VQA calls
                self.hle_image_dir = args.hle_image_dir
        except Exception:
            self.hle_image_dir = None
        if args.dataset_name in ["toolbench"]:
            from tools.rapid_api import RapidAPICaller, api_json_to_openai_json, standardize, change_name
            import pandas as pd
            all_tool_docs = []
            documents_df = pd.read_csv(args.toolbench_corpus_tsv_path, sep='\t')
            for row in documents_df.itertuples():
                tool_documentation = json.loads(row.document_content)
                tool_name = standardize(tool_documentation.get('tool_name', ''))
                openai_function, _, _ = api_json_to_openai_json(tool_documentation, tool_name)
                tool_doc = {
                    "category_name": tool_documentation.get('category_name', ''),
                    "tool_name": tool_name,
                    "api_name": change_name(standardize(tool_documentation.get('api_name', ''))),
                    "openai_function": openai_function,
                }
                all_tool_docs.append(tool_doc)
            self.caller = RapidAPICaller(tool_docs=all_tool_docs, service_url=args.toolbench_service_url, toolbench_key=args.toolbench_api)
            self._tool_docs_cache = all_tool_docs
        elif args.dataset_name == 'alfworld':
            from envs.alfworld import ALFWorldEnvWrapper, get_alfworld_function_definitions
            env = ALFWorldEnvWrapper(batch_size=134)
            self.caller = env
            self.initial_obs_list = env.reset()
        elif args.dataset_name == 'webshop':
            from envs.webshop import WebshopEnvWrapper, get_webshop_function_definitions
            webshop_url = args.webshop_service_url if webshop_url_id == 0 else args.webshop_service_url1 if webshop_url_id == 1 else args.webshop_service_url2 if webshop_url_id == 2 else args.webshop_service_url3
            env = WebshopEnvWrapper(batch_size=500, webshop_url=webshop_url)
            self.caller = env
            self.initial_obs_list = env.initial_obs_list
        elif args.dataset_name in ['tmdb', 'spotify']:
            # RestBench datasets - no global caller needed, tools are integrated in executor
            self.caller = None
        elif args.dataset_name == 'api_bank':
            from tools.api_bank import APIBankExecutor, APIBankRetriever
            if not args.enable_tool_search:
                self.caller = APIBankExecutor(apis_dir=args.api_bank_apis_dir, database_dir=args.api_bank_database_dir)
            else:
                self.caller = APIBankExecutor(apis_dir=args.api_bank_lv3_apis_abs_dir, database_dir=args.api_bank_database_dir)
                # Initialize local retriever for API-Bank when tool search is enabled
                try:
                    apis_dir = getattr(args, 'api_bank_lv3_apis_abs_dir', None) or getattr(args, 'api_bank_apis_dir', None)
                    model_path = getattr(args, 'tool_retriever_model_path', '')
                    cache_dir = getattr(args, 'tool_index_cache_dir', './cache')
                    print(apis_dir)
                    print(model_path)
                    print(cache_dir)
                    if apis_dir and model_path:
                        self.retriever = APIBankRetriever(
                            model_path=model_path,
                            apis_dir=apis_dir,
                            cache_dir=cache_dir,
                            load_cache=False,
                        )
                except Exception:
                    self.retriever = None
        else:
            self.caller = None

        # Load web caches from disk if available
        self.read_web_cache()

    def retrieve_tools(self, query: str, top_k: int, executable_tools: Optional[List[Dict]] = None) -> List[Dict]:
        # Special-case: use local retriever for API-Bank when available
        if getattr(self.args, 'dataset_name', '') == 'api_bank' and self.retriever is not None:
            try:
                return self.retriever.retrieving(query=query, top_k=int(top_k))
            except Exception:
                return []

        # Always use remote retrieval API; local retriever is disabled for other datasets
        api_base = getattr(self, 'tool_retriever_api_base', None)
        if not api_base:
            raise RuntimeError("Remote tool retriever API base not configured. Set 'tool_retriever_api_base' in args or env TOOL_RETRIEVER_API_BASE.")
        try:
            payload = {
                "dataset_name": getattr(self.args, 'dataset_name', ''),
                "query": query,
                "top_k": int(top_k),
            }
            # Narrowing for ToolHop
            if getattr(self.args, 'dataset_name', '') == 'toolhop' and executable_tools:
                # normalized = []
                # for t in executable_tools:
                #     if isinstance(t, dict) and "name" in t:
                #         normalized.append({"name": t["name"]})
                #     elif isinstance(t, str):
                #         normalized.append({"name": t.strip().split("\n")[0]})
                # if normalized:
                #     payload["executable_tools"] = normalized
                payload["executable_tools"] = executable_tools

            url = api_base.rstrip("/") + "/retrieve"
            resp = requests.post(url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            return results if isinstance(results, list) else []
        except Exception:
            return []

    async def call_tool(self, adapted_tool_call: Dict, seq: Dict) -> Any:
        args = self.args

        # Dataset-specific execution
        if args.dataset_name == 'toolhop':
            tool_name_to_call = adapted_tool_call["function"]["name"]
            target_tool_def = next((tool for tool in reversed(seq['available_tools']) if tool['tool_name'] == tool_name_to_call), None)
            if target_tool_def:
                from tools.toolhop import ToolHopCaller
                dynamic_caller = await ToolHopCaller.create(functions=target_tool_def['all_functions'])
                return dynamic_caller.call_api(adapted_tool_call)
            else:
                return {'error': f"Tool '{tool_name_to_call}' not found in available tools."}
        
        elif args.dataset_name == 'alfworld':
            arguments = adapted_tool_call["function"].get("arguments", {})
            tool_name = adapted_tool_call["function"].get("name")
            # Prefer explicit env_id passed by callers; fallback to seq['id']
            env_index = seq.get('env_id', seq['id'])
            ob_text, won, done = self.caller.step_action(env_index, tool_name, arguments)
            if done:
                seq['finished'] = True
                seq['success'] = True
                seq['reward'] = 1.0
                return ob_text + "\n\nCongratulations! You have completed the task!\n\n"
            return ob_text
        
        elif args.dataset_name == 'webshop':
            arguments = adapted_tool_call["function"].get("arguments", {})
            tool_name = adapted_tool_call["function"].get("name")
            # Prefer explicit env_id passed by callers; fallback to seq['id']
            env_index = seq.get('env_id', seq['id'])
            ob_text, reward, done = self.caller.step_action(env_index, tool_name, arguments)
            if done:
                seq['finished'] = True
                seq['success'] = reward == 1.0
                seq['reward'] = reward
                return ob_text + f"\n\nTask Completed! Final reward: {reward}\n\n"
            return ob_text
        
        elif args.dataset_name in ['tmdb', 'spotify']:
            from tools.restbench_api import execute_restbench_tool
            arguments = adapted_tool_call["function"].get("arguments", {})
            tool_name = adapted_tool_call["function"].get("name")
            return execute_restbench_tool(tool_name, arguments, args.dataset_name, args)
        
        elif args.dataset_name in ['gaia', 'hle', 'browsecomp']:
            # GAIA/HLE/BrowseComp local tools execution
            function_name = adapted_tool_call["function"].get("name")
            arguments = adapted_tool_call["function"].get("arguments", {})

            # Web search via Serper
            if function_name == 'web_search':
                # return {"error": f"web_search failed"}
                try:
                    from tools.google_search import google_serper_search_async, google_serper_search
                    api_key = getattr(args, 'serper_api_key', None) or getattr(args, 'google_serper_api', None)
                    if not api_key:
                        return {"error": "Missing Serper API key (serper_api_key or google_serper_api)."}
                    query = arguments.get('query', '')
                    if not query:
                        return {"error": "Missing required parameter: query"}
                    # Use cache if available
                    if query in self.search_cache and isinstance(self.search_cache.get(query), list):
                        print(f"Using cached search results for query: {query}")
                        return self.search_cache[query]
                    search_results = await google_serper_search_async(query=query, api_key=api_key)
                    search_results = google_serper_search(query=query, api_key=api_key)
                    if search_results:
                        for result in search_results:
                            self.url_to_snippet[result['url']] = result['snippet']
                        self.search_cache[query] = search_results
                    return search_results
                except Exception as e:
                    return {"error": f"web_search failed: {e}"}

            # Browse multiple pages (with optional Jina)
            if function_name == 'browse_pages':
                # return {"error": f"process_file failed"}
                try:
                    from tools.google_search import fetch_page_content_async, fetch_page_content, extract_snippet_with_context
                    urls = arguments.get('urls', [])
                    if not isinstance(urls, list) or len(urls) == 0:
                        return {"error": "Missing or invalid parameter: urls (non-empty list required)"}
                    use_jina = getattr(args, 'use_jina', False)
                    jina_api_key = getattr(args, 'jina_api_key', None)
                    # use_jina = False
                    # jina_api_key = None
                    extracted_text_dict = {}

                    # Handle cached URLs first
                    for url in urls:
                        if url in self.url_cache:
                            full_text = self.url_cache[url]
                            snippet = self.url_to_snippet.get(url)
                            if snippet:
                                try:
                                    ok, context = extract_snippet_with_context(full_text, snippet)
                                    extracted_text = context if ok else full_text[:10000]
                                except Exception:
                                    extracted_text = full_text[:10000]
                            else:
                                extracted_text = full_text[:10000]
                            extracted_text_dict[url] = extracted_text
                            print(f"Using cached URL: {url}")

                    # Fetch only uncached URLs
                    uncached_urls = [u for u in urls if u not in self.url_cache]
                    if uncached_urls:
                        # results_dict = await fetch_page_content_async(urls=uncached_urls, use_jina=use_jina, jina_api_key=jina_api_key, snippets=self.url_to_snippet)
                        results_dict = fetch_page_content(urls=uncached_urls, use_jina=use_jina, jina_api_key=jina_api_key, snippets=self.url_to_snippet)
                        for url, text_tuple in results_dict.items():
                            # Expect (extracted_text, full_text) from fetch helpers
                            if isinstance(text_tuple, tuple) and len(text_tuple) == 2:
                                extracted_text, full_text = text_tuple
                            else:
                                # Fallback if unexpected shape
                                extracted_text = str(text_tuple)[:10000]
                                full_text = str(text_tuple)
                            extracted_text_dict[url] = extracted_text
                            self.url_cache[url] = full_text  # save full text

                    return extracted_text_dict
                except Exception as e:
                    return {"error": f"browse_pages failed: {e}"}

            # File processing (GAIA only tool spec, safe to expose for both)
            if function_name == 'process_file':
                # return {"error": f"process_file failed"}
                try:
                    from tools.file_process import process_file_content
                    file_name = arguments.get('file_name', '')
                    if not file_name:
                        return {"error": "Missing required parameter: file_name"}
                    return await process_file_content(self.file_processor, file_name)
                except Exception as e:
                    return {"error": f"process_file failed: {e}"}

            # Python Code Execution
            if function_name == 'execute_python_code':
                # return {"error": f"execute_python_code failed"}
                try:
                    from tools.python_executor import execute_python_code
                    code = arguments.get('code', '')
                    if not code:
                        return {"error": "Missing required parameter: code"}
                    result = await execute_python_code(code)
                    return result
                except Exception as e:
                    return {"error": f"execute_python_code failed: {e}"}

            # Visual Question Answering
            if function_name == 'visual_question_answering':
                # return {"error": f"visual_question_answering failed"}
                try:
                    from tools.multimodal_tools import get_vl_completion
                    vqa_model_name = getattr(args, 'vqa_model_name', None)
                    image_name = arguments.get('image_name', '')
                    question = arguments.get('question', '')
                    if args.dataset_name == 'hle':
                        image_path = os.path.join(self.hle_image_dir, image_name)
                    elif args.dataset_name == 'gaia':
                        image_path = os.path.join(self.gaia_file_dir, image_name)
                    else:
                        image_path = image_name
                    completion, response_time = await get_vl_completion(self.vqa_client, vqa_model_name, image_path, question)
                    if completion is None:
                        return {"error": "VQA model call failed"}
                    # Standardize return
                    try:
                        text = completion.choices[0].message.content if hasattr(completion.choices[0], 'message') else completion.choices[0].text
                    except Exception:
                        text = str(completion)
                    return text
                except Exception as e:
                    return {"error": f"visual_question_answering failed: {e}"}

            # YouTube Video Question Answering
            if function_name == 'youtube_video_question_answering':
                # return {"error": f"youtube_video_question_answering failed"}
                try:
                    from tools.multimodal_tools import get_youtube_video_completion
                    vqa_model_name = getattr(args, 'vqa_model_name', None)
                    youtube_id = arguments.get('youtube_id', '')
                    question = arguments.get('question', '')
                    if not youtube_id:
                        return {"error": "Missing required parameter: youtube_id"}
                    if not question:
                        return {"error": "Missing required parameter: question"}
                    completion, response_time = await get_youtube_video_completion(self.vqa_client, vqa_model_name, youtube_id, question)
                    if completion is None:
                        return {"error": "YouTube video question answering failed"}
                    # Standardize return
                    try:
                        text = completion.choices[0].message.content if hasattr(completion.choices[0], 'message') else completion.choices[0].text
                    except Exception:
                        text = str(completion)
                    return text
                except Exception as e:
                    return {"error": f"youtube_video_question_answering failed: {e}"}

            return {"error": f"Unknown function for dataset {args.dataset_name}: {function_name}"}
        else:
            # Global caller (e.g., RapidAPI)
            if self.caller is None:
                raise RuntimeError("Tool caller is not initialized for this dataset")
            """# API simulation for ToolBench
            if self.args.dataset_name == 'toolbench' and hasattr(self.caller, 'call_api_simulation') and hasattr(self, 'aux_client'):
                return await self.caller.call_api_simulation(self.aux_client, getattr(self, 'aux_model_name', None) or getattr(self.args, 'aux_model_name', ''), adapted_tool_call)"""
            return await self.caller.call_api(adapted_tool_call)

    def call_tool_sync(self, adapted_tool_call: Dict, seq: Dict) -> Any:
        return asyncio.run(self.call_tool(adapted_tool_call, seq))

    def read_web_cache(self) -> None:
        """Load search and URL caches from disk into memory if present."""
        try:
            args = self.args
            search_dir = getattr(args, 'search_cache_dir', None)
            url_dir = getattr(args, 'url_cache_dir', None)
            if search_dir:
                os.makedirs(search_dir, exist_ok=True)
                search_path = os.path.join(search_dir, 'search_cache.json')
                if os.path.exists(search_path):
                    with open(search_path, 'r', encoding='utf-8') as f:
                        on_disk = json.load(f)
                        if isinstance(on_disk, dict):
                            # merge into memory, prefer in-memory entries
                            merged = {**on_disk, **self.search_cache}
                            self.search_cache = merged
            if url_dir:
                os.makedirs(url_dir, exist_ok=True)
                url_path = os.path.join(url_dir, 'url_cache.json')
                if os.path.exists(url_path):
                    with open(url_path, 'r', encoding='utf-8') as f:
                        on_disk = json.load(f)
                        if isinstance(on_disk, dict):
                            merged = {**on_disk, **self.url_cache}
                            self.url_cache = merged
        except Exception:
            # Fail silently to avoid breaking runs due to cache issues
            pass

    def update_web_cache(self) -> None:
        """Merge current caches with on-disk versions and persist them."""
        try:
            args = self.args
            search_dir = getattr(args, 'search_cache_dir', None)
            url_dir = getattr(args, 'url_cache_dir', None)

            # Merge with on-disk then write back for search cache
            if search_dir:
                os.makedirs(search_dir, exist_ok=True)
                search_path = os.path.join(search_dir, 'search_cache.json')
                on_disk_cache = {}
                if os.path.exists(search_path):
                    try:
                        with open(search_path, 'r', encoding='utf-8') as f:
                            on_disk_cache = json.load(f)
                            if not isinstance(on_disk_cache, dict):
                                on_disk_cache = {}
                    except Exception:
                        on_disk_cache = {}
                merged_search = {**on_disk_cache, **self.search_cache}
                with open(search_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_search, f, ensure_ascii=False, indent=2)

            # Merge with on-disk then write back for url cache
            if url_dir:
                os.makedirs(url_dir, exist_ok=True)
                url_path = os.path.join(url_dir, 'url_cache.json')
                on_disk_url = {}
                if os.path.exists(url_path):
                    try:
                        with open(url_path, 'r', encoding='utf-8') as f:
                            on_disk_url = json.load(f)
                            if not isinstance(on_disk_url, dict):
                                on_disk_url = {}
                    except Exception:
                        on_disk_url = {}
                merged_url = {**on_disk_url, **self.url_cache}
                with open(url_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_url, f, ensure_ascii=False, indent=2)
        except Exception:
            # Fail silently to avoid breaking runs due to cache issues
            pass

    # convenience alias used by run_deep_agent.py
    def save_caches(self) -> None:
        self.update_web_cache()

    def set_runtime_clients(self, vqa_client=None, semaphore=None, aux_client=None, aux_model_name=None) -> None:
        """Set runtime clients/resources after initialization."""
        if vqa_client is not None:
            self.vqa_client = vqa_client
        if semaphore is not None:
            self.semaphore = semaphore
        if aux_client is not None:
            self.aux_client = aux_client
        if aux_model_name is not None:
            self.aux_model_name = aux_model_name




def get_gaia_tool_docs(task_type: str = 'text'):
    from tools.google_search import (
        get_openai_function_web_search,
        get_openai_function_browse_pages,
    )
    from tools.python_executor import get_openai_function_execute_python_code
    from tools.file_process import get_openai_function_process_file
    from tools.multimodal_tools import (
        get_openai_function_visual_question_answering,
        get_openai_function_youtube_video_question_answering,
    )

    tool_list = [
        get_openai_function_web_search(), 
        get_openai_function_browse_pages(),
    ]
    if task_type == 'text':
        tool_list.append(get_openai_function_execute_python_code(file_process=False))
    elif task_type == 'mm':
        tool_list.extend([
            get_openai_function_execute_python_code(file_process=False), 
            # get_openai_function_youtube_video_question_answering()
        ])
    elif task_type == 'file':
        tool_list.extend([
            # get_openai_function_execute_python_code(file_process=True), 
            get_openai_function_execute_python_code(file_process=False), 
            get_openai_function_process_file(), 
            get_openai_function_visual_question_answering()
        ])

    return tool_list




def get_hle_tool_docs(task_type: str = 'text'):
    from tools.google_search import (
        get_openai_function_web_search,
        get_openai_function_browse_pages,
    )
    from tools.python_executor import get_openai_function_execute_python_code
    from tools.multimodal_tools import (
        get_openai_function_visual_question_answering,
    )

    tool_list = [
        get_openai_function_web_search(), 
        get_openai_function_browse_pages(),
        get_openai_function_execute_python_code(file_process=False)
    ]
    if task_type == 'mm':
        tool_list.append(get_openai_function_visual_question_answering())

    return tool_list




def get_browsecomp_tool_docs():
    from tools.google_search import (
        get_openai_function_web_search,
        get_openai_function_browse_pages,
    )
    # BrowseComp requires only web_search and browse_pages
    tool_list = [
        get_openai_function_web_search(),
        get_openai_function_browse_pages(),
    ]
    return tool_list

