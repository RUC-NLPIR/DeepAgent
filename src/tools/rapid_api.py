import re
import json
import pandas as pd
from tqdm import tqdm
from .tool_search import ToolRetriever
import asyncio
import aiohttp
import time
import requests
from prompts.prompts_deepagent import get_rapidapi_simulation_prompt


def process_retrieval_document(documents_df):
    ir_corpus = {}
    corpus2tool = {}
    for row in tqdm(documents_df.itertuples(), desc="Processing retrieval documents"):
        tool_documentation = json.loads(row.document_content)
        index_content = (tool_documentation.get('category_name', '') or '') + ', ' + \
            (tool_documentation.get('tool_name', '') or '') + ', ' + \
            (tool_documentation.get('api_name', '') or '') + ', ' + \
            (tool_documentation.get('api_description', '') or '') + \
            ', required_params: ' + json.dumps(tool_documentation.get('required_parameters', '')) + \
            ', optional_params: ' + json.dumps(tool_documentation.get('optional_parameters', '')) + \
            ', return_schema: ' + json.dumps(tool_documentation.get('template_response', ''))
        ir_corpus[row.docid] = index_content
        corpus2tool[index_content] = tool_documentation
    return ir_corpus, corpus2tool


def standardize_category(category):
    save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save_category or "," in save_category:
        save_category = save_category.replace(" ", "_").replace(",", "_")
    save_category = save_category.replace("__", "_")
    return save_category

def standardize(string):
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break
    if len(string) > 0 and string[0].isdigit():
        string = "get_" + string
    return string

def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name


def api_json_to_openai_json(api_json, standard_tool_name):
    description_max_length=256
    templete =     {
        "name": "",
        "description": "",
        "parameters": {
            "type": "object",
            "properties": {
            },
            "required": [],
            "optional": [],
        }
    }
    
    map_type = {
        "NUMBER": "integer",
        "STRING": "string",
        "BOOLEAN": "boolean"
    }

    pure_api_name = change_name(standardize(api_json["api_name"]))
    templete["name"] = pure_api_name+ f"_for_{standard_tool_name}"
    templete["name"] = templete["name"][-64:]

    templete["description"] = f"This is the subfunction for tool \"{standard_tool_name}\", you can use this tool."
        
    if api_json["api_description"].strip() != "":
        tuncated_description = api_json['api_description'].strip().replace(api_json['api_name'],templete['name'])[:description_max_length]
        templete["description"] = templete["description"] + f"The description of this function is: \"{tuncated_description}\""
    if "required_parameters" in api_json.keys() and len(api_json["required_parameters"]) > 0:
        for para in api_json["required_parameters"]:
            name = standardize(para["name"])
            name = change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else:
                param_type = "string"
            prompt = {
                "type":param_type,
                "description":para["description"][:description_max_length],
            }

            default_value = para['default']
            if len(str(default_value)) != 0:    
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                    "example_value": default_value
                }
            else:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length]
                }

            templete["parameters"]["properties"][name] = prompt
            templete["parameters"]["required"].append(name)
        for para in api_json["optional_parameters"]:
            name = standardize(para["name"])
            name = change_name(name)
            if para["type"] in map_type:
                param_type = map_type[para["type"]]
            else:
                param_type = "string"

            default_value = para['default']
            if len(str(default_value)) != 0:    
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length],
                    "example_value": default_value
                }
            else:
                prompt = {
                    "type":param_type,
                    "description":para["description"][:description_max_length]
                }

            templete["parameters"]["properties"][name] = prompt
            templete["parameters"]["optional"].append(name)

    return templete, api_json["category_name"],  pure_api_name


class RapidAPIRetriever(ToolRetriever):
    def __init__(self, corpus_tsv_path, model_path, cache_dir, load_cache=True):
        self.corpus_tsv_path = corpus_tsv_path
        documents_df = pd.read_csv(self.corpus_tsv_path, sep='\t')
        corpus_dict, corpus2tool = process_retrieval_document(documents_df)
        corpus = list(corpus_dict.values())
        
        super().__init__(
            corpus=corpus,
            corpus2tool=corpus2tool,
            model_path=model_path,
            cache_dir=cache_dir,
            load_cache=load_cache,
            corpus_identifier=self.corpus_tsv_path 
        )

    def retrieving(self, query, top_k=10, excluded_tools={}):
        retrieved_tool_docs = super().retrieving(query, top_k=top_k * 10) # Retrieve more to filter
        
        retrieved_tools = []
        seen_tools = set()

        for tool_doc in retrieved_tool_docs:
            if len(retrieved_tools) >= top_k:
                break

            category = standardize_category(tool_doc.get('category_name', ''))
            tool_name = standardize(tool_doc.get('tool_name', ''))
            api_name = change_name(standardize(tool_doc.get('api_name', '')))

            if excluded_tools and category in excluded_tools and tool_name in excluded_tools[category]:
                continue
            
            # Avoid adding duplicate tools (based on category and tool_name)
            if (category, tool_name, api_name) in seen_tools:
                continue
            
            openai_function, _, _ = api_json_to_openai_json(tool_doc, tool_name)
            
            tmp_dict = {
                "category_name": tool_doc.get('category_name', ''),
                "tool_name": tool_name,
                "api_name": api_name,
                "openai_function": openai_function,
            }
            retrieved_tools.append(tmp_dict)
            seen_tools.add((category, tool_name, api_name))

        return retrieved_tools[:top_k]


class RapidAPICaller:
    """
    Executes a tool from the RapidAPI dataset within a specific context.
    This class is initialized with a list of tool documents and can execute
    any tool within that set based on a standard tool call format.
    """
    def __init__(self, tool_docs, service_url="http://8.130.32.149:8080/rapidapi", toolbench_key=None):
        """
        Initializes the caller with the execution context.

        Args:
            tool_docs (list): A list of tool document dictionaries, as returned
                              by RapidAPIRetriever or processed from an instruction file.
            service_url (str): The URL of the RapidAPI execution service.
            toolbench_key (str, optional): The key for ToolBench services.
        """
        self.service_url = service_url
        self.toolbench_key = toolbench_key
        # Map from OpenAI function name to the full tool document for quick lookup
        self.tool_map = {
            doc['openai_function']['name']: doc for doc in tool_docs
        }

    async def call_api_simulation(self, client, model_name, tool_call, temperature: float = 0.2, top_p: float = 1.0, max_tokens: int = 2048):
        """Simulate the API call via aux LLM using a strict JSON-only prompt.

        Args:
            client: AsyncOpenAI client to use (aux_client)
            model_name: model name to use for simulation (args.aux_model_name)
            tool_call: standard tool call dict
        Returns:
            dict parsed from JSON, or {"response": raw_text} if parsing fails
        """
        try:
            tool_name_from_call = tool_call['function']['name']
            action_input_str = tool_call['function']['arguments']
        except KeyError as e:
            return {'error': f'Invalid tool call format: Missing key {e}'}

        tool_doc = self.tool_map.get(tool_name_from_call)
        if not tool_doc:
            return {'error': f"Tool '{tool_name_from_call}' not found in the current tool set."}

        api_name = tool_doc.get('api_name', '')
        tool_name = tool_doc.get('tool_name', '')
        category_name = tool_doc.get('category_name', '')
        openai_function_def = tool_doc.get('openai_function', {})

        prompt = get_rapidapi_simulation_prompt(
            api_name=api_name,
            tool_name=tool_name,
            category_name=category_name,
            openai_function_def=openai_function_def,
            arguments_json=action_input_str,
        )

        try:
            completion = await client.completions.create(
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            text = completion.choices[0].text if completion.choices else ""
            # Try to extract JSON
            json_text = _extract_json_block(text)
            try:
                return json.loads(json_text)
            except Exception:
                # return raw text in a standard field if not valid JSON
                return {"response": text.strip()}
        except Exception as e:
            return {"error": f"Simulation error: {str(e)}"}


def _extract_json_block(text: str) -> str:
    """Extract JSON from output (supports fenced code blocks)."""
    if not isinstance(text, str):
        return "{}"
    # match ```json ... ``` or ``` ... ```
    m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if m2:
        return m2.group(1).strip()
    return text.strip()

    async def call_api(self, tool_call):
        """
        Calls a single RapidAPI tool based on a standard tool call format.

        Args:
            tool_call (dict): A standard tool call dictionary, e.g.,
                              {"function": {"name": "tool_name", "arguments": '{"arg": "value"}'}}

        Returns:
            dict: A dictionary containing either the 'response' from the API or an 'error' message.
        """
        try:
            tool_name_from_call = tool_call['function']['name']
            action_input_str = tool_call['function']['arguments']
        except KeyError as e:
            return {'error': f'Invalid tool call format: Missing key {e}'}

        tool_doc = self.tool_map.get(tool_name_from_call)
        if not tool_doc:
            return {'error': f"Tool '{tool_name_from_call}' not found in the current tool set."}
        
        payload = {
            "category": tool_doc['category_name'],
            "tool_name": tool_doc['tool_name'],
            "api_name": tool_doc['api_name'],
            "tool_input": action_input_str,  # The arguments should already be a JSON string.
            "strip": "raw",
            "toolbench_key": self.toolbench_key
        }
        
        await asyncio.sleep(2) # rate limit: 30 per minute
        headers = {"content-type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=15)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.service_url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        return {"error": f"Request invalid, data error. status_code={response.status}", "response": ""}
                    text = await response.text()
                    try:
                        response_json = json.loads(text)
                    except json.JSONDecodeError:
                        return {"error": f"Request invalid, not a valid json response. The response is: {text}", "response": ""}
                    return response_json
        except asyncio.TimeoutError:
            return {"error": "Timeout error...", "response": ""}
        except Exception as e:
            return {"error": f"Request error: {str(e)}", "response": ""}

    def call_api_sync(self, tool_call):
        """
        Synchronous version of call_api. Performs the same operation using 'requests'.

        Args:
            tool_call (dict): A standard tool call dictionary, e.g.,
                              {"function": {"name": "tool_name", "arguments": '{"arg": "value"}'}}

        Returns:
            dict: A dictionary containing either the 'response' from the API or an 'error' message.
        """
        try:
            tool_name_from_call = tool_call['function']['name']
            action_input_str = tool_call['function']['arguments']
        except KeyError as e:
            return {'error': f'Invalid tool call format: Missing key {e}'}

        tool_doc = self.tool_map.get(tool_name_from_call)
        if not tool_doc:
            return {'error': f"Tool '{tool_name_from_call}' not found in the current tool set."}

        payload = {
            "category": tool_doc['category_name'],
            "tool_name": tool_doc['tool_name'],
            "api_name": tool_doc['api_name'],
            "tool_input": action_input_str,
            "strip": "raw",
            "toolbench_key": self.toolbench_key
        }

        time.sleep(2)  # rate limit: 30 per minute
        headers = {"content-type": "application/json"}
        try:
            resp = requests.post(self.service_url, json=payload, headers=headers, timeout=15)
            if resp.status_code != 200:
                return {"error": f"Request invalid, data error. status_code={resp.status_code}", "response": ""}
            text = resp.text
            try:
                response_json = json.loads(text)
            except json.JSONDecodeError:
                return {"error": f"Request invalid, not a valid json response. The response is: {text}", "response": ""}
            return response_json
        except requests.Timeout:
            return {"error": "Timeout error...", "response": ""}
        except Exception as e:
            return {"error": f"Request error: {str(e)}", "response": ""}


if __name__ == "__main__":
    # Common paths and keys
    corpus_tsv_path = "/mnt/ali-sh-1/usr/tusen/xiaoxi/DeepAgent/data/ToolBench/retrieval/G1/corpus.tsv"
    instruction_file_path = "/mnt/ali-sh-1/usr/tusen/xiaoxi/DeepAgent/data/ToolBench/test_instruction/G3_instruction.json"
    model_path = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/bge/bge-large-en-v1.5"
    cache_dir = "/mnt/ali-sh-1/usr/tusen/xiaoxi/DeepAgent/cache/tool_index"
    toolbench_key="K7s90nVFqHDokzwSPZCLjEAIxpuM3NWR1QyG5284tfaYOUvgBc"

    async def demo():
        # --- Mode 1: Retrieval-based tool calling ---
        print("="*40)
        print(" DEMONSTRATING MODE 1: RETRIEVAL-BASED CALLING ")
        print("="*40)

        # 1. Initialize Retriever
        print("\n----- 1. Initializing Retriever -----")
        retriever = RapidAPIRetriever(corpus_tsv_path=corpus_tsv_path, model_path=model_path, cache_dir=cache_dir, load_cache=True)

        # 2. Retrieve a set of tools for a query
        query = "What is the weather in New York City tomorrow?"
        print(f"\n----- 2. Retrieving tools for query: '{query}' -----")
        retrieved_tools = retriever.retrieving(query, top_k=5)

        if not retrieved_tools:
            print("No tools were retrieved. Cannot demonstrate Mode 1.")
        else:
            print(f"Retrieved {len(retrieved_tools)} tools.")
            
            # 3. Initialize Caller with the retrieved tools context
            print("\n----- 3. Initializing Caller with retrieved tools context -----")
            caller_mode_1 = RapidAPICaller(tool_docs=retrieved_tools, toolbench_key=toolbench_key)
            
            # 4. Choose a tool and execute a standard tool call
            tool_to_call_doc = retrieved_tools[0]
            tool_name_to_call = tool_to_call_doc['openai_function']['name']
            print(f"\n----- 4. Simulating a call to tool '{tool_name_to_call}' -----")

            action_input = {"city": "New York,US"}
            standard_tool_call = {
                "function": {
                    "name": tool_name_to_call,
                    "arguments": json.dumps(action_input)
                }
            }
            
            print(f"Standard Tool Call Payload:\n{json.dumps(standard_tool_call, indent=2)}")
            
            result = await caller_mode_1.call_api(standard_tool_call)
            print("\nCaller result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

        # --- Mode 2: Direct tool calling with a given tool set ---
        print("\n\n" + "="*40)
        print(" DEMONSTRATING MODE 2: DIRECT CALLING (from instruction file) ")
        print("="*40)

        # 1. Load a specific sample from the instruction file
        print(f"\n----- 1. Loading instruction sample from {instruction_file_path} -----")
        try:
            with open(instruction_file_path, 'r', encoding='utf-8') as f:
                instructions = json.load(f)
            
            instruction_sample = instructions[0]
            raw_api_list = instruction_sample['api_list']
            print(f"Loaded sample for query: '{instruction_sample['query'][:50]}...' with {len(raw_api_list)} labeled tools.")

            # 2. Process the raw api_list into the standard tool_docs format
            direct_context_tool_docs = []
            for raw_api_json in raw_api_list:
                # Add 'tool_name' if it's missing from a level, assuming it exists
                if 'tool_name' not in raw_api_json and 'name' in raw_api_json:
                     raw_api_json['tool_name'] = raw_api_json['name'] # Fallback
                
                standard_tool_name = standardize(raw_api_json.get('tool_name', ''))
                openai_function, _, _ = api_json_to_openai_json(raw_api_json, standard_tool_name)
                
                tool_doc = {
                    "category_name": raw_api_json.get('category_name', ''),
                    "tool_name": standard_tool_name,
                    "api_name": change_name(standardize(raw_api_json.get('api_name', ''))),
                    "openai_function": openai_function 
                }
                direct_context_tool_docs.append(tool_doc)
            
            # 3. Initialize Caller with this specific context
            print("\n----- 2. Initializing Caller for the direct context -----")
            caller_mode_2 = RapidAPICaller(tool_docs=direct_context_tool_docs, toolbench_key=toolbench_key)
            
            # 4. Choose a tool from THIS context and execute it
            tool_to_call_doc_direct = direct_context_tool_docs[0]
            tool_name_to_call_direct = tool_to_call_doc_direct['openai_function']['name']
            print(f"\n----- 3. Simulating a direct call to tool '{tool_name_to_call_direct}' -----")
            
            # Arguments for the first tool in G3_instructions.json (Vimeo - GetRelatedChannels)
            direct_action_input = {"category": "cinema", "format": "json"}
            direct_standard_tool_call = {
                "function": {
                    "name": tool_name_to_call_direct,
                    "arguments": json.dumps(direct_action_input)
                }
            }
            
            print(f"Standard Tool Call Payload:\n{json.dumps(direct_standard_tool_call, indent=2)}")

            direct_result = await caller_mode_2.call_api(direct_standard_tool_call)
            print("\nCaller result:")
            print(json.dumps(direct_result, indent=2, ensure_ascii=False))

        except FileNotFoundError:
            print(f"\nERROR: Could not find instruction file at '{instruction_file_path}'. Cannot demonstrate Mode 2.")
        except Exception as e:
            print(f"\nAn unexpected error occurred during Mode 2 demonstration: {e}")

    asyncio.run(demo())
