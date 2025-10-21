import json
import sys
sys.path.append('./src/tools')
from tool_search import ToolRetriever
from func_timeout import func_set_timeout, FunctionTimedOut
import re
import asyncio
from fuzzywuzzy import process


def read_toolhop_file(file_path):
    """Reads a ToolHop JSON data file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


class ToolHopRetriever(ToolRetriever):
    """
    A retriever for the ToolHop dataset.
    It loads tools from a ToolHop JSON file, builds a searchable corpus,
    and retrieves tools in the standard OpenAI format.
    """
    def __init__(self, corpus_json_path, model_path, cache_dir, load_cache=True):
        print("Initializing ToolHopRetriever...")
        self.corpus_json_path = corpus_json_path
        
        corpus, corpus2tool = self._build_corpus()
        
        super().__init__(
            corpus=corpus,
            corpus2tool=corpus2tool,
            model_path=model_path,
            cache_dir=cache_dir,
            load_cache=load_cache,
            corpus_identifier=self.corpus_json_path
        )
        print("ToolHopRetriever initialized.")

    def _build_corpus(self):
        data = read_toolhop_file(self.corpus_json_path)
        corpus = []
        corpus2tool = {}
        tool_name_counts = {}  # Tracks the occurrences of each tool name.
        
        print(f"Building corpus from {len(data)} samples in {self.corpus_json_path}...")
        
        for sample in data:
            functions_in_sample = sample.get('functions', [])
            
            for sub_question, tool_spec in sample.get('tools', {}).items():
                original_tool_name = tool_spec.get('name', '')
                if not original_tool_name:
                    continue
                
                description = tool_spec.get('description', '')
                parameters = json.dumps(tool_spec.get('parameters', {}))
                index_content = f"{original_tool_name}\nDescription: {description}\nParameters: {parameters}"
                
                if index_content in corpus2tool:
                    continue

                corpus.append(index_content)
                
                corpus2tool[index_content] = {
                    "tool_name": original_tool_name,
                    "openai_function": tool_spec,
                    "all_functions": functions_in_sample
                }
        
        print(f"Corpus built with {len(corpus)} unique tool entries.")
        return corpus, corpus2tool

    def retrieving(self, query, top_k=5, executable_tools=None):
        """
        Retrieves the top_k most relevant tools for a given query with unique names.
        If executable_tools are provided, they are prioritized by using their specific versions.
        Returns a list of tools, each in a dictionary containing the tool name, OpenAI function schema, 
        and all related Python function definitions from its sample.
        NOTE: The ToolHop toolset includes numerous tools with highly similar names but entirely distinct functionalities.
        We need to merge the similar tools' functionalities (to be done) or use fuzzy matching to return the executable tool.
        """
        print(f"Retrieving top {top_k} tools for query: '{query}'")
        # Retrieve more candidates to have enough for deduplication and filtering
        candidate_docs = super().retrieving(query, top_k=top_k * 20)
        
        if executable_tools is None:
            executable_tools = []
            
        # Create a mapping from executable tool name & description to its full document from the corpus
        executable_tool_docs_map = {}
        executable_tool_docs_map_name = {}
        for lt in executable_tools:
            name = lt.get('name', '')
            if not name:
                continue
            desc = lt.get('description', '')
            params = json.dumps(lt.get('parameters', {}))
            index_content = f"{name}\nDescription: {desc}\nParameters: {params}"
            match_key = f"{name} {desc}"
            if index_content in self.corpus2tool:
                executable_tool_docs_map[match_key] = self.corpus2tool[index_content]
            executable_tool_docs_map_name[name] = self.corpus2tool[index_content]
        
        unique_tools = []
        seen_tool_names = set()
        for doc in candidate_docs:
            if len(unique_tools) >= top_k:
                break
            
            tool_name = doc.get("tool_name")
            # Check if this tool_name has been seen before
            if not tool_name or tool_name in seen_tool_names:
                continue

            # If this tool is (or very close to) an executable tool, use the version from executable_tool_docs_map. 
            # Otherwise, use the one from retrieval.
            if tool_name in executable_tool_docs_map_name.keys():
                tool_to_consider = executable_tool_docs_map_name[tool_name]
            else:
                doc_desc = doc.get('openai_function', {}).get('description', '')
                match_query = f"{tool_name} {doc_desc}"
                matches = process.extract(match_query, list(executable_tool_docs_map.keys()), limit=1)
                if matches and matches[0][1] >= 70:
                    matched_tool_key = matches[0][0]
                    # print(f"Matched retrieved tool:\t{match_query}\nMatched executable tool:\t{matched_tool_key}")
                    tool_to_consider = executable_tool_docs_map[matched_tool_key]
                    if tool_to_consider['tool_name'] in seen_tool_names:
                        continue
                else:
                    tool_to_consider = doc

            unique_tools.append(tool_to_consider)
            seen_tool_names.add(tool_name)

        return unique_tools


class ToolHopCaller:
    """
    Executes a tool from the ToolHop dataset within a specific context.
    This class is initialized with the Python function definitions for a given
    ToolHop sample and can execute any tool within that sample.
    """
    def __init__(self, functions, scope=None):
        """
        Initializes the caller with the execution context.

        Args:
            functions (list): A list of strings, where each string is a 
                              complete Python function definition.
            scope (dict, optional): Pre-built scope for async construction.
        """
        if not isinstance(functions, list):
            raise TypeError("`functions` must be a list of strings.")
        self.functions = functions
        if scope is not None:
            self.scope = scope
        else:
            self.scope = self._prepare_scope()

    @classmethod
    async def create(cls, functions):
        """
        Async constructor for ToolHopCaller. Builds the scope in parallel.
        Usage: caller = await ToolHopCaller.create(functions)
        """
        scope = await cls._prepare_scope_async(functions)
        return cls(functions, scope=scope)

    def _prepare_scope(self):
        """Pre-compiles all functions into a scope for execution (sync)."""
        scope = {}
        for func_str in self.functions:
            try:
                exec(func_str, scope)
            except Exception as e:
                print(f"Warning: Could not execute function string: {e}\nString: {func_str[:100]}...")
        return scope

    @staticmethod
    async def _prepare_scope_async(functions):
        """Async version: compiles all functions in parallel using asyncio.to_thread."""
        scope = {}
        async def exec_func(func_str):
            def _exec():
                try:
                    exec(func_str, scope)
                except Exception as e:
                    print(f"Warning: Could not execute function string: {e}\nString: {func_str[:100]}...")
            await asyncio.to_thread(_exec)
        await asyncio.gather(*(exec_func(f) for f in functions))
        return scope

    @func_set_timeout(10)
    def _execute_in_scope(self, function_to_call, args):
        """Executes a function within the pre-compiled scope."""
        return self.scope[function_to_call](**args)

    def call_api(self, tool_call):
        """
        Calls a single ToolHop tool based on a standard tool call format.

        Args:
            tool_call (dict): A standard tool call dictionary, e.g.,
                              {"function": {"name": "tool_name", "arguments": '{"arg": "value"}'}}

        Returns:
            dict: A dictionary containing either the 'response' from the tool or an 'error' message.
        """
        try:
            tool_name = tool_call['function']['name']
            tool_args = tool_call['function']['arguments']
        except (KeyError, json.JSONDecodeError) as e:
            return {'error': f'Invalid tool call format: {str(e)}'}

        if tool_name not in self.scope:
            return {'error': f"Tool '{tool_name}' not found in the current execution scope."}

        try:
            result = self._execute_in_scope(tool_name, tool_args)
            return {'response': result}
        except FunctionTimedOut:
            return {'error': f'Tool call for {tool_name} timed out.'}
        except Exception as e:
            # This will catch both errors during execution and potential TypeError from wrong args.
            return {'error': f'Error executing tool {tool_name}: {str(e)}'}


if __name__ == '__main__':
    import asyncio
    # Please ensure these paths are correct for your environment.
    corpus_path = '/mnt/ali-sh-1/usr/tusen/xiaoxi/DeepAgent/data/ToolHop/ToolHop.json' 
    model_path = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/bge/bge-large-en-v1.5"
    cache_dir = "/mnt/ali-sh-1/usr/tusen/xiaoxi/DeepAgent/cache/tool_index"
      
    async def demo():
        # --- Mode 1: Retrieval-based tool calling ---
        print("="*40)
        print(" DEMONSTRATING MODE 1: RETRIEVAL-BASED CALLING ")
        print("="*40)
        
        # 1. Initialize Retriever and get tools
        print("\n----- 1. Initializing Retriever -----")
        retriever = ToolHopRetriever(
            corpus_json_path=corpus_path,
            model_path=model_path,
            cache_dir=cache_dir,
            load_cache=True
        )
        
        query = "Salisbury Woodland Gardens links a zoo with which park?"
        print(f"\n----- 2. Retrieving tools for query: '{query}' -----")
        retrieved_tools = retriever.retrieving(query, top_k=1)
        
        if not retrieved_tools:
            print("No tools were retrieved. Cannot demonstrate Mode 1.")
        else:
            tool_to_call_doc = retrieved_tools[0]
            print("\nRetrieved Tool Document:")
            print(json.dumps(tool_to_call_doc, indent=2, ensure_ascii=False))

            # 3. Initialize Caller with the context from the retrieved tool (ASYNC)
            print(f"\n----- 3. Initializing Caller for tool '{tool_to_call_doc['tool_name']}' (async) -----")
            caller_mode_1 = await ToolHopCaller.create(functions=tool_to_call_doc['all_functions'])
            
            # 4. Construct a standard tool call and execute
            print("\n----- 4. Constructing and executing tool call -----")
            standard_tool_call = {
                "function": {
                    "name": tool_to_call_doc['tool_name'],
                    "arguments": {
                        "location_name": "Salisbury Woodland Gardens",
                        "link_type": "park",
                    }
                }
            }
            
            print(f"Standard Tool Call Payload:\n{json.dumps(standard_tool_call, indent=2)}")
            
            result = caller_mode_1.call_api(standard_tool_call)
            print("\nCaller result:")
            print(json.dumps(result, indent=2, ensure_ascii=False))

        # --- Mode 2: Direct tool calling with a given tool set ---
        print("\n\n" + "="*40)
        print(" DEMONSTRATING MODE 2: DIRECT CALLING ")
        print("="*40)

        # 1. Load a specific sample from the dataset (simulating a "labeled tool" context)
        print("\n----- 1. Loading a specific sample context -----")
        all_data = read_toolhop_file(corpus_path)
        # Using the first sample for demonstration
        sample_context = all_data[0] 
        sample_tools = sample_context['tools']
        sample_functions = sample_context['functions']
        print(f"Loaded sample '{sample_context['id']}' with {len(sample_tools)} tools.")
        
        # 2. Initialize Caller with the functions from this specific sample (ASYNC)
        print("\n----- 2. Initializing Caller for the sample context (async) -----")
        caller_mode_2 = await ToolHopCaller.create(functions=sample_functions)

        # 3. Choose a tool from this context and execute it
        if sample_tools:
            tool_name_to_call = list(sample_tools.values())[0]['name']
            print(f"\n----- 3. Simulating a call to tool '{tool_name_to_call}' from this context -----")
            direct_action_input = {"location_name": "Salisbury Woodland Gardens", "entity_types": ["zoo"]}
            direct_standard_tool_call = {
                "function": {
                    "name": tool_name_to_call,
                    "arguments": direct_action_input
                }
            }
            print(f"Standard Tool Call Payload:\n{json.dumps(direct_standard_tool_call, indent=2)}")
            direct_result = caller_mode_2.call_api(direct_standard_tool_call)
            print("\nCaller result:")
            print(json.dumps(direct_result, indent=2, ensure_ascii=False))
    
    asyncio.run(demo())