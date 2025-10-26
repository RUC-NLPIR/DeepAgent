import os
import json
import importlib.util
import hashlib
import time
import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict, Any, Optional
import re
from rouge import Rouge
import inspect


class APIBankTool:
    """API-Bank tool base class, defining the basic structure of tools"""
    
    def __init__(self, name: str, description: str, input_parameters: Dict, output_parameters: Dict):
        self.name = name
        self.description = description
        self.input_parameters = input_parameters
        self.output_parameters = output_parameters
    
    def to_openai_function(self) -> Dict:
        """Convert to OpenAI function format"""
        properties = {}
        required = []
        
        for param_name, param_info in self.input_parameters.items():
            param_type = param_info.get('type', 'string')
            # Type mapping
            if param_type == 'int':
                openai_type = 'integer'
            elif param_type == 'float':
                openai_type = 'number'
            elif param_type == 'bool':
                openai_type = 'boolean'
            elif param_type == 'list':
                openai_type = 'array'
            else:
                openai_type = 'string'
            
            # Build parameter properties
            param_property = {
                "type": openai_type,
                "description": param_info.get('description', '')
            }
            
            # For array type, add required items field
            if openai_type == 'array':
                # Try to infer array element type from description and parameter name
                items_type = 'string'  # Default to string
                
                # Infer based on parameter name
                param_name_lower = param_name.lower()
                if any(keyword in param_name_lower for keyword in ['preferences', 'genres', 'categories', 'types', 'names', 'titles', 'descriptions', 'results', 'options', 'choices']):
                    items_type = 'string'
                elif any(keyword in param_name_lower for keyword in ['numbers', 'values', 'amounts', 'prices', 'scores', 'ratings']):
                    items_type = 'number'
                elif any(keyword in param_name_lower for keyword in ['ids', 'counts', 'quantities', 'ages', 'years', 'months', 'days']):
                    items_type = 'integer'
                elif any(keyword in param_name_lower for keyword in ['flags', 'enabled', 'active', 'available']):
                    items_type = 'boolean'
                
                # Further infer based on description content
                description_lower = param_info.get('description', '').lower()
                if 'dictionary' in description_lower or 'object' in description_lower:
                    # If description mentions dictionary or object, array elements are complex objects
                    items_type = 'object'
                elif 'number' in description_lower or 'numeric' in description_lower:
                    items_type = 'number'
                elif 'integer' in description_lower or 'id' in description_lower:
                    items_type = 'integer'
                elif 'boolean' in description_lower or 'flag' in description_lower:
                    items_type = 'boolean'
                
                # Set items field
                if items_type == 'object':
                    # For complex objects, use generic object type
                    param_property["items"] = {"type": "object"}
                else:
                    param_property["items"] = {"type": items_type}
            
            properties[param_name] = param_property
            required.append(param_name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class APIBankRetriever:
    """API-Bank tool retriever, retrieves relevant tools based on semantic similarity"""
    
    def __init__(self, model_path: str, apis_dir: str, cache_dir: str = "./cache", load_cache: bool = True):
        self.model_path = model_path
        self.apis_dir = apis_dir
        self.cache_dir = cache_dir
        self.load_cache = load_cache
        
        # Load embedding model
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(model_path)
        
        # Build tool corpus
        self.tools = self._load_all_tools()
        self.corpus = self._build_corpus()
        self.corpus_embeddings = self._build_corpus_embeddings()
        
        print(f"Loaded {len(self.tools)} tools")
    
    def _load_all_tools(self) -> List[APIBankTool]:
        """Load all API-Bank tools"""
        tools = []
        
        # Excluded files
        except_files = ['__init__.py', 'api.py', 'tool_search.py']
        
        for file in os.listdir(self.apis_dir):
            if file.endswith('.py') and file not in except_files:
                try:
                    api_file = file.split('.')[0]
                    module_path = os.path.join(self.apis_dir, file)
                    
                    # Dynamically import module
                    spec = importlib.util.spec_from_file_location(api_file, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find classes that inherit from API
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            hasattr(attr, 'description') and 
                            hasattr(attr, 'input_parameters') and
                            hasattr(attr, 'output_parameters')):
                            
                            tool = APIBankTool(
                                name=attr_name,
                                description=attr.description,
                                input_parameters=attr.input_parameters,
                                output_parameters=attr.output_parameters
                            )
                            tools.append(tool)
                            
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
        
        return tools
    
    def _build_corpus(self) -> List[str]:
        """Build tool corpus for retrieval"""
        corpus = []
        for tool in self.tools:
            # Build index content: name + description + parameter information
            index_content = f"{tool.name}, {tool.description}"
            
            # Add parameter information
            for param_name, param_info in tool.input_parameters.items():
                index_content += f", {param_name}: {param_info.get('description', '')}"
            
            corpus.append(index_content)
        
        return corpus
    
    def _get_cache_path(self) -> str:
        """Get cache file path"""
        os.makedirs(self.cache_dir, exist_ok=True)
        unique_str = self.model_path + "_" + str(len(self.tools))
        cache_name = hashlib.md5(unique_str.encode('utf-8')).hexdigest() + ".pt"
        return os.path.join(self.cache_dir, cache_name)
    
    def _build_corpus_embeddings(self) -> torch.Tensor:
        """Build corpus embedding vectors"""
        cache_path = self._get_cache_path()
        
        # if os.path.exists(cache_path) and self.load_cache:
        #     print(f"Loading corpus embeddings from cache: {cache_path}")
        #     return torch.load(cache_path)
        
        print("Building corpus embeddings...")
        start_time = time.time()
        
        # Format text based on model type
        if "bge" in self.model_path.lower():
            formatted_corpus = self.corpus
            normalize = True
        elif "e5" in self.model_path.lower():
            formatted_corpus = [f"passage: {text}" for text in self.corpus]
            normalize = False
        else:
            formatted_corpus = self.corpus
            normalize = False
        
        # Calculate embedding vectors
        corpus_embeddings = self.embedder.encode(
            formatted_corpus, 
            normalize_embeddings=normalize,
            convert_to_tensor=True
        )
        
        print(f"Corpus embeddings calculated in {time.time() - start_time:.2f} seconds")
        
        # Save to cache
        torch.save(corpus_embeddings, cache_path)
        print(f"Corpus embeddings saved to cache: {cache_path}")
        
        return corpus_embeddings
    
    def retrieving(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant tools"""
        print(f"Retrieving tools for query: '{query}'")
        
        # Format query
        if "bge" in self.model_path.lower():
            formatted_query = query
            normalize = True
        elif "e5" in self.model_path.lower():
            formatted_query = f"query: {query}"
            normalize = False
        else:
            formatted_query = query
            normalize = False
        
        # Calculate query embedding
        query_embedding = self.embedder.encode(
            formatted_query,
            normalize_embeddings=normalize,
            convert_to_tensor=True
        )
        
        # Semantic search
        hits = util.semantic_search(
            query_embedding, 
            self.corpus_embeddings, 
            top_k=top_k, 
            score_function=util.cos_sim
        )
        
        # Build return results
        retrieved_tools = []
        for hit in hits[0]:
            tool = self.tools[hit['corpus_id']]
            retrieved_tools.append({
                'tool': tool,
                'score': hit['score'],
                'openai_function': tool.to_openai_function()
            })
        
        return retrieved_tools


class APIBankExecutor:
    """API-Bank tool executor, executes specific tool calls"""
    
    def __init__(self, apis_dir: str, database_dir: Optional[str] = None):
        self.apis_dir = apis_dir
        self.tools = self._load_all_tools()
        self.tool_map = {tool.name: tool for tool in self.tools}
        self.init_databases: Dict[str, Any] = {}
        if database_dir and os.path.isdir(database_dir):
            for file in os.listdir(database_dir):
                if file.endswith('.json'):
                    db_name = file.split('.')[0]
                    try:
                        with open(os.path.join(database_dir, file), 'r', encoding='utf-8') as f:
                            self.init_databases[db_name] = json.load(f)
                    except Exception:
                        continue
        # Initialize shared CheckToken instance (if exists)
        self.token_checker = self._init_token_checker()
    
    def _init_token_checker(self):
        try:
            # Find and load CheckToken class in apis directory
            for file in os.listdir(self.apis_dir):
                if file.endswith('.py') and file not in ['__init__.py', 'api.py', 'tool_search.py']:
                    api_file = file.split('.')[0]
                    module_path = os.path.join(self.apis_dir, file)
                    spec = importlib.util.spec_from_file_location(api_file, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    check_cls = getattr(module, 'CheckToken', None)
                    if check_cls is not None:
                        init_kwargs = {}
                        if hasattr(check_cls, 'database_name') and check_cls.database_name in self.init_databases:
                            init_kwargs['init_database'] = self.init_databases[check_cls.database_name]
                        return check_cls(**init_kwargs) if init_kwargs else check_cls()
        except Exception:
            return None
        return None
    
    def _load_all_tools(self) -> List[APIBankTool]:
        """Load all tools (same logic as retriever)"""
        tools = []
        except_files = ['__init__.py', 'api.py', 'tool_search.py']
        
        for file in os.listdir(self.apis_dir):
            if file.endswith('.py') and file not in except_files:
                try:
                    api_file = file.split('.')[0]
                    module_path = os.path.join(self.apis_dir, file)
                    
                    spec = importlib.util.spec_from_file_location(api_file, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (isinstance(attr, type) and 
                            hasattr(attr, 'description') and 
                            hasattr(attr, 'input_parameters') and
                            hasattr(attr, 'output_parameters')):
                            
                            tool = APIBankTool(
                                name=attr_name,
                                description=attr.description,
                                input_parameters=attr.input_parameters,
                                output_parameters=attr.output_parameters
                            )
                            tools.append(tool)
                            
                except Exception as e:
                    print(f"Error loading {file}: {e}")
                    continue
        
        return tools
    
    def execute_tool(self, tool_call: Dict) -> Dict:
        """Execute tool call
        
        Args:
            tool_call: OpenAI function format tool call
                {
                    "function": {
                        "name": "tool_name",
                        "arguments": '{"param1": "value1"}'
                    }
                }
        
        Returns:
            Execution result dictionary
        """
        try:
            function_name = tool_call['function']['name']
            arguments_str = tool_call['function']['arguments']
            
            # Parse arguments
            if isinstance(arguments_str, str):
                arguments = json.loads(arguments_str)
            else:
                arguments = arguments_str
            
            # Find tool
            if function_name not in self.tool_map:
                return {
                    'error': f"Tool '{function_name}' not found",
                    'result': None
                }
            
            # Dynamically import and execute tool
            result = self._execute_tool_dynamically(function_name, arguments)
            
            return {
                'success': True,
                'tool_name': function_name,
                'arguments': arguments,
                'result': result,
                'error': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Execution error: {str(e)}",
                'result': None
            }
    
    async def call_api(self, tool_call: Dict) -> Dict:
        """Asynchronous interface compatible with generic calling end, internally executes tools synchronously."""
        return self.execute_tool(tool_call)
    
    def _execute_tool_dynamically(self, tool_name: str, arguments: Dict) -> Any:
        """Dynamically execute tool"""
        
        # Find tool file
        for file in os.listdir(self.apis_dir):
            if file.endswith('.py') and file not in ['__init__.py', 'api.py', 'tool_search.py']:
                try:
                    api_file = file.split('.')[0]
                    module_path = os.path.join(self.apis_dir, file)
                    
                    spec = importlib.util.spec_from_file_location(api_file, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Find tool class
                    tool_class = getattr(module, tool_name, None)
                    if tool_class and hasattr(tool_class, 'call'):
                        # Instantiate and call (prioritize using kwargs to inject init_database/token_checker)
                        init_kwargs: Dict[str, Any] = {}
                        # Inject database
                        if hasattr(tool_class, 'database_name') and self.init_databases:
                            db_name = getattr(tool_class, 'database_name')
                            if db_name in self.init_databases:
                                # If constructor contains init_database, pass it as keyword argument
                                if 'init_database' in inspect.signature(tool_class.__init__).parameters:
                                    init_kwargs['init_database'] = self.init_databases[db_name]
                        # Inject token_checker (when tool needs token and constructor supports it)
                        needs_token = False
                        try:
                            if hasattr(tool_class, 'input_parameters') and isinstance(tool_class.input_parameters, dict):
                                needs_token = 'token' in tool_class.input_parameters
                        except Exception:
                            needs_token = False
                        if needs_token and self.token_checker is not None:
                            if 'token_checker' in inspect.signature(tool_class.__init__).parameters:
                                init_kwargs['token_checker'] = self.token_checker
                        # If no matching kwargs, try to fallback to positional argument order (init_database, token_checker)
                        if not init_kwargs:
                            init_args: List[Any] = []
                            if hasattr(tool_class, 'database_name') and self.init_databases:
                                db_name = getattr(tool_class, 'database_name')
                                if db_name in self.init_databases:
                                    init_args.append(self.init_databases[db_name])
                            if needs_token and self.token_checker is not None:
                                init_args.append(self.token_checker)
                            tool_instance = tool_class(*init_args)
                        else:
                            tool_instance = tool_class(**init_kwargs)
                        result = tool_instance.call(**arguments)
                        return result
                        
                except Exception as e:
                    continue
        
        raise Exception(f"Tool {tool_name} not found or cannot be executed")
    
    def list_available_tools(self) -> List[str]:
        """List all available tools"""
        return [tool.name for tool in self.tools]
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """Get tool information"""
        if tool_name in self.tool_map:
            tool = self.tool_map[tool_name]
            return {
                'name': tool.name,
                'description': tool.description,
                'input_parameters': tool.input_parameters,
                'output_parameters': tool.output_parameters,
                'openai_function': tool.to_openai_function()
            }
        return None


class APIBankDataLoader:
    """API-Bank data loader"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.level1_data_path = os.path.join(data_path, 'lv1-lv2-samples', 'level-1-given-desc-e2e')
        self.level2_data_path = os.path.join(data_path, 'lv1-lv2-samples', 'level-2-toolsearcher')
        self.level3_data_path = os.path.join(data_path, 'lv3-samples')
        self.lv3_apis_path = os.path.join(data_path, 'lv3_apis')
    
    def load_level1_data(self) -> List[Dict]:
        """Load Level-1 data (scenarios with given candidate APIs)"""
        data_list = []
        
        if not os.path.exists(self.level1_data_path):
            print(f"Level-1 data path not found: {self.level1_data_path}")
            return data_list
        
        jsonl_files = [f for f in os.listdir(self.level1_data_path) if f.endswith('.jsonl')]
        
        for file in jsonl_files:
            file_path = os.path.join(self.level1_data_path, file)
            chat_history = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    chat_history.append(json.loads(line.strip()))
            
            # Extract user query and API calls
            user_query = ""
            api_calls = []
            
            for item in chat_history:
                if item['role'] == 'User':
                    user_query = item['text']
                elif item['role'] == 'API':
                    api_calls.append({
                        'api_name': item['api_name'],
                        'param_dict': item['param_dict'],
                        'result': item['result']
                    })
            
            if user_query and api_calls:
                data_list.append({
                    'file': file,
                    'query': user_query,
                    'api_calls': api_calls,
                    'chat_history': chat_history
                })
        
        return data_list
    
    def load_level3_data(self) -> List[Dict]:
        """Load Level-3 data (scenarios requiring tool search)"""
        data_list = []
        
        # Check Level-3 JSON data file
        level3_json_path = os.path.join(self.data_path, 'test-data', 'level-3.json')
        if os.path.exists(level3_json_path):
            with open(level3_json_path, 'r', encoding='utf-8') as f:
                level3_data = json.load(f)
            
            for i, item in enumerate(level3_data):
                # Convert Level-3 data format
                converted_item = {
                    'id': i,
                    'requirement': item['requirement'],
                    'response': item['response'],
                    'apis': item['apis'],
                    'file': f'level-3-{i+1}.json'
                }
                data_list.append(converted_item)
            
            print(f"Loaded {len(data_list)} Level-3 samples from {level3_json_path}")
        else:
            print(f"Level-3 JSON data path not found: {level3_json_path}")
        
        return data_list
    
    def _parse_level3_scene(self, content: str) -> Dict:
        """Parse Level-3 scene file content"""
        lines = content.strip().split('\n')
        scene_data = {
            'scene': '',
            'first_utterance': '',
            'key_info': {},
            'api_calls': []
        }
        
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.startswith('Scene:'):
                scene_data['scene'] = line.replace('Scene:', '').strip()
            elif line.startswith('First Utterance:'):
                scene_data['first_utterance'] = line.replace('First Utterance:', '').strip()
            elif line.startswith('Key Info:'):
                current_section = 'key_info'
            elif line.startswith('API Call:'):
                current_section = 'api_calls'
            elif current_section == 'key_info':
                if ':' in line and not line.startswith('-'):
                    # Parse user information
                    if '"' in line:
                        # User information format
                        user_info_match = re.search(r'"([^"]+)":\s*{([^}]+)}', line)
                        if user_info_match:
                            username = user_info_match.group(1)
                            user_data = user_info_match.group(2)
                            # Parse user data
                            user_dict = {}
                            for item in user_data.split(','):
                                if ':' in item:
                                    key, value = item.split(':', 1)
                                    user_dict[key.strip()] = value.strip().strip('"')
                            scene_data['key_info'][username] = user_dict
                elif line.startswith('-'):
                    # Other key information
                    info = line[1:].strip()
                    if info not in scene_data['key_info']:
                        scene_data['key_info']['other_info'] = scene_data['key_info'].get('other_info', [])
                        scene_data['key_info']['other_info'].append(info)
            elif current_section == 'api_calls':
                if line and not line.startswith('API Call:'):
                    # Parse API call
                    api_call = self._parse_api_call(line)
                    if api_call:
                        scene_data['api_calls'].append(api_call)
        
        return scene_data
    
    def _parse_api_call(self, api_call_str: str) -> Dict:
        """Parse API call string"""
        # Format: GetUserToken(username="JohnDoe", password="pass123")
        match = re.match(r'(\w+)\((.*)\)', api_call_str)
        if not match:
            return None
        
        api_name = match.group(1)
        params_str = match.group(2)
        
        # Parse parameters
        param_dict = {}
        if params_str:
            # Simple parameter parsing, handle string parameters
            params = re.findall(r'(\w+)="([^"]*)"', params_str)
            for param_name, param_value in params:
                param_dict[param_name] = param_value
        
        return {
            'api_name': api_name,
            'param_dict': param_dict
        }
    
    def get_lv3_apis_path(self) -> str:
        """Get Level-3 APIs path"""
        return self.lv3_apis_path


def parse_api_call(api_call_str: str) -> tuple:
    """Parse API call string, return (api_name, param_dict)"""
    # Format: GetUserToken(username="JohnDoe", password="pass123")
    match = re.match(r'(\w+)\((.*)\)', api_call_str)
    if not match:
        return None, None
    
    api_name = match.group(1)
    params_str = match.group(2)
    
    # Parse parameters
    param_dict = {}
    if params_str:
        # Handle string parameters
        params = re.findall(r'(\w+)="([^"]*)"', params_str)
        for param_name, param_value in params:
            param_dict[param_name] = param_value
    
    return api_name, param_dict


def get_api_call(text: str) -> str:
    """Extract API call from text"""
    # Find API calls in format [ApiName(param1=value1, param2=value2)]
    api_call_pattern = r"\[(\w+)\((.*)\)\]"
    match = re.search(api_call_pattern, text)
    if match:
        return match.group(0)
    return None


def calculate_rouge_l_score(reference: str, hypothesis: str) -> float:
    """Calculate Rouge-L score"""
    rouge = Rouge()
    if not hypothesis:
        return 0.0
    try:
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]['rouge-l']['f']
    except:
        return 0.0


def main():
    """Main function, test tool indexing, retrieval and execution functionality"""
    
    # Configuration paths
    model_path = "/mnt/tidal-alsh01/dataset/llm_ckpt_tidal/bge/bge-large-en-v1.5"
    apis_dir = "./data/API-Bank/apis"
    cache_dir = "./cache"
    
    print("=" * 60)
    print("API-Bank Tool Management System Test")
    print("=" * 60)
    
    try:
        # 1. Test tool indexing
        print("\n1. Testing tool indexing...")
        executor = APIBankExecutor(apis_dir=apis_dir)
        available_tools = executor.list_available_tools()
        print(f"Found {len(available_tools)} tools:")
        for i, tool_name in enumerate(available_tools[:10]):  # Only show first 10
            print(f"  {i+1}. {tool_name}")
        if len(available_tools) > 10:
            print(f"  ... and {len(available_tools) - 10} more tools")
        
        # 2. Test tool retrieval
        print("\n2. Testing tool retrieval...")
        retriever = APIBankRetriever(model_path=model_path, apis_dir=apis_dir, cache_dir=cache_dir)
        
        # Test queries
        test_queries = [
            "Calculate mathematical formula",
            "Add schedule",
            "Translate text",
            "Search information",
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            retrieved = retriever.retrieve(query, top_k=3)
            print(f"Retrieved {len(retrieved)} relevant tools:")
            for i, item in enumerate(retrieved):
                print(f"  {i+1}. {item['tool'].name} (similarity: {item['score']:.3f})")
                print(f"     Description: {item['tool'].description[:100]}...")
        
        # 3. Test tool execution
        print("\n3. Testing tool execution...")
        
        # Test calculator tool
        if 'Calculator' in available_tools:
            print("\nTesting Calculator tool:")
            calculator_call = {
                "function": {
                    "name": "Calculator",
                    "arguments": '{"formula": "(5+6)*3"}'
                }
            }
            
            result = executor.execute_tool(calculator_call)
            print(f"Execution result: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        # Test get today date tool
        if 'GetToday' in available_tools:
            print("\nTesting GetToday tool:")
            get_today_call = {
                "function": {
                    "name": "GetToday",
                    "arguments": '{}'
                }
            }
            
            result = executor.execute_tool(get_today_call)
            print(f"Execution result: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        # 4. Test OpenAI function format
        print("\n4. Testing OpenAI function format...")
        if available_tools:
            sample_tool = executor.get_tool_info(available_tools[0])
            if sample_tool:
                print(f"OpenAI function format for tool '{sample_tool['name']}':")
                print(json.dumps(sample_tool['openai_function'], indent=2, ensure_ascii=False))
        
        print("\n" + "=" * 60)
        print("Test completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error occurred during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
