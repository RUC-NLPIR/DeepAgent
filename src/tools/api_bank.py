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
    """API-Bank工具基类，定义工具的基本结构"""
    
    def __init__(self, name: str, description: str, input_parameters: Dict, output_parameters: Dict):
        self.name = name
        self.description = description
        self.input_parameters = input_parameters
        self.output_parameters = output_parameters
    
    def to_openai_function(self) -> Dict:
        """转换为OpenAI function格式"""
        properties = {}
        required = []
        
        for param_name, param_info in self.input_parameters.items():
            param_type = param_info.get('type', 'string')
            # 类型映射
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
            
            # 构建参数属性
            param_property = {
                "type": openai_type,
                "description": param_info.get('description', '')
            }
            
            # 对于数组类型，添加必需的 items 字段
            if openai_type == 'array':
                # 尝试从描述和参数名中推断数组元素类型
                items_type = 'string'  # 默认为字符串
                
                # 基于参数名推断
                param_name_lower = param_name.lower()
                if any(keyword in param_name_lower for keyword in ['preferences', 'genres', 'categories', 'types', 'names', 'titles', 'descriptions', 'results', 'options', 'choices']):
                    items_type = 'string'
                elif any(keyword in param_name_lower for keyword in ['numbers', 'values', 'amounts', 'prices', 'scores', 'ratings']):
                    items_type = 'number'
                elif any(keyword in param_name_lower for keyword in ['ids', 'counts', 'quantities', 'ages', 'years', 'months', 'days']):
                    items_type = 'integer'
                elif any(keyword in param_name_lower for keyword in ['flags', 'enabled', 'active', 'available']):
                    items_type = 'boolean'
                
                # 基于描述内容进一步推断
                description_lower = param_info.get('description', '').lower()
                if 'dictionary' in description_lower or 'object' in description_lower:
                    # 如果描述中提到字典或对象，说明数组元素是复杂对象
                    items_type = 'object'
                elif 'number' in description_lower or 'numeric' in description_lower:
                    items_type = 'number'
                elif 'integer' in description_lower or 'id' in description_lower:
                    items_type = 'integer'
                elif 'boolean' in description_lower or 'flag' in description_lower:
                    items_type = 'boolean'
                
                # 设置 items 字段
                if items_type == 'object':
                    # 对于复杂对象，使用通用对象类型
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
    """API-Bank工具检索器，基于语义相似度检索相关工具"""
    
    def __init__(self, model_path: str, apis_dir: str, cache_dir: str = "./cache", load_cache: bool = True):
        self.model_path = model_path
        self.apis_dir = apis_dir
        self.cache_dir = cache_dir
        self.load_cache = load_cache
        
        # 加载嵌入模型
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(model_path)
        
        # 构建工具语料库
        self.tools = self._load_all_tools()
        self.corpus = self._build_corpus()
        self.corpus_embeddings = self._build_corpus_embeddings()
        
        print(f"Loaded {len(self.tools)} tools")
    
    def _load_all_tools(self) -> List[APIBankTool]:
        """加载所有API-Bank工具"""
        tools = []
        
        # 排除的文件
        except_files = ['__init__.py', 'api.py', 'tool_search.py']
        
        for file in os.listdir(self.apis_dir):
            if file.endswith('.py') and file not in except_files:
                try:
                    api_file = file.split('.')[0]
                    module_path = os.path.join(self.apis_dir, file)
                    
                    # 动态导入模块
                    spec = importlib.util.spec_from_file_location(api_file, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 查找继承自API的类
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
        """构建工具语料库用于检索"""
        corpus = []
        for tool in self.tools:
            # 构建索引内容：名称 + 描述 + 参数信息
            index_content = f"{tool.name}, {tool.description}"
            
            # 添加参数信息
            for param_name, param_info in tool.input_parameters.items():
                index_content += f", {param_name}: {param_info.get('description', '')}"
            
            corpus.append(index_content)
        
        return corpus
    
    def _get_cache_path(self) -> str:
        """获取缓存文件路径"""
        os.makedirs(self.cache_dir, exist_ok=True)
        unique_str = self.model_path + "_" + str(len(self.tools))
        cache_name = hashlib.md5(unique_str.encode('utf-8')).hexdigest() + ".pt"
        return os.path.join(self.cache_dir, cache_name)
    
    def _build_corpus_embeddings(self) -> torch.Tensor:
        """构建语料库嵌入向量"""
        cache_path = self._get_cache_path()
        
        # if os.path.exists(cache_path) and self.load_cache:
        #     print(f"Loading corpus embeddings from cache: {cache_path}")
        #     return torch.load(cache_path)
        
        print("Building corpus embeddings...")
        start_time = time.time()
        
        # 根据模型类型格式化文本
        if "bge" in self.model_path.lower():
            formatted_corpus = self.corpus
            normalize = True
        elif "e5" in self.model_path.lower():
            formatted_corpus = [f"passage: {text}" for text in self.corpus]
            normalize = False
        else:
            formatted_corpus = self.corpus
            normalize = False
        
        # 计算嵌入向量
        corpus_embeddings = self.embedder.encode(
            formatted_corpus, 
            normalize_embeddings=normalize,
            convert_to_tensor=True
        )
        
        print(f"Corpus embeddings calculated in {time.time() - start_time:.2f} seconds")
        
        # 保存到缓存
        torch.save(corpus_embeddings, cache_path)
        print(f"Corpus embeddings saved to cache: {cache_path}")
        
        return corpus_embeddings
    
    def retrieving(self, query: str, top_k: int = 5) -> List[Dict]:
        """检索相关工具"""
        print(f"Retrieving tools for query: '{query}'")
        
        # 格式化查询
        if "bge" in self.model_path.lower():
            formatted_query = query
            normalize = True
        elif "e5" in self.model_path.lower():
            formatted_query = f"query: {query}"
            normalize = False
        else:
            formatted_query = query
            normalize = False
        
        # 计算查询嵌入
        query_embedding = self.embedder.encode(
            formatted_query,
            normalize_embeddings=normalize,
            convert_to_tensor=True
        )
        
        # 语义搜索
        hits = util.semantic_search(
            query_embedding, 
            self.corpus_embeddings, 
            top_k=top_k, 
            score_function=util.cos_sim
        )
        
        # 构建返回结果
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
    """API-Bank工具执行器，执行具体的工具调用"""
    
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
        # 初始化共享的 CheckToken 实例（若存在）
        self.token_checker = self._init_token_checker()
    
    def _init_token_checker(self):
        try:
            # 在 apis 目录中查找并加载 CheckToken 类
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
        """加载所有工具（与检索器相同的逻辑）"""
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
        """执行工具调用
        
        Args:
            tool_call: OpenAI function格式的工具调用
                {
                    "function": {
                        "name": "tool_name",
                        "arguments": '{"param1": "value1"}'
                    }
                }
        
        Returns:
            执行结果字典
        """
        try:
            function_name = tool_call['function']['name']
            arguments_str = tool_call['function']['arguments']
            
            # 解析参数
            if isinstance(arguments_str, str):
                arguments = json.loads(arguments_str)
            else:
                arguments = arguments_str
            
            # 查找工具
            if function_name not in self.tool_map:
                return {
                    'error': f"Tool '{function_name}' not found",
                    'result': None
                }
            
            # 动态导入并执行工具
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
        """与通用调用端兼容的异步接口，内部同步执行工具。"""
        return self.execute_tool(tool_call)
    
    def _execute_tool_dynamically(self, tool_name: str, arguments: Dict) -> Any:
        """动态执行工具"""
        
        # 查找工具文件
        for file in os.listdir(self.apis_dir):
            if file.endswith('.py') and file not in ['__init__.py', 'api.py', 'tool_search.py']:
                try:
                    api_file = file.split('.')[0]
                    module_path = os.path.join(self.apis_dir, file)
                    
                    spec = importlib.util.spec_from_file_location(api_file, module_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # 查找工具类
                    tool_class = getattr(module, tool_name, None)
                    if tool_class and hasattr(tool_class, 'call'):
                        # 实例化并调用（优先使用 kwargs 注入 init_database/token_checker）
                        init_kwargs: Dict[str, Any] = {}
                        # 注入数据库
                        if hasattr(tool_class, 'database_name') and self.init_databases:
                            db_name = getattr(tool_class, 'database_name')
                            if db_name in self.init_databases:
                                # 如果构造函数包含 init_database，则用关键字传入
                                if 'init_database' in inspect.signature(tool_class.__init__).parameters:
                                    init_kwargs['init_database'] = self.init_databases[db_name]
                        # 注入 token_checker（当工具需要 token 且构造函数支持）
                        needs_token = False
                        try:
                            if hasattr(tool_class, 'input_parameters') and isinstance(tool_class.input_parameters, dict):
                                needs_token = 'token' in tool_class.input_parameters
                        except Exception:
                            needs_token = False
                        if needs_token and self.token_checker is not None:
                            if 'token_checker' in inspect.signature(tool_class.__init__).parameters:
                                init_kwargs['token_checker'] = self.token_checker
                        # 若没有匹配的 kwargs，尝试回退到位置参数顺序 (init_database, token_checker)
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
        """列出所有可用工具"""
        return [tool.name for tool in self.tools]
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict]:
        """获取工具信息"""
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
    """API-Bank数据加载器"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.level1_data_path = os.path.join(data_path, 'lv1-lv2-samples', 'level-1-given-desc-e2e')
        self.level2_data_path = os.path.join(data_path, 'lv1-lv2-samples', 'level-2-toolsearcher')
        self.level3_data_path = os.path.join(data_path, 'lv3-samples')
        self.lv3_apis_path = os.path.join(data_path, 'lv3_apis')
    
    def load_level1_data(self) -> List[Dict]:
        """加载Level-1数据（给定候选APIs的场景）"""
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
            
            # 提取用户查询和API调用
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
        """加载Level-3数据（需要工具搜索的场景）"""
        data_list = []
        
        # 检查Level-3 JSON数据文件
        level3_json_path = os.path.join(self.data_path, 'test-data', 'level-3.json')
        if os.path.exists(level3_json_path):
            with open(level3_json_path, 'r', encoding='utf-8') as f:
                level3_data = json.load(f)
            
            for i, item in enumerate(level3_data):
                # 转换Level-3数据格式
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
        """解析Level-3场景文件内容"""
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
                    # 解析用户信息
                    if '"' in line:
                        # 用户信息格式
                        user_info_match = re.search(r'"([^"]+)":\s*{([^}]+)}', line)
                        if user_info_match:
                            username = user_info_match.group(1)
                            user_data = user_info_match.group(2)
                            # 解析用户数据
                            user_dict = {}
                            for item in user_data.split(','):
                                if ':' in item:
                                    key, value = item.split(':', 1)
                                    user_dict[key.strip()] = value.strip().strip('"')
                            scene_data['key_info'][username] = user_dict
                elif line.startswith('-'):
                    # 其他关键信息
                    info = line[1:].strip()
                    if info not in scene_data['key_info']:
                        scene_data['key_info']['other_info'] = scene_data['key_info'].get('other_info', [])
                        scene_data['key_info']['other_info'].append(info)
            elif current_section == 'api_calls':
                if line and not line.startswith('API Call:'):
                    # 解析API调用
                    api_call = self._parse_api_call(line)
                    if api_call:
                        scene_data['api_calls'].append(api_call)
        
        return scene_data
    
    def _parse_api_call(self, api_call_str: str) -> Dict:
        """解析API调用字符串"""
        # 格式: GetUserToken(username="JohnDoe", password="pass123")
        match = re.match(r'(\w+)\((.*)\)', api_call_str)
        if not match:
            return None
        
        api_name = match.group(1)
        params_str = match.group(2)
        
        # 解析参数
        param_dict = {}
        if params_str:
            # 简单的参数解析，处理字符串参数
            params = re.findall(r'(\w+)="([^"]*)"', params_str)
            for param_name, param_value in params:
                param_dict[param_name] = param_value
        
        return {
            'api_name': api_name,
            'param_dict': param_dict
        }
    
    def get_lv3_apis_path(self) -> str:
        """获取Level-3 APIs路径"""
        return self.lv3_apis_path


def parse_api_call(api_call_str: str) -> tuple:
    """解析API调用字符串，返回(api_name, param_dict)"""
    # 格式: GetUserToken(username="JohnDoe", password="pass123")
    match = re.match(r'(\w+)\((.*)\)', api_call_str)
    if not match:
        return None, None
    
    api_name = match.group(1)
    params_str = match.group(2)
    
    # 解析参数
    param_dict = {}
    if params_str:
        # 处理字符串参数
        params = re.findall(r'(\w+)="([^"]*)"', params_str)
        for param_name, param_value in params:
            param_dict[param_name] = param_value
    
    return api_name, param_dict


def get_api_call(text: str) -> str:
    """从文本中提取API调用"""
    # 查找格式为 [ApiName(param1=value1, param2=value2)] 的API调用
    api_call_pattern = r"\[(\w+)\((.*)\)\]"
    match = re.search(api_call_pattern, text)
    if match:
        return match.group(0)
    return None


def calculate_rouge_l_score(reference: str, hypothesis: str) -> float:
    """计算Rouge-L分数"""
    rouge = Rouge()
    if not hypothesis:
        return 0.0
    try:
        scores = rouge.get_scores(hypothesis, reference)
        return scores[0]['rouge-l']['f']
    except:
        return 0.0


def main():
    """主函数，测试工具的索引、检索和执行功能"""
    
    # 配置路径
    model_path = "./models/bge-large-en-v1.5"
    apis_dir = "./data/API-Bank/apis"
    cache_dir = "./cache"
    
    print("=" * 60)
    print("API-Bank 工具管理系统测试")
    print("=" * 60)
    
    try:
        # 1. 测试工具索引
        print("\n1. 测试工具索引...")
        executor = APIBankExecutor(apis_dir=apis_dir)
        available_tools = executor.list_available_tools()
        print(f"发现 {len(available_tools)} 个工具:")
        for i, tool_name in enumerate(available_tools[:10]):  # 只显示前10个
            print(f"  {i+1}. {tool_name}")
        if len(available_tools) > 10:
            print(f"  ... 还有 {len(available_tools) - 10} 个工具")
        
        # 2. 测试工具检索
        print("\n2. 测试工具检索...")
        retriever = APIBankRetriever(model_path=model_path, apis_dir=apis_dir, cache_dir=cache_dir)
        
        # 测试查询
        test_queries = [
            "Calculate mathematical formula",
            "Add schedule",
            "Translate text",
            "Search information",
        ]
        
        for query in test_queries:
            print(f"\n查询: '{query}'")
            retrieved = retriever.retrieve(query, top_k=3)
            print(f"检索到 {len(retrieved)} 个相关工具:")
            for i, item in enumerate(retrieved):
                print(f"  {i+1}. {item['tool'].name} (相似度: {item['score']:.3f})")
                print(f"     描述: {item['tool'].description[:100]}...")
        
        # 3. 测试工具执行
        print("\n3. 测试工具执行...")
        
        # 测试计算器工具
        if 'Calculator' in available_tools:
            print("\n测试 Calculator 工具:")
            calculator_call = {
                "function": {
                    "name": "Calculator",
                    "arguments": '{"formula": "(5+6)*3"}'
                }
            }
            
            result = executor.execute_tool(calculator_call)
            print(f"执行结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        # 测试获取今天日期工具
        if 'GetToday' in available_tools:
            print("\n测试 GetToday 工具:")
            get_today_call = {
                "function": {
                    "name": "GetToday",
                    "arguments": '{}'
                }
            }
            
            result = executor.execute_tool(get_today_call)
            print(f"执行结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        # 4. 测试OpenAI function格式
        print("\n4. 测试OpenAI function格式...")
        if available_tools:
            sample_tool = executor.get_tool_info(available_tools[0])
            if sample_tool:
                print(f"工具 '{sample_tool['name']}' 的OpenAI function格式:")
                print(json.dumps(sample_tool['openai_function'], indent=2, ensure_ascii=False))
        
        print("\n" + "=" * 60)
        print("测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
