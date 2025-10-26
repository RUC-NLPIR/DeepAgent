import json
import requests
import base64
from typing import Dict, Any, List, Optional
import yaml
import sys
sys.path.append('./src')
from utils.oas_utils import ReducedOpenAPISpec, reduce_openapi_spec
from tools.tool_search import ToolRetriever
import os
import spotipy


class RestBenchAPITools:
    """RestBench API tools for DeepAgent integration"""
    
    def __init__(self, dataset_name: str, args):
        """
        Initialize RestBench API tools
        
        Args:
            dataset_name: 'tmdb' or 'spotify'
            args: Global args namespace containing API keys and spec paths
        """
        self.dataset_name = dataset_name
        self.args = args
        
        # Load API specification using paths from args
        if dataset_name == 'tmdb':
            self.api_spec = self._load_spec(self.args.tmdb_toolset_path)
            # self.base_url = "https://api.themoviedb.org/3"  # Not callable
            self.base_url = "https://api.tmdb.org/3"
            self.headers = {"Authorization": f"Bearer {self.args.tmdb_access_token}"}
        elif dataset_name == 'spotify':
            self.api_spec = self._load_spec(self.args.spotify_toolset_path)
            self.base_url = "https://api.spotify.com/v1"
            # Prefer user-auth like run_spotify.py; fallback to client credentials on failure
            try:
                self.access_token = self._get_spotify_user_token()
                # self.access_token = "BQBuhvmmGGdyT1dzvuBFKt7C98zceoJEGeHo5BsurAhiffRML6cQwl8vSWqNbX289PdBA24msDbKtwK704G5lNtbld1pzjvhczgwO1mCzy4Ci3ABzXGsXnjobRrxgY2r1-ZCQmdd9kWxWw5MErfl7W04cc1B7JB5eOt3qAjT8wHRudYb8hll3eaTuigne5rqQAqpAf5OzBZTfgiVrWr4SlCyCqiiVxm4d7KbYe3bs4F5oEGbPNAV3CzVbS-xTmUVNfHs5U94ewG61Ae0HLamIk4s7bBfsi4Id11gFjR9bSpVPbJdxB6-rBS2GLQVYZEXsgk8c8VHmYdShqdOBVTMs58aKnKWkO01pqfCaiLwyDVsTVCWprygqtx56mPT"
            except Exception:
                self.access_token = self._get_spotify_token_client_credentials()
            self.headers = {"Authorization": f"Bearer {self.access_token}"}
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Create endpoint name + description mapping
        self.endpoint_descriptions = self._create_endpoint_descriptions()
        # Create endpoint tool name mapping for retrieval/execution
        self.endpoint_tool_map = self._create_endpoint_tool_map()
    
    def _load_spec(self, spec_path: str) -> ReducedOpenAPISpec:
        """Load OpenAPI specification from provided path"""
        with open(spec_path, 'r', encoding='utf-8') as f:
            raw_spec = json.load(f)
        return reduce_openapi_spec(raw_spec, only_required=False, merge_allof=True)
    
    def _get_spotify_token_client_credentials(self) -> str:
        """Get Spotify access token using client credentials flow (app-only)"""
        auth_url = "https://accounts.spotify.com/api/token"
        client_id = self.args.spotipy_client_id
        client_secret = self.args.spotipy_client_secret
        
        auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
        headers = {
            "Authorization": f"Basic {auth_header}",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        data = {"grant_type": "client_credentials"}
        
        response = requests.post(auth_url, headers=headers, data=data)
        response.raise_for_status()
        return response.json().get("access_token")
    
    def _get_spotify_user_token(self) -> str:
        """Get Spotify user access token using Authorization Code Flow (run_spotify.py standard)."""
        # Load raw OAS from args path to extract scopes like run_spotify.py
        with open(self.args.spotify_toolset_path, 'r', encoding='utf-8') as f:
            raw_api_spec = json.load(f)
        scopes = list(raw_api_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
        # Set env vars for spotipy
        os.environ['SPOTIPY_CLIENT_ID'] = self.args.spotipy_client_id
        os.environ['SPOTIPY_CLIENT_SECRET'] = self.args.spotipy_client_secret
        os.environ['SPOTIPY_REDIRECT_URI'] = getattr(self.args, 'spotipy_redirect_uri', '') or ''
        if not os.environ['SPOTIPY_REDIRECT_URI']:
            raise RuntimeError("SPOTIPY_REDIRECT_URI missing in args for user auth")
        # Prompt user to authorize in browser
        return spotipy.util.prompt_for_user_token(scope=','.join(scopes))
    
    def _create_endpoint_descriptions(self) -> Dict[str, str]:
        """Create mapping of endpoint names to descriptions"""
        descriptions = {}
        for name, description, docs in self.api_spec.endpoints:
            # Clean up description (take first sentence if too long)
            if description:
                desc = description.split('.')[0] if '.' in description else description
                descriptions[name] = desc
            else:
                # Fallback: create description from endpoint name
                method, path = name.split(' ', 1)
                descriptions[name] = f"{method} request to {path}"
        return descriptions
    
    def _normalize_endpoint_name(self, endpoint_name: str) -> str:
        """Normalize endpoint name to a safe OpenAI function name"""
        # Example: 'GET /movie/{id}/keywords' -> 'get_movie_id_keywords'
        name = endpoint_name.strip().lower()
        # Replace path params braces with nothing
        name = name.replace('{', '').replace('}', '')
        # Replace spaces and slashes with underscores
        name = name.replace(' ', '_').replace('/', '_')
        # Remove duplicate underscores
        while '__' in name:
            name = name.replace('__', '_')
        # Ensure starts with letter
        if not name[0].isalpha():
            name = 'rb_' + name
        return name
    
    def _create_endpoint_tool_map(self) -> Dict[str, Dict[str, Any]]:
        """Create a mapping from normalized function name to endpoint details"""
        tool_map: Dict[str, Dict[str, Any]] = {}
        for name, description, docs in self.api_spec.endpoints:
            func_name = self._normalize_endpoint_name(name)
            tool_map[func_name] = {
                'endpoint_name': name,
                'description': self.endpoint_descriptions.get(name, ''),
                'docs': docs,
            }
        return tool_map
    
    def get_api_details(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific API endpoint
        
        Args:
            endpoint_name: The endpoint name (e.g., "GET /search/movie")
            
        Returns:
            Dictionary containing endpoint details
        """
        try:
            # Find the endpoint in our spec
            endpoint_info = None
            for name, description, docs in self.api_spec.endpoints:
                if name == endpoint_name:
                    endpoint_info = (name, description, docs)
                    break
            
            if not endpoint_info:
                return {
                    "error": f"Endpoint '{endpoint_name}' not found. Available endpoints: {list(self.endpoint_descriptions.keys())}"
                }
            
            name, description, docs = endpoint_info
            
            # Extract key information
            details = {
                "endpoint": name,
                "description": description or "No description available",
                "base_url": self.base_url,
                "full_url": f"{self.base_url}{name.split(' ', 1)[1]}" if ' ' in name else self.base_url
            }
            
            # Add parameters if available
            if docs.get("parameters"):
                details["parameters"] = []
                for param in docs["parameters"]:
                    param_info = {
                        "name": param.get("name"),
                        "in": param.get("in"),  # query, path, header
                        "required": param.get("required", False),
                        "type": param.get("schema", {}).get("type", "string"),
                        "description": param.get("description", "")
                    }
                    details["parameters"].append(param_info)
            
            # Add request body if available
            if docs.get("requestBody"):
                details["request_body"] = {
                    "required": docs["requestBody"].get("required", False),
                    "content_type": list(docs["requestBody"].get("content", {}).keys())
                }
            
            # Add response schema if available
            if docs.get("responses") and "200" in docs["responses"]:
                response_200 = docs["responses"]["200"]
                if "content" in response_200 and "application/json" in response_200["content"]:
                    schema = response_200["content"]["application/json"].get("schema", {})
                    if "properties" in schema:
                        details["response_properties"] = list(schema["properties"].keys())
            
            return details
            
        except Exception as e:
            return {"error": f"Failed to get API details: {str(e)}"}
    
    def call_api(self, endpoint_name: str, method: str, path: str, 
                 params: Optional[Dict[str, Any]] = None, 
                 data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Call a specific API endpoint
        
        Args:
            endpoint_name: The endpoint name (e.g., "GET /search/movie")
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path (e.g., "/search/movie")
            params: Query parameters for GET requests
            data: Request body for POST/PUT requests
            
        Returns:
            Dictionary containing API response or error information
        """
        try:
            # Validate endpoint exists
            if endpoint_name not in self.endpoint_descriptions:
                return {
                    "error": f"Endpoint '{endpoint_name}' not found. Available endpoints: {list(self.endpoint_descriptions.keys())}"
                }
            
            # Construct full URL
            full_url = f"{self.base_url}{path}"
            
            # Make the request
            if method.upper() == "GET":
                response = requests.get(full_url, headers=self.headers, params=params, timeout=30)
            elif method.upper() == "POST":
                response = requests.post(full_url, headers=self.headers, params=params, json=data, timeout=30)
            elif method.upper() == "PUT":
                response = requests.put(full_url, headers=self.headers, params=params, json=data, timeout=30)
            elif method.upper() == "DELETE":
                response = requests.delete(full_url, headers=self.headers, params=params, timeout=30)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
            
            # Process response
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    return {
                        "success": True,
                        "status_code": response.status_code,
                        "endpoint": endpoint_name,
                        "url": full_url,
                        "response": response_data
                    }
                except json.JSONDecodeError:
                    return {
                        "success": True,
                        "status_code": response.status_code,
                        "endpoint": endpoint_name,
                        "url": full_url,
                        "response": response.text
                    }
            else:
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "endpoint": endpoint_name,
                    "url": full_url,
                    "error": response.text
                }
                
        except Exception as e:
            return {"error": f"API call failed: {str(e)}"}
    
    def get_all_endpoints_summary(self) -> List[str]:
        """Get a summary of all available endpoints for the model"""
        summary = []
        for name, description in self.endpoint_descriptions.items():
            summary.append(f"{name}: {description}")
        return summary
    
    def get_endpoint_openai_functions(self) -> List[Dict[str, Any]]:
        """Generate OpenAI function schemas for each endpoint (for retrieval)"""
        functions: List[Dict[str, Any]] = []
        for endpoint_name, meta in self.endpoint_tool_map.items():
            full_name = meta['endpoint_name']
            docs = meta['docs']
            description = meta['description']
            # Build parameter schema from docs.parameters and requestBody
            properties: Dict[str, Any] = {}
            required_fields: List[str] = []
            # Query/path params under a single 'params' object to keep interface consistent
            param_props: Dict[str, Any] = {}
            if docs.get('parameters'):
                for p in docs['parameters']:
                    pname = p.get('name')
                    ptype = (p.get('schema') or {}).get('type', 'string')
                    pdesc = p.get('description', '')
                    param_props[pname] = {"type": ptype, "description": pdesc}
                    if p.get('required', False):
                        # We'll keep params object optional; inside it's up to the model
                        pass
            if param_props:
                properties['params'] = {
                    "type": "object",
                    "description": "Query or path parameters for this endpoint",
                    "properties": param_props,
                    "additionalProperties": True
                }
            # Request body under 'data'
            if docs.get('requestBody'):
                properties['data'] = {
                    "type": "object",
                    "description": "Request body for this endpoint",
                    "additionalProperties": True
                }
            # Also allow generic fields for flexibility
            properties.setdefault('params', {"type": "object", "additionalProperties": True})
            
            functions.append({
                "name": self._normalize_endpoint_name(full_name),
                "description": f"{full_name} — {description}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                }
            })
        return functions
    
    def call_endpoint_function(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a dynamic per-endpoint function by mapping to call_api"""
        meta = self.endpoint_tool_map.get(tool_name)
        if not meta:
            return {"error": f"Unknown endpoint tool: {tool_name}"}
        full_name = meta['endpoint_name']  # e.g., 'GET /search/movie'
        try:
            method, path = full_name.split(' ', 1)
        except ValueError:
            return {"error": f"Invalid endpoint format: {full_name}"}
        params = arguments.get('params')
        data = arguments.get('data')
        return self.call_api(full_name, method, path, params=params, data=data)


class RestBenchRetriever(ToolRetriever):
    """
    Retriever for RestBench (TMDB/Spotify) endpoints.
    Builds a corpus of full endpoint function schemas (OpenAI function format) and supports semantic retrieval.
    Embeddings are cached under args.tool_index_cache_dir via base ToolRetriever.
    """
    def __init__(self, dataset_name: str, model_path: str, cache_dir: str, args, load_cache: bool = True):
        self.dataset_name = dataset_name
        self.args = args
        
        # Prepare endpoint functions in OpenAI format
        tools_helper = RestBenchAPITools(dataset_name, args)
        endpoint_functions = tools_helper.get_endpoint_openai_functions()
        
        # Build corpus: one string per endpoint containing name, description, parameters
        corpus: List[str] = []
        corpus2tool: Dict[str, Dict[str, Any]] = {}
        for fn in endpoint_functions:
            name = fn.get('name', '')
            desc = fn.get('description', '')
            params = json.dumps(fn.get('parameters', {}), ensure_ascii=False)
            entry = f"{name}\nDescription: {desc}\nParameters: {params}"
            corpus.append(entry)
            corpus2tool[entry] = {
                "tool_name": name,
                "openai_function": {
                    "name": name,
                    "description": desc,
                    "parameters": fn.get('parameters', {})
                }
            }
        
        super().__init__(
            corpus=corpus,
            corpus2tool=corpus2tool,
            model_path=model_path,
            cache_dir=cache_dir,
            load_cache=load_cache,
            corpus_identifier=f"restbench_{dataset_name}_endpoint_functions"
        ) 


def get_restbench_tools(dataset_name: str, args) -> List[Dict[str, Any]]:
    """
    Get RestBench tools in OpenAI function format
    
    Args:
        dataset_name: 'tmdb' or 'spotify'
        args: Global args namespace
        
    Returns:
        List of tools in OpenAI function format
    """
        
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_api_details",
                "description": f"Get detailed information about a specific API endpoint. Use this to understand the parameters, request format, and response structure of an endpoint before calling it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "endpoint_name": {
                            "type": "string",
                            "description": "The name of the endpoint to get details for (e.g., 'GET /search/movie', 'POST /users/{user_id}/playlists')"
                        }
                    },
                    "required": ["endpoint_name"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "call_api",
                "description": f"Call a specific API endpoint. Make sure to use get_api_details first to understand the endpoint's parameters and format.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "endpoint_name": {
                            "type": "string",
                            "description": "The name of the endpoint to call (e.g., 'GET /search/movie', 'POST /users/{user_id}/playlists')"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE"],
                            "description": "HTTP method to use"
                        },
                        "path": {
                            "type": "string",
                            "description": "The API path (e.g., '/search/movie', '/users/123/playlists')"
                        },
                        "params": {
                            "type": "object",
                            "description": "Query parameters for GET requests or URL parameters",
                            "additionalProperties": True
                        },
                        "data": {
                            "type": "object",
                            "description": "Request body data for POST/PUT requests",
                            "additionalProperties": True
                        }
                    },
                    "required": ["endpoint_name", "method", "path"]
                }
            }
        }
    ]
    
    return tools


# Global instance for tool execution
_restbench_tools_instance = None

def execute_restbench_tool(tool_name: str, arguments: Dict[str, Any], 
                          dataset_name: str, args) -> Dict[str, Any]:
    """
    Execute a RestBench tool
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
        dataset_name: Dataset name ('tmdb' or 'spotify')
        args: Global args namespace
        
    Returns:
        Tool execution result
    """
    global _restbench_tools_instance
    
    # Initialize tools instance if not exists or dataset changed
    if _restbench_tools_instance is None or _restbench_tools_instance.dataset_name != dataset_name:
        _restbench_tools_instance = RestBenchAPITools(dataset_name, args)
    
    if tool_name == "get_api_details":
        endpoint_name = arguments.get("endpoint_name")
        if not endpoint_name:
            return {"error": "endpoint_name is required"}
        return _restbench_tools_instance.get_api_details(endpoint_name)
    
    elif tool_name == "call_api":
        endpoint_name = arguments.get("endpoint_name")
        method = arguments.get("method")
        path = arguments.get("path")
        params = arguments.get("params")
        data = arguments.get("data")
        
        if not all([endpoint_name, method, path]):
            return {"error": "endpoint_name, method, and path are required"}
        
        return _restbench_tools_instance.call_api(endpoint_name, method, path, params, data)
    else:
        # Try dynamic endpoint tool
        return _restbench_tools_instance.call_endpoint_function(tool_name, arguments)
