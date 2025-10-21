import json
import os
import re
import logging
from typing import List, Dict, Any, Tuple
import numpy as np


def extract_api_calls_from_output(output: str) -> List[Dict[str, Any]]:
    """
    Extract API calls from DeepAgent output
    
    Args:
        output: The model output text
        
    Returns:
        List of extracted API calls with metadata
    """
    api_calls = []
    
    # Look for tool call patterns - be more flexible with whitespace
    tool_call_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
    tool_calls = re.findall(tool_call_pattern, output, re.DOTALL)
    
    for i, tool_call in enumerate(tool_calls):
        try:
            # Parse the tool call JSON
            tool_data = json.loads(tool_call)
            tool_name = tool_data.get("name")
            
            # Check for RestBench tools: get_api_details, call_api, or dynamic endpoint tools
            if tool_name in ["get_api_details", "call_api"]:
                api_calls.append({
                    "step": i + 1,
                    "tool_name": tool_name,
                    "arguments": tool_data.get("arguments", {}),
                    "raw_call": tool_call
                })
            else:
                # This is likely a dynamic endpoint tool (e.g., get_search_movie, post_users_user_id_playlists)
                # We need to map it back to the actual endpoint name
                api_calls.append({
                    "step": i + 1,
                    "tool_name": "dynamic_endpoint",
                    "arguments": tool_data.get("arguments", {}),
                    "raw_call": tool_call,
                    "dynamic_tool_name": tool_name  # Store the actual tool name for mapping
                })
        except json.JSONDecodeError:
            # Skip malformed JSON
            continue
    
    return api_calls


def extract_endpoint_usage(api_calls: List[Dict[str, Any]]) -> List[str]:
    """
    Extract the endpoints that were actually used
    
    Args:
        api_calls: List of API calls from the model
        
    Returns:
        List of endpoint names that were used
    """
    endpoints_used = []
    
    for call in api_calls:
        if call["tool_name"] == "call_api":
            endpoint_name = call["arguments"].get("endpoint_name")
            if endpoint_name:
                endpoints_used.append(endpoint_name)
        elif call["tool_name"] == "get_api_details":
            # Also count endpoints that were queried for details
            endpoint_name = call["arguments"].get("endpoint_name")
            if endpoint_name:
                endpoints_used.append(endpoint_name)
        elif call["tool_name"] == "dynamic_endpoint":
            # Handle dynamic endpoint tools (e.g., get_search_movie -> GET /search/movie)
            dynamic_tool_name = call.get("dynamic_tool_name", "")
            if dynamic_tool_name:
                # Convert dynamic tool name back to endpoint format
                endpoint_name = _convert_dynamic_tool_to_endpoint(dynamic_tool_name)
                if endpoint_name:
                    endpoints_used.append(endpoint_name)
    
    return list(set(endpoints_used))  # Remove duplicates


def _endpoint_to_dynamic_tool_name(endpoint: str) -> str:
    """
    Convert a standard endpoint format (e.g., 'GET /users/{user_id}/playlists')
    to the dynamic tool name used by the agent (e.g., 'post_users_user_id_playlists').

    The transformation mirrors RestBenchAPITools._normalize_endpoint_name:
    - Lowercase the full string
    - Remove curly braces around path params
    - Replace spaces and '/' with '_'
    - Collapse duplicate underscores
    - Ensure result starts with a letter (prefix 'rb_' if needed)
    """
    if not endpoint:
        return ""
    name = endpoint.strip().lower()
    # Remove braces
    name = name.replace('{', '').replace('}', '')
    # Replace spaces and slashes with underscores
    name = name.replace(' ', '_').replace('/', '_')
    # Collapse duplicate underscores
    while '__' in name:
        name = name.replace('__', '_')
    # Ensure starts with a letter
    if name and not name[0].isalpha():
        name = 'rb_' + name
    return name


def extract_used_tool_names(api_calls: List[Dict[str, Any]]) -> List[str]:
    """
    Extract dynamic tool names actually used by the model.
    - For call_api/get_api_details: convert provided endpoint_name to dynamic tool name
    - For dynamic endpoint tools: use the tool name directly
    """
    tool_names: List[str] = []
    for call in api_calls:
        tool_name = call.get("tool_name")
        if tool_name == "call_api" or tool_name == "get_api_details":
            endpoint_name = call.get("arguments", {}).get("endpoint_name")
            if endpoint_name:
                tool_names.append(_endpoint_to_dynamic_tool_name(endpoint_name))
        else:
            # Treat any other tool as dynamic endpoint tool
            dyn_name = call.get("dynamic_tool_name") or tool_name
            if dyn_name:
                tool_names.append(dyn_name)
    # Deduplicate
    return list(dict.fromkeys(tool_names))


def _convert_dynamic_tool_to_endpoint(dynamic_tool_name: str) -> str:
    """
    Convert a dynamic tool name back to the original endpoint format
    
    Args:
        dynamic_tool_name: e.g., 'get_search_movie', 'post_users_user_id_playlists'
        
    Returns:
        Original endpoint format: e.g., 'GET /search/movie', 'POST /users/{user_id}/playlists'
    """
    if not dynamic_tool_name:
        return ""
    
    # Remove common prefixes
    tool_name = dynamic_tool_name.lower()
    
    # Map common HTTP method prefixes
    method_mapping = {
        'get_': 'GET /',
        'post_': 'POST /',
        'put_': 'PUT /',
        'delete_': 'DELETE /',
        'patch_': 'PATCH /'
    }
    
    method = "GET /"  # Default
    for prefix, http_method in method_mapping.items():
        if tool_name.startswith(prefix):
            method = http_method
            tool_name = tool_name[len(prefix):]
            break
    
    # # Convert underscores back to slashes
    # params_suffix = ['id', 'number', 'type', 'window']
    # for suffix in params_suffix:
    #     if f'_{suffix}' in tool_name:
    #         tool_name = tool_name.replace(f'_{suffix}', f'*{suffix}')
    
    # # 将'_'替换为'/'，将'*'替换为'_'
    # path = tool_name.replace('_', '/').replace('*', '_')
    
    # # 将path按'/'split，遍历每一项，若包含'_'，则在这一项前后加'{}'，再将path组合起来
    # path_parts = path.split('/')
    # for i, part in enumerate(path_parts):
    #     if '_' in part:
    #         path_parts[i] = '{' + part + '}'
    # path = '/'.join(path_parts)
    
    return method + tool_name


def _endpoint_matches(used_endpoint: str, required_endpoint: str) -> bool:
    """
    Check if a used endpoint matches a required endpoint
    
    Args:
        used_endpoint: The endpoint that was actually used
        required_endpoint: The endpoint that was required
        
    Returns:
        bool: True if they match
    """
    # Normalize endpoints for comparison
    used = used_endpoint.strip().upper()
    required = required_endpoint.strip().upper()
    
    # Direct match
    if used == required:
        return True
    
    # Handle path parameters (e.g., /users/{user_id}/playlists)
    # Replace path parameters with wildcards for comparison
    used_normalized = re.sub(r'\{[^}]+\}', r'[^/]+', used)
    required_normalized = re.sub(r'\{[^}]+\}', r'[^/]+', required)
    
    # Check if normalized versions match
    if re.match(required_normalized, used_normalized):
        return True
    
    return False



def evaluate_restbench_predictions(
    data: List[Dict[str, Any]], 
    output_list: List[str], 
    output_dir: str, 
    output_metrics_path: str, 
    output_metrics_overall_path: str
) -> Dict[str, Any]:
    """
    Evaluate RestBench predictions using path_rate and success_rate metrics.
    
    Args:
        data: List of data items with 'solution' and other fields
        output_list: List of model outputs
        output_dir: Directory to save results
        output_metrics_path: Path for detailed metrics file
        output_metrics_overall_path: Path for overall metrics file
        
    Returns:
        Dictionary containing overall metrics
    """
    # Add model outputs to data
    for item, output in zip(data, output_list):
        item['output'] = output
    
    # Compute metrics
    path_rate = 0.0
    success_rate = 0.0
    
    # Add metrics to each item
    for item in data:
        api_calls = extract_api_calls_from_output(item.get('output', ''))
        used_tool_names = extract_used_tool_names(api_calls)
        
        solution_endpoints = item.get('solution', [])
        correct_count = 0
        for required in solution_endpoints:
            required_tool_name = _endpoint_to_dynamic_tool_name(required)
            if required_tool_name in used_tool_names:
                correct_count += 1
        path_rate_item = (correct_count / len(solution_endpoints)) if solution_endpoints else 0.0
        
        path_rate += path_rate_item
        success_rate += 1.0 if path_rate_item == 1.0 else 0.0

        item['metrics'] = {
            'path_rate': path_rate_item,
            'success_rate': 1.0 if path_rate_item == 1.0 else 0.0,
            'endpoints_used': used_tool_names,
            'api_calls_count': len(api_calls)
        }
    
    # Calculate overall metrics
    overall_metrics = {
        'total_instance': len(data),
        'path_rate': path_rate / len(data),
        'success_rate': success_rate / len(data),
        'avg_api_calls_per_instance': np.mean([item['metrics']['api_calls_count'] for item in data]) if data else 0.0
    }
    
    print("RestBench Evaluation Metrics:")
    print(f"Total instances: {overall_metrics['total_instance']}")
    print(f"Path rate: {overall_metrics['path_rate']:.4f}")
    print(f"Success rate: {overall_metrics['success_rate']:.4f}")
    print(f"Average API calls per instance: {overall_metrics['avg_api_calls_per_instance']:.2f}")
    
    # Save prediction results and metrics
    with open(os.path.join(output_dir, output_metrics_path), mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, output_metrics_overall_path), mode='w', encoding='utf-8') as json_file:
        json.dump(overall_metrics, json_file, indent=4, ensure_ascii=False)
    
    return overall_metrics


def analyze_restbench_performance(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze RestBench performance in detail
    
    Args:
        data: List of evaluated data items
        
    Returns:
        Dictionary containing detailed analysis
    """
    if not data:
        return {}
    
    # Analyze endpoint usage patterns
    all_endpoints_used = []
    for item in data:
        all_endpoints_used.extend(item['metrics']['endpoints_used'])
    
    endpoint_frequency = {}
    for endpoint in all_endpoints_used:
        endpoint_frequency[endpoint] = endpoint_frequency.get(endpoint, 0) + 1
    
    # Sort by frequency
    most_used_endpoints = sorted(endpoint_frequency.items(), key=lambda x: x[1], reverse=True)
    
    # Analyze failure patterns
    failed_items = [item for item in data if item['metrics']['success_rate'] == 0.0]
    path_failed_items = [item for item in data if item['metrics']['path_rate'] == 0.0]
    
    analysis = {
        'total_items': len(data),
        'successful_items': len([item for item in data if item['metrics']['success_rate'] == 1.0]),
        'path_successful_items': len([item for item in data if item['metrics']['path_rate'] == 1.0]),
        'most_used_endpoints': most_used_endpoints[:10],  # Top 10
        'failed_items_count': len(failed_items),
        'path_failed_items_count': len(path_failed_items),
        'avg_api_calls': np.mean([item['metrics']['api_calls_count'] for item in data]),
        'median_api_calls': np.median([item['metrics']['api_calls_count'] for item in data])
    }
    
    return analysis


def print_restbench_analysis(analysis: Dict[str, Any]):
    """Print detailed RestBench analysis"""
    print("\n" + "="*50)
    print("RestBench Detailed Analysis")
    print("="*50)
    print(f"Total items: {analysis['total_items']}")
    print(f"Successful items: {analysis['successful_items']} ({analysis['successful_items']/analysis['total_items']*100:.1f}%)")
    print(f"Path successful items: {analysis['path_successful_items']} ({analysis['path_successful_items']/analysis['total_items']*100:.1f}%)")
    print(f"Failed items: {analysis['failed_items_count']} ({analysis['failed_items_count']/analysis['total_items']*100:.1f}%)")
    print(f"Path failed items: {analysis['path_failed_items_count']} ({analysis['path_failed_items_count']/analysis['total_items']*100:.1f}%)")
    print(f"Average API calls per item: {analysis['avg_api_calls']:.2f}")
    print(f"Median API calls per item: {analysis['median_api_calls']:.2f}")
    
    print("\nMost used endpoints:")
    for endpoint, count in analysis['most_used_endpoints'][:5]:
        print(f"  {endpoint}: {count} times")
    
    print("="*50)
