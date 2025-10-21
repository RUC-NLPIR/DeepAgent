import os
import json
import re
import numpy as np
from prompts.prompts_deepagent import BEGIN_TOOL_RESPONSE, END_TOOL_RESPONSE


def evaluate_predictions_alfworld(data, output_list, output_dir, output_metrics_path, output_metrics_overall_path):
    """
    Evaluate ALFWorld predictions using success_rate and path_rate metrics.
    
    Args:
        data: List of data items with 'success' and 'subgoals' fields
        output_list: List of model outputs
        output_dir: Directory to save results
        output_metrics_path: Path for detailed metrics file
        output_metrics_overall_path: Path for overall metrics file
    """
    all_success_rates = []
    all_path_rates = []

    for item, result in zip(data, output_list):
        if isinstance(result, str):
            output = result
        else:
            output = result.outputs[0].text

        # (1) success_rate - directly from recorded success field
        success_rate = 1.0 if item.get('success', False) else 0.0
        all_success_rates.append(success_rate)
        
        # (2) path_rate - based on subgoal completion
        subgoals_text = item.get('subgoals', '')
        if not subgoals_text:
            path_rate = 0.0
            all_path_rates.append(path_rate)
            continue
            
        # Extract subgoal patterns from the subgoals field
        subgoal_patterns = []
        for line in subgoals_text.split('\n'):
            line = line.strip()
            if line.startswith('Subgoal'):
                # Extract the pattern part after the colon
                pattern_match = re.search(r'Subgoal \d+: (.+)', line)
                if pattern_match:
                    pattern = pattern_match.group(1).strip()
                    subgoal_patterns.append(pattern)
        
        if not subgoal_patterns:
            path_rate = 0.0
            all_path_rates.append(path_rate)
            continue
            
        # Extract tool call results from output
        tool_call_result_pattern = rf"{BEGIN_TOOL_RESPONSE}(.*?){END_TOOL_RESPONSE}"
        tool_call_results = re.findall(tool_call_result_pattern, output, re.DOTALL)
        
        # Count matched subgoals
        success_subgoals = 0
        for pattern in subgoal_patterns:
            pattern_matched = False
            for result_text in tool_call_results:
                # Clean the result text and convert to lowercase for matching
                clean_result = result_text.strip().lower()
                # Check if the pattern matches in the result
                if re.search(pattern.lower(), clean_result):
                    pattern_matched = True
                    break
            if pattern_matched:
                success_subgoals += 1
        
        # Calculate path_rate
        path_rate = success_subgoals / len(subgoal_patterns) if subgoal_patterns else 0.0
        all_path_rates.append(path_rate)
        
        # Store metrics in the item
        item['metrics'] = {
            'success_rate': success_rate,
            'path_rate': path_rate,
            'success_subgoals': success_subgoals,
            'total_subgoals': len(subgoal_patterns),
            'subgoal_patterns': subgoal_patterns,
        }

    # Calculate overall metrics
    overall_metrics = {
        'total_instance': len(data),
        'success_rate': float(np.mean(all_success_rates)) if all_success_rates else 0.0,
        'path_rate': float(np.mean(all_path_rates)) if all_path_rates else 0.0,
    }
    
    print("ALFWorld Evaluation Metrics:")
    print(overall_metrics)
    
    # Save prediction results and metrics
    with open(os.path.join(output_dir, output_metrics_path), mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, output_metrics_overall_path), mode='w', encoding='utf-8') as json_file:
        json.dump(overall_metrics, json_file, indent=4, ensure_ascii=False)
