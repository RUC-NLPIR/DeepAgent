import os
import json
import numpy as np


def evaluate_predictions_webshop(data, output_list, output_dir, output_metrics_path, output_metrics_overall_path):
    """
    Evaluate WebShop predictions using path_score, success_rate, and fail_rate metrics.
    
    Args:
        data: List of data items with 'success' and other fields
        output_list: List of model outputs
        output_dir: Directory to save results
        output_metrics_path: Path for detailed metrics file
        output_metrics_overall_path: Path for overall metrics file
    """
    all_path_scores = []
    all_success_rates = []

    for item, result in zip(data, output_list):
        if isinstance(result, str):
            output = result
        else:
            output = result.outputs[0].text

        # (1) path_score - reward from the environment (from 0.0 to 1.0)
        # This represents how well the agent performed in the task
        path_score = item.get('reward', 0.0)
        all_path_scores.append(path_score)
        
        # (2) success_rate - whether the task was fully completed (0.0 or 1.0)
        success_rate = 1.0 if path_score == 1.0 else 0.0
        all_success_rates.append(success_rate)
        
        # Store metrics in the item
        item['metrics'] = {
            'success_rate': success_rate,
            'path_score': path_score,
        }

    # Calculate overall metrics
    overall_metrics = {
        'total_instance': len(data),
        'success_rate': float(np.mean(all_success_rates)) if all_success_rates else 0.0,
        'path_score': float(np.mean(all_path_scores)) if all_path_scores else 0.0,
    }
    
    print("WebShop Evaluation Metrics:")
    print(f"Total instances: {overall_metrics['total_instance']}")
    print(f"Average path_score: {overall_metrics['path_score']:.4f}")
    print(f"Success rate: {overall_metrics['success_rate']:.4f}")
    
    # Save prediction results and metrics
    with open(os.path.join(output_dir, output_metrics_path), mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

    with open(os.path.join(output_dir, output_metrics_overall_path), mode='w', encoding='utf-8') as json_file:
        json.dump(overall_metrics, json_file, indent=4, ensure_ascii=False)
    
    return overall_metrics
