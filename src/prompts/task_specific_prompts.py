# This file contains the task-specific prompts for all agent approaches
# Providing a clear description of your task enables the agent to better understand your requirements and complete it more effectively.




def get_toolhop_prompt():
    return """You will only receive valid information if you call the exact ground-truth tool and provide completely accurate parameters.
Tools will not return valid results unless they are specifically intended for this question and your parameters are entirely correct, even if they seem related.

After you have completed the task, provide your final answer in the following format: \\boxed{your answer here}
"""







