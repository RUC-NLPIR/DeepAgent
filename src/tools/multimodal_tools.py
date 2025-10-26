import base64
import time
import requests
from openai import AsyncOpenAI
import os
from pathlib import Path



async def get_vl_completion(client: AsyncOpenAI, model_name: str, image_path: str, question: str):
    """
    Test VL model's image understanding capability
    
    Args:
        image_path: Image file path
        question: Question about the image
    
    Returns:
        completion: Model response
        response_time: Response time
    """
    start_time = time.time()
    
    # Read image file and convert to base64
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: Image file not found {image_path}")
        return None, 0
    except Exception as e:
        print(f"Error reading image file: {e}")
        return None, 0
    
    # Build message containing the image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    
    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            timeout=360,
            max_tokens=5000,
        )
        end_time = time.time()
        response_time = end_time - start_time
        return completion, response_time
    except Exception as e:
        print(f"Error calling VL model: {e}")
        return None, 0


async def get_youtube_video_completion(client: AsyncOpenAI, model_name: str, youtube_id: str, question: str):
    """
    Test VL model's YouTube video understanding capability
    
    Args:
        client: OpenAI client
        model_name: Model name
        youtube_id: YouTube video ID (e.g.: L1vXCYZAYYM)
        question: Question about the video
    
    Returns:
        completion: Model response
        response_time: Response time
    """
    start_time = time.time()
    
    # Build video file path
    video_filename = f"{youtube_id}.mp4"
    video_path = f"/mnt/ali-sh-1/usr/tusen/xiaoxi/DeepAgent/data/GAIA/downloaded_files/{video_filename}"
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Unable to get video {youtube_id}, file {video_filename} does not exist")
        return None, 0
    
    try:
        # Read video file and convert to base64
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()
            base64_video = base64.b64encode(video_data).decode('utf-8')
        
        # Build message containing the video
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{base64_video}"
                        }
                    }
                ]
            }
        ]
        
        # Call the model
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,
            timeout=360,
            max_tokens=5000,
        )
        end_time = time.time()
        response_time = end_time - start_time
        return completion, response_time
        
    except Exception as e:
        print(f"Error processing YouTube video: {e}")
        return None, 0


def get_openai_function_visual_question_answering():
    return {
        "type": "function",
        "function": {
            "name": "visual_question_answering",
            "description": "Analyze images and answer questions about them using a vision-language model. This tool can help with image understanding, object recognition, scene analysis, and answering questions about visual content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_name": {
                        "type": "string",
                        "description": "The name of the image to be analyzed."
                    },
                    "question": {
                        "type": "string",
                        "description": "A clear, concise question about the image's content. For best results, ask straightforward factual questions rather than complex or multi-step reasoning questions."
                    }
                },
                "required": ["image_name", "question"]
            }
        }
    }

def get_openai_function_youtube_video_question_answering():
    return {
        "type": "function",
        "function": {
            "name": "youtube_video_question_answering",
            "description": "Analyze YouTube videos and answer questions about them. This tool can help with video understanding, content analysis, and answering questions about video content by processing the video's information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "youtube_id": {
                        "type": "string",
                        "description": "The YouTube video ID (e.g., '2vq3COPZbKo'). This is the unique identifier found in the YouTube URL after 'v='."
                    },
                    "question": {
                        "type": "string",
                        "description": "A clear, concise question about the video's content. For best results, ask straightforward factual questions rather than complex or multi-step reasoning questions."
                    }
                },
                "required": ["youtube_id", "question"]
            }
        }
    } 
