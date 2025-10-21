import base64
import time
import requests
from openai import AsyncOpenAI
import os
from pathlib import Path



async def get_vl_completion(client: AsyncOpenAI, model_name: str, image_path: str, question: str):
    """
    测试VL模型的图片理解能力
    
    Args:
        image_path: 图片文件路径
        question: 关于图片的问题
    
    Returns:
        completion: 模型响应
        response_time: 响应时间
    """
    start_time = time.time()
    
    # 读取图片文件并转换为base64
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')
    except FileNotFoundError:
        print(f"错误：找不到图片文件 {image_path}")
        return None, 0
    except Exception as e:
        print(f"读取图片文件时出错：{e}")
        return None, 0
    
    # 构建包含图片的消息
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
        print(f"调用VL模型时出错：{e}")
        return None, 0


async def get_youtube_video_completion(client: AsyncOpenAI, model_name: str, youtube_id: str, question: str):
    """
    测试VL模型的YouTube视频理解能力
    
    Args:
        client: OpenAI客户端
        model_name: 模型名称
        youtube_id: YouTube视频ID (如: L1vXCYZAYYM)
        question: 关于视频的问题
    
    Returns:
        completion: 模型响应
        response_time: 响应时间
    """
    start_time = time.time()
    
    # 构建视频文件路径
    video_filename = f"{youtube_id}.mp4"
    video_path = f"/mnt/ali-sh-1/usr/tusen/xiaoxi/DeepAgent/data/GAIA/downloaded_files/{video_filename}"
    
    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：无法获取视频 {youtube_id}，文件 {video_filename} 不存在")
        return None, 0
    
    try:
        # 读取视频文件并转换为base64
        with open(video_path, "rb") as video_file:
            video_data = video_file.read()
            base64_video = base64.b64encode(video_data).decode('utf-8')
        
        # 构建包含视频的消息
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
        
        # 调用模型
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
        print(f"处理YouTube视频时出错：{e}")
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
