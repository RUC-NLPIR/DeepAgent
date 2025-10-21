import requests
from typing import List, Dict, Union
from urllib.parse import urljoin

class WebParserClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化Web解析器客户端
        
        Args:
            base_url: API服务器的基础URL，默认为本地测试服务器
        """
        self.base_url = base_url.rstrip('/')
        
    def parse_urls(self, urls: List[str]) -> List[Dict[str, Union[str, bool]]]:
        """
        发送URL列表到解析服务器并获取解析结果
        
        Args:
            urls: 需要解析的URL列表
            
        Returns:
            解析结果列表
            
        Raises:
            requests.exceptions.RequestException: 当API请求失败时
        """
        endpoint = urljoin(self.base_url, "/parse_urls")
        response = requests.post(endpoint, json={"urls": urls})
        response.raise_for_status()  # 如果响应状态码不是200，抛出异常
        
        return response.json()["results"]


# 使用示例
if __name__ == "__main__":
    # 创建客户端实例（如果API运行在其他服务器上，请修改base_url）
    client = WebParserClient("http://xxxx")
    
    # 测试URL列表
    test_urls = [
        "http://xxxx",
    ]
    
    try:
        # 调用API解析URL
        results = client.parse_urls(test_urls)
        print(results[1]['content'])

    except requests.exceptions.RequestException as e:
        print(f"API调用失败: {str(e)}") 