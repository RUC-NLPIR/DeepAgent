import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import asyncio
import aiohttp

WEBSHOP_URL = "http://10.148.30.88:3000"
ACTION_TO_TEMPLATE = {
    'Description': 'description_page.html',
    'Features': 'features_page.html',
    'Reviews': 'review_page.html',
    'Attributes': 'attributes_page.html',
}

def clean_str(p):
    return p.encode().decode("unicode-escape").encode("latin1").decode("utf-8")

def tag_visible(element):
    ignore = {'style', 'script', 'head', 'title', 'meta', '[document]'}
    return (
        element.parent.name not in ignore and not isinstance(element, Comment)
    )

def webshop_text(session, page_type, query_string='', page_num=1, asin='', options={}, subpage='', webshop_url=WEBSHOP_URL, **kwargs):
    """同步版本的webshop_text函数"""
    if page_type == 'init':
        url = f'{webshop_url}/{session}'
    elif page_type == 'search':
        url = f'{webshop_url}/search_results/{session}/{query_string}/{page_num}'
    elif page_type == 'item':
        url = f'{webshop_url}/item_page/{session}/{asin}/{query_string}/{page_num}/{options}'
    elif page_type == 'item_sub':
        url = f'{webshop_url}/item_sub_page/{session}/{asin}/{query_string}/{page_num}/{subpage}/{options}'
    elif page_type == 'end':
        url = f'{webshop_url}/done/{session}/{asin}/{options}'
    
    html = requests.get(url).text
    html_obj = BeautifulSoup(html, 'html.parser')
    texts = html_obj.findAll(text=True)
    visible_texts = list(filter(tag_visible, texts))
    
    observation = ''
    option_type = ''
    options = {}
    asins = []
    cnt = 0
    prod_cnt = 0
    just_prod = 0
    
    for t in visible_texts:
        if t == '\n': 
            continue
        if t.replace('\n', '').replace('\\n', '').replace(' ', '') == '': 
            continue
        
        if t.parent.name == 'button':  # button
            if t not in ['Next >', '< Prev', 'Description', 'Features', 'Reviews', 'Attributes']:
                processed_t = f'\n[{t}] '
        elif t.parent.name == 'label':  # options
            if f"'{t}'" in url:
                processed_t = f'[[{t}]]'
            else:
                processed_t = f'[{t}]'
            options[str(t)] = option_type
        elif t.parent.get('class') == ["product-link"]: # product asins
            processed_t = f'\n[{t}] '
            if prod_cnt >= 10:
                processed_t = ''
            prod_cnt += 1
            asins.append(str(t))
            just_prod = 0
        else: # regular, unclickable text
            processed_t = '\n' + str(t) + ' '
            if cnt < 2 and page_type != 'init': processed_t = ''
            # if just_prod <= 2 and prod_cnt >= 4: processed_t = ''
            option_type = str(t)
            cnt += 1
        just_prod += 1
        observation += processed_t
    
    info = {}
    if options:
        info['option_types'] = options
    if asins:
        info['asins'] = asins
    if 'Your score (min 0.0, max 1.0)' in visible_texts:
        idx = visible_texts.index('Your score (min 0.0, max 1.0)')
        info['reward'] = float(visible_texts[idx + 1])
        observation = 'Your score (min 0.0, max 1.0): ' + (visible_texts[idx + 1])
    
    return clean_str(observation), info

class WebshopEnv:
    """Webshop环境类"""
    
    def __init__(self, webshop_url=WEBSHOP_URL):
        self.webshop_url = webshop_url
        self.sessions = {}
    
    def step(self, session, action):
        """执行一步动作"""
        done = False
        observation_ = None
        
        if action == 'reset':
            self.sessions[session] = {'session': session, 'page_type': 'init'}
        elif action.startswith('think['):
            observation = 'OK.'
        elif action.startswith('search['):
            if self.sessions[session]['page_type'] != 'init':
                observation_ = "Please click the [Back to Search] button to return to the search page before performing a new search."
                return observation_, 0.0, False
            query = action[7:-1]
            self.sessions[session] = {
                'session': session, 
                'page_type': 'search',
                'query_string': query, 
                'page_num': 1
            }
        elif action.startswith('click['):
            button = action[6:-1]
            if button == 'Buy Now':
                assert self.sessions[session]['page_type'] == 'item'
                self.sessions[session]['page_type'] = 'end'
                done = True
            elif button == 'Back to Search':
                assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
                self.sessions[session] = {'session': session, 'page_type': 'init'}
            elif button == 'Next >':
                assert False  # ad hoc page limitation
                assert self.sessions[session]['page_type'] == 'search'
                self.sessions[session]['page_num'] += 1
            elif button == '< Prev':
                assert self.sessions[session]['page_type'] in ['search', 'item_sub', 'item']
                if self.sessions[session]['page_type'] == 'search':
                    assert False
                    self.sessions[session]['page_num'] -= 1
                elif self.sessions[session]['page_type'] == 'item_sub':
                    self.sessions[session]['page_type'] = 'item'
                elif self.sessions[session]['page_type'] == 'item':
                    self.sessions[session]['page_type'] = 'search'
                    self.sessions[session]['options'] = {}
            elif button in ACTION_TO_TEMPLATE:
                assert self.sessions[session]['page_type'] == 'item'
                self.sessions[session]['page_type'] = 'item_sub'
                self.sessions[session]['subpage'] = button
            else:
                if self.sessions[session]['page_type'] == 'search':
                    assert button in self.sessions[session].get('asins', [])  # must be asins
                    self.sessions[session]['page_type'] = 'item'
                    self.sessions[session]['asin'] = button
                elif self.sessions[session]['page_type'] == 'item':
                    assert 'option_types' in self.sessions[session]
                    assert button in self.sessions[session]['option_types'], (button, self.sessions[session]['option_types'])
                    option_type = self.sessions[session]['option_types'][button]
                    if not 'options' in self.sessions[session]:
                        self.sessions[session]['options'] = {}
                    self.sessions[session]['options'][option_type] = button
                    observation_ = f'You have clicked {button}.'
        else:
            assert False
        
        observation, info = webshop_text(**self.sessions[session], webshop_url=self.webshop_url)
        if observation_:
            observation = observation_
        self.sessions[session].update(info)
        reward = info.get('reward', 0.0)
        return observation, reward, done
    

class WebshopEnvWrapper:
    """Webshop环境包装器，用于集成到DeepAgent中"""
    
    def __init__(self, batch_size=250, webshop_url=WEBSHOP_URL):
        self.batch_size = batch_size
        self.env = WebshopEnv(webshop_url=webshop_url)
        self.initial_obs_list = []
        
        # 初始化所有session
        for i in range(batch_size):
            obs, _, _ = self.env.step(f'fixed_{i}', 'reset')
            self.initial_obs_list.append(obs)
    
    def step_action(self, session_id, action_name, arguments):
        """执行动作并返回结果"""
        try:
            if action_name == 'reset':
                action = 'reset'
            elif action_name == 'think':
                action = f'think[{arguments.get("thought", "")}]'
            elif action_name == 'search':
                action = f'search[{arguments.get("query", "")}]'
            elif action_name == 'click':
                action = f'click[{arguments.get("button", "")}]'
            else:
                return f"Invalid action: {action_name}", 0.0, False
            
            observation, reward, done = self.env.step(f"fixed_{session_id}", action)
            return observation, reward, done
            
        except Exception as e:
            return f"Invalid action: {str(e)}", 0.0, False
    

def get_webshop_function_definitions():
    """获取Webshop环境的函数定义"""
    return [
        # {
        #     "name": "reset",
        #     "description": "Reset the webshop environment to initial state",
        #     "parameters": {
        #         "type": "object",
        #         "properties": {},
        #         "required": []
        #     }
        # },
        {
            "name": "search",
            "description": "Search for products with the given query. Use this only if a [Search] button appears in the tool call result. Note: If you wish to search and there's no [Search] button, click the [Back to Search] button instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for products"
                    }
                },
                "required": ["query"]
            }
        },
        {
            "name": "click",
            "description": "Click on a button or product. Use this only if a [button] is present in the tool call result. When you have identified the most suitable product, click the [Buy Now] button on its product page to finish the shopping task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "button": {
                        "type": "string",
                        "description": "Name of the button or product ASIN, don't add '[]' around the button name"
                    }
                },
                "required": ["button"]
            }
        }
    ]
