import os
import io
import regex
import pickle
import traceback
import copy
import datetime
import dateutil.relativedelta
import asyncio
from typing import Any, Dict, Optional
from contextlib import redirect_stdout
import numpy as np
import sympy
import math
from sympy import symbols, Eq, solve
from scipy import optimize


class UnsafeCodeError(Exception):
    pass

UNSAFE_PATTERNS = [
    # Prohibit dangerous imports
    r'import\s+(os|sys|subprocess|shutil|multiprocessing|threading|ctypes|_thread)',
    r'from\s+(os|sys|subprocess|shutil|multiprocessing|threading|ctypes|_thread)\s+import',
    # Prohibit dangerous built-in functions
    r'(?<!\w)(input|eval|exec|exit|quit|__import__)\s*\(',
    # Prohibit dangerous os functions (in case os is somehow imported)
    r'os\.(system|popen|fork|kill|remove|rmdir)',
]


class GenericRuntime:
    GLOBAL_DICT = {}
    LOCAL_DICT = None
    HEADERS = []

    def __init__(self):
        self._global_vars = copy.copy(self.GLOBAL_DICT)
        self._local_vars = copy.copy(self.LOCAL_DICT) if self.LOCAL_DICT else None
        
        # 在初始化时不做任何导入
        for c in self.HEADERS:
            self.exec_code(c)

    def exec_code(self, code_piece: str) -> None:
        # Security check for unsafe code patterns
        for pattern in UNSAFE_PATTERNS:
            if regex.search(pattern, code_piece):
                raise UnsafeCodeError("Your process is not safe. Execution of potentially unsafe code was blocked.")
            
        # 预处理：在执行用户代码前添加必要的导入
        imports = """
import numpy as np
import sympy
import math
from sympy import symbols, Eq, solve
x, y, z = sympy.symbols('x y z')
"""
        if regex.search(r'(\s|^)?input\(', code_piece) or regex.search(r'(\s|^)?os.system\(', code_piece):
            raise RuntimeError()
            
        # 先执行导入语句
        exec(imports, self._global_vars)
        # 再执行用户代码
        exec(code_piece, self._global_vars)
        
    def eval_code(self, expr: str) -> Any:
        return eval(expr, self._global_vars)
    
    def inject(self, var_dict: Dict[str, Any]) -> None:
        for k, v in var_dict.items():
            self._global_vars[k] = v
    
    @property
    def answer(self):
        return self._global_vars['answer']

class DateRuntime(GenericRuntime):
    GLOBAL_DICT = {
        'datetime': datetime.datetime, 
        'timedelta': dateutil.relativedelta.relativedelta,
        'relativedelta': dateutil.relativedelta.relativedelta
    }


class CustomDict(dict):
    def __iter__(self):
        return list(super().__iter__()).__iter__()

class ColorObjectRuntime(GenericRuntime):
    GLOBAL_DICT = {'dict': CustomDict}


class PythonExecutor:
    def __init__(
        self,
        runtime: Optional[Any] = None,
        get_answer_symbol: Optional[str] = None,
        get_answer_expr: Optional[str] = None,
        get_answer_from_stdout: bool = False,
        timeout_length: int = 5,
        max_concurrency: int = 8,
    ) -> None:
        self.runtime = runtime if runtime else GenericRuntime()
        self.answer_symbol = get_answer_symbol
        self.answer_expr = get_answer_expr
        self.get_answer_from_stdout = get_answer_from_stdout
        self.timeout_length = timeout_length
        self._semaphore = asyncio.Semaphore(max_concurrency)

    def process_generation_to_code(self, gens: str):
        return [g.split('\n') for g in gens]

    @staticmethod
    async def execute(
        code,
        get_answer_from_stdout = None,
        runtime = None,
        answer_symbol = None,
        answer_expr = None,
        timeout_length = 10,
    ):
        runtime = runtime if runtime else GenericRuntime()
        get_answer_from_stdout = get_answer_from_stdout if get_answer_from_stdout is not None else False
        try:
            # 确保代码是字符串而不是列表
            if isinstance(code, list):
                code = '\n'.join(code)
            
            # 移除所有前导空格
            code = code.strip()

            def run_exec_sync(code_snippet: str):
                return runtime.exec_code(code_snippet)

            async def run_eval(expr: str):
                return await asyncio.to_thread(runtime.eval_code, expr)
            
            if get_answer_from_stdout:
                program_io = io.StringIO()
                def _exec_with_capture():
                    with redirect_stdout(program_io):
                        runtime.exec_code(code)
                await asyncio.wait_for(asyncio.to_thread(_exec_with_capture), timeout=timeout_length)
                program_io.seek(0)
                result = program_io.read()
            elif answer_symbol:
                await asyncio.wait_for(run_exec_sync(code), timeout=timeout_length)
                result = runtime._global_vars[answer_symbol]
            elif answer_expr:
                await asyncio.wait_for(run_exec_sync(code), timeout=timeout_length)
                result = await asyncio.wait_for(run_eval(answer_expr), timeout=timeout_length)
            else:
                # 分离最后一行作为表达式
                code_lines = code.split('\n')
                if len(code_lines) > 1:
                    exec_code = '\n'.join(code_lines[:-1])
                    eval_code = code_lines[-1]
                    await asyncio.wait_for(run_exec_sync(exec_code), timeout=timeout_length)
                    result = await asyncio.wait_for(run_eval(eval_code), timeout=timeout_length)
                else:
                    result = await asyncio.wait_for(run_eval(code), timeout=timeout_length)
                    
            report = "Done"
            
            # 安全地处理结果
            try:
                # 尝试序列化
                pickle.dumps(result)
            except (pickle.PicklingError, TypeError):
                # 如果不能序列化，转换为字符串
                try:
                    result = str(result)
                except Exception:
                    # 如果连字符串转换都失败，返回类型信息
                    result = f"<unprintable object of type {type(result).__name__}>"
            
        except Exception as e:
            result = ''
            report = str(e)
        return result, report

    @staticmethod
    def execute_sync(
        code,
        get_answer_from_stdout = None,
        runtime = None,
        answer_symbol = None,
        answer_expr = None,
        timeout_length = 10,
    ):
        return asyncio.run(PythonExecutor.execute(
            code,
            get_answer_from_stdout=get_answer_from_stdout,
            runtime=runtime,
            answer_symbol=answer_symbol,
            answer_expr=answer_expr,
            timeout_length=timeout_length,
        ))

    async def apply(self, code):
        return (await self.batch_apply([code]))[0]

    def apply_sync(self, code):
        return asyncio.run(self.apply(code))

    @staticmethod
    def truncate(s, max_length=400):
        half = max_length // 2
        if len(s) > max_length:
            s = s[:half] + "..." + s[-half:]
        return s

    async def batch_apply(self, batch_code):
        all_code_snippets = self.process_generation_to_code(batch_code)

        async def _bounded_execute(snippet):
            async with self._semaphore:
                return await PythonExecutor.execute(
                    snippet,
                    get_answer_from_stdout=self.get_answer_from_stdout,
                    runtime=self.runtime,
                    answer_symbol=self.answer_symbol,
                    answer_expr=self.answer_expr,
                    timeout_length=self.timeout_length,
                )

        tasks = [asyncio.create_task(_bounded_execute(code_snippet)) for code_snippet in all_code_snippets]
        all_exec_results = await asyncio.gather(*tasks, return_exceptions=False)

        batch_results = []
        for code, (res, report) in zip(all_code_snippets, all_exec_results):
            # post processing
            res, report = str(res).strip(), str(report).strip()
            res, report = self.truncate(res), self.truncate(report)
            batch_results.append((res, report))
        return batch_results

    def batch_apply_sync(self, batch_code):
        return asyncio.run(self.batch_apply(batch_code))

async def execute_python_code(code: str) -> str:
    """
    Asynchronously execute Python code and return the result
    
    Args:
        code: Python code string
        
    Returns:
        execution result
    """
    try:
        # 创建执行器实例
        executor = PythonExecutor(get_answer_from_stdout=True)
        
        # 执行代码并获取结果
        result, report = await executor.apply(code)

        returned_result = "Execution result: " + result + "\nExecution status: " + report
        
        return returned_result
        
    except Exception as e:
        return f"Execution error: {str(e)}"
    

def execute_python_code_sync(code: str) -> str:
    """
    Synchronously execute Python code and return the result

    Args:
        code: Python code string

    Returns:
        execution result
    """
    try:
        executor = PythonExecutor(get_answer_from_stdout=True)
        result, report = executor.apply_sync(code)
        returned_result = "Execution result: " + result + "\nExecution status: " + report
        return returned_result
    except Exception as e:
        return f"Execution error: {str(e)}"
    

def get_openai_function_execute_python_code(file_process: bool = False):
    if file_process:
        return {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute Python code in a safe sandbox environment and return the execution results from stdout. This could help you with mathematical computations, reading tables, data analysis, and general computation-intensive tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                            "code": {
                            "type": "string",
                            "description": "The Python code to execute. Note: all data files are located in the './data/GAIA/files/' directory. Please use the appropriate relative or absolute path when accessing files.",
                            "examples": [
                                "x = 5\nprint(x * 2)",
                                "import sympy\nx = sympy.symbols('x')\nexpr = x**2 + 2*x + 1\nprint(sympy.factor(expr))",
                                "import pandas as pd\ndf = pd.read_csv('./data/GAIA/files/example.csv')\nprint(df.head())"
                            ]
                        }
                    },
                    "required": ["code"]
                }
            }
        }
    else:
        return {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute Python code in a safe sandbox environment and return the execution results from stdout. This could help you with mathematical computations, reading tables, data analysis, and general computation-intensive tasks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The Python code to execute. Avoid using unsafe functions or packages.",
                            "examples": [
                                "x = 5\nprint(x * 2)",
                                "import sympy\nx = sympy.symbols('x')\nexpr = x**2 + 2*x + 1\nprint(sympy.factor(expr))",
                                "import pandas as pd\ndf = pd.read_csv('example.csv')\nprint(df.head())"
                            ]
                        }
                    },
                    "required": ["code"]
                }
            }
        }

async def _test():
    code = "import sympy\nx = sympy.symbols('x')\nexpr = x**2 + 2*x + 1\nprint(sympy.factor(expr))"
    result = await execute_python_code(code)
    print(result)


if __name__ == '__main__':
    asyncio.run(_test())