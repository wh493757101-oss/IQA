import requests
import redis
import json
import logging
import subprocess
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [QA_TOOLS] - %(message)s')

def fetch_openapi_spec(base_url="http://127.0.0.1:8000"):
    """
    工具 1：读取并精简 FastAPI 的 Swagger 文档，提取接口信息喂给 LLM。
    """
    try:
        res = requests.get(f"{base_url}/openapi.json", timeout=5)
        res.raise_for_status()
        openapi_data = res.json()
        

        endpoints_summary = {}
        for path, methods in openapi_data.get("paths", {}).items():
            endpoints_summary[path] = list(methods.keys())
            
        logging.info(f"Successfully fetched API spec. Found endpoints: {endpoints_summary}")
        return json.dumps({
            "title": openapi_data["info"]["title"],
            "endpoints": endpoints_summary
        }, indent=2)
        
    except Exception as e:
        error_msg = f"Failed to fetch OpenAPI spec: {str(e)}"
        logging.error(error_msg)
        return error_msg
def query_redis_backend(task_id: str, host='localhost', port=6379, db=0):
    """
    工具 2：绕过 HTTP 网关，直接穿透到 Redis 查询任务底层的真实状态和耗时。
    """
    try:
 
        r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        
        redis_key = f"iqa:task:{task_id}"
        task_data = r.hgetall(redis_key)
        
        if not task_data:
            return f"Error: Task ID {task_id} not found in Redis."
            
        logging.info(f"Fetched raw Redis data for task {task_id[:8]}...")
        return json.dumps(task_data, indent=2)
        
    except Exception as e:
        error_msg = f"Redis connection or query failed: {str(e)}"
        logging.error(error_msg)
        return error_msg
def execute_pytest_code(code_string: str) -> str:
    """
    工具 3：执行沙箱。将大模型生成的代码写入临时文件，运行 Pytest，并返回控制台的输出给模型分析。
    """
    file_name = "temp_agent_test.py"
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(code_string)
            
        logging.info(f"Preparing to execute AI-generated test script ({len(code_string)} bytes)...")
        

        pytest_args = [
            "pytest", file_name, 
            "-v", 
            "-s",                 
            "--tb=short",         
            "--disable-warnings",
            "--cov=main",         
            "--cov=worker",       
            "--cov-report=term-missing" 
        ]
        
        result = subprocess.run(
            pytest_args, 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        
        execution_summary = (
            f"Exit Code: {result.returncode}\n"
            f"--- STDOUT ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}"
        )
        
        if result.returncode == 0:
            logging.info("Agent script PASSED (Coverage generated)!")
        else:
            logging.warning("Agent script FAILED. Waiting for Agent to analyze...")
            
        return execution_summary

    except subprocess.TimeoutExpired:
        error_msg = "Execution Error: Script timed out after 30 seconds. Did you forget to add 'await asyncio.sleep()', or write an infinite while loop?"
        logging.error(error_msg)
        return error_msg
        
    except Exception as e:
        error_msg = f"System Error during execution: {str(e)}"
        logging.error(error_msg)
        return error_msg
        
    finally:
        if os.path.exists(file_name):
            os.remove(file_name)

def execute_locust_load_test(code_string: str) -> str:
    """
    工具 4：让 AI 编写 Locust 压测脚本，并在沙箱中拉起并发打流。
    """
    file_name = "locustfile.py"
    try:
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(code_string)
            
        logging.info(" 正在启动 Locust 性能压测...")
        
        locust_args = [
            "locust", "-f", file_name, 
            "--headless", "-u", "100", "-r", "20", "--run-time", "10s"
        ]
        
        result = subprocess.run(locust_args, capture_output=True, text=True, timeout=30)
        
        execution_summary = (
            f"Locust Exit Code: {result.returncode}\n"
            f"--- 压测输出与 QPS 报告 ---\n{result.stdout}\n"
            f"--- STDERR ---\n{result.stderr}"
        )
        
        if result.returncode == 0 or "Aggregated" in result.stdout:
            logging.info(" Locust 压测执行完毕，已生成性能报告！")
        else:
            logging.warning(" Locust 压测执行异常...")
            
        return execution_summary

    except subprocess.TimeoutExpired:
        return "Execution Error: Locust run timed out. Please check if the script blocks."
    except Exception as e:
        return f"System Error during execution: {str(e)}"
    finally:
        if os.path.exists(file_name):
            os.remove(file_name)

if __name__ == "__main__":
    print("\n--- Testing Tool 3: Execute Pytest Code ---")
    
    fake_ai_code = """
import pytest

def test_always_fail():
    print("This is AI running a test!")
    assert 1 == 2, "Math is broken"
    """
    
    execution_result = execute_pytest_code(fake_ai_code)
    print("\n[Return to LLM]:\n")
    print(execution_result)