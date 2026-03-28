import json
import logging
import datetime
import os
from openai import OpenAI
from qa_tools import fetch_openapi_spec, query_redis_backend, execute_pytest_code, execute_locust_load_test


logging.basicConfig(level=logging.INFO, format='%(asctime)s - [QA_AGENT] - %(message)s')

class MarkdownReporter:
    def __init__(self, test_name="QA_Task", output_dir=None):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"Report_{test_name}_{timestamp}.md"
        

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.filepath = os.path.join(output_dir, filename)
        else:
            self.filepath = os.path.join(os.getcwd(), filename)
        
        with open(self.filepath, "w", encoding="utf-8") as f:
            f.write(f"#  VisionGuard AI QA Agent 诊断报告 ({test_name})\n\n")
            f.write(f"**任务执行时间**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n")
        logging.info(f" 已成功创建并打开测试报告文档: {self.filepath}")
    def append_text(self, title: str, content: str):
        if not content: return
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"###  {title}\n\n{content}\n\n")

    def append_code(self, title: str, code: str, lang: str = "python"):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"###  {title}\n\n```{lang}\n{code}\n```\n\n")

    def append_result(self, title: str, result: str):
        with open(self.filepath, "a", encoding="utf-8") as f:
            f.write(f"###  {title}\n\n```text\n{result}\n```\n\n---\n")

api_key = os.environ.get("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError(" Not Found DEEPSEEK_API_KEY ")

client = OpenAI(
    api_key=api_key, 
    base_url="https://api.deepseek.com"
)

AVAILABLE_TOOLS = {
    "fetch_openapi_spec": fetch_openapi_spec,
    "query_redis_backend": query_redis_backend,
    "execute_pytest_code": execute_pytest_code,
    "execute_locust_load_test": execute_locust_load_test
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "fetch_openapi_spec",
            "description": "获取网关接口定义。当你不确定路径或传参时调用。",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_redis_backend",
            "description": "查询底层任务状态。用于排查是脚本Bug还是后台Worker崩溃。",
            "parameters": {
                "type": "object",
                "properties": {"task_id": {"type": "string"}},
                "required": ["task_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_pytest_code",
            "description": "在沙箱中执行 Pytest 测试脚本。必须传入完整可运行的Python代码。",
            "parameters": {
                "type": "object",
                "properties": {"code_string": {"type": "string"}}
            },
            "required": ["code_string"]
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_locust_load_test",
            "description": "执行 Locust 性能压测。传入完整的 locustfile.py 字符串代码（需包含 HttpUser 类）。工具会模拟 100 个并发用户压测 10 秒，并返回 QPS、失败率和 P99 延迟报告。",
            "parameters": {
                "type": "object",
                "properties": {"code_string": {"type": "string"}}
            },
            "required": ["code_string"]
        }
    }
]

SYSTEM_PROMPT = """
你是一个顶级的自动化测试开发架构师（QA Agent）。被测系统是一个异步解耦的图像评测网关（VisionGuard）。
【核心机制】（极度重要！）：
1. 提交接口 `/api/v1/submit_eval` 接收的是 `multipart/form-data`。你必须使用 `requests.post(url, files={"pred_file": ...})` 来上传文件。绝对不要发送 JSON 格式的数据！
2. 网关目前拥有极度强悍的安全防御（扩展名、大小、魔数、最小分辨率 10x10 校验）。如果是不合法文件、伪装文件或极限小文件，它会直接返回 HTTP 400, 422 或 413。
3. 真实计算在后台。对于合法请求，网关瞬间返回 202 和 task_id。你必须编写带有 `time.sleep(1)` 的轮询逻辑去请求 `/api/v1/task_status/{task_id}`。

【沙箱代码编写铁律】（务必遵守！）：
1. 你的主函数必须以 `test_` 开头（例如 `def test_upload_pdf():`），绝对不要只写 `if __name__ == "__main__":` 的普通脚本！因为这是在 Pytest 沙箱中运行的。

【任务终结与判定原则】（最高优先级指令！）：
1. 如果你的测试用例是恶意的（比如空文件、伪装PDF、损坏的流、1x1像素），而网关正确返回了 400/413/422，**这说明网关成功防御了你的攻击！** 此时，请直接在代码中使用 `assert True` 让测试通过！
2. 一旦确认网关防御成功，**立刻停止调用 `execute_pytest_code` 工具！** 不要纠结，不要反复修改代码！直接在对话中输出你最终的 Markdown《诊断报告》，给出“测试通过”的结论。
3. 只有当恶意文件收到了 202，才说明存在漏洞，此时也请停止工具调用并输出包含“发现漏洞”的报告。
"""

def run_agent(test_name: str, user_instruction: str, output_dir: str = None) -> dict:
    report = MarkdownReporter(test_name=test_name, output_dir=output_dir)
    report.append_text(" 测试任务目标", user_instruction)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_instruction}
    ]
    
    max_loops = 10
    status = "Unknown"
    
    for loop in range(max_loops):
        logging.info(f"--- {test_name}：开始第 {loop + 1} 轮思考 ---")
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        messages.append(response_message)
        
        if response_message.content:
            report.append_text(f"第 {loop + 1} 轮：Agent 分析与推理", response_message.content)
        
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except Exception as e:
                    logging.error(f" 生成非法 JSON 格式: {e}")
                    
                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": f"系统错误: 你的参数 JSON 格式不合法 ({e})。请确保你生成的 Python 代码字符串正确转义了引号和换行符，并保持代码极简！"
                    })
                    continue 
                
                logging.info(f"  Agent 调用工具: {function_name}")
                
                if function_name == "execute_pytest_code":
                    report.append_code(f"执行自动生成的测试脚本", function_args.get("code_string", ""), "python")
                else:
                    report.append_code(f"调用辅助工具: {function_name}", json.dumps(function_args, indent=2), "json")
                
                function_response = AVAILABLE_TOOLS[function_name](**function_args)
                report.append_result(f"工具返回结果 (Traceback / Status)", str(function_response))
                
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response)
                })
        else:
            logging.info(f"  {test_name} 测试任务完成！报告已封存。")
            final_content = response_message.content
            report.append_text("  最终诊断报告", final_content)
            
            if "发现漏洞" in final_content or "Bug" in final_content or "缺陷" in final_content:
                status = " 发现漏洞"
            elif "未发现漏洞" in final_content or "未发现安全漏洞" in final_content or "测试通过" in final_content:
                status = " 测试通过"
            else:
                status = " 需人工复核"
            break
            
    if loop == max_loops - 1:
        status = " Agent 运行超时"
        report.append_text(" 异常终止", "达到最大思考循环次数，Agent 被强制终止。")

    return {
        "test_name": test_name,
        "status": status,
        "report_file": report.filepath
    }
if __name__ == "__main__":
    # 保留直接运行的能力，方便单点调试
    task = "请帮我写一个测试用例：模拟用户上传了一个叫 document.pdf 的非图片文件，验证网关的容错性。"
    run_agent("Debug_PDF_Upload", task)