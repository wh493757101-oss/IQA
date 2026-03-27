import json
import logging
import os
import sys
from openai import OpenAI
from qa_tools import fetch_openapi_spec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [PLANNER] - %(message)s')

def generate_dynamic_test_matrix():
    """
      Planner：根据 API 文档和业务上下文，全自动生成原子化的测试矩阵。
    """
    logging.info(" Planner 正在读取网关 API 文档并生成测试点...")
    
    api_spec = fetch_openapi_spec()
    
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        logging.error(" 未找到 DEEPSEEK_API_KEY 环境变量，Planner 无法启动！")
        sys.exit(1)
        
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    prompt = f"""
    你是 VisionGuard 图像网关的顶级测试架构师 (Planner)。
    这是当前的接口定义：{api_spec}
    
    请你自动生成包含 4 个维度的测试场景 Prompt 矩阵：
    1. "Security_Validation" (安全与防伪装攻击：如扩展名篡改、空文件等)
    2. "Algorithm_Robustness" (算法容错：如极小分辨率、损坏的图像流等)
    3. "WhiteBox_Unit_Test" (白盒单元测试：直接 import worker.py，Mock Redis 测试不同逻辑分支)
    4. "Full_Flow_Integration" (端到端集成测试：使用 httpx 上传真实文件，拿到 task_id 后，在代码中手动调用 process_single_task 去消费队列，最后验证状态流转)
    5. "Performance_Load" (性能压测：使用 locust 编写压测脚本)
    【 最高级架构铁律 (防止底层 Agent 崩溃)】：
    你生成的每个测试用例 `prompt` 必须是 **极度单一、原子化（Atomic）的**！
    在每个 `prompt` 的末尾，你【必须】强制加上类似的约束指令：
    - "【极简铁律】：只写这一个单一测试函数！只要跑通并断言成功，立刻输出最终报告并结束任务！绝对禁止编写额外的异常处理、多格式并发测试或重写框架底层代码！"
    - "【Locust 高阶铁律】：如果压测的是带有动态参数的路径（如 task_status/{task_id}），你必须在 self.client.get 中强制使用 `name='/api/v1/task_status/[id]'` 来聚合统计！并且必须使用 `catch_response=True` 上下文，在代码中强行 `if response.status_code == 404: response.success()`，将 404 视为压测成功损耗，绝对不能让 Locust 抛出失败导致系统崩溃！"
    
    请严格以 JSON 格式返回，结构必须如下：
    {{
      "Security_Validation": [
        {{
          "name": "PDF_Upload_Spoofing",
          "prompt": "模拟上传PDF并伪装成image/jpeg...【极简铁律】：只测这一个场景，拿到422即停止思考出报告！"
        }}
      ],
      "WhiteBox_Unit_Test": [ ... ],
      ...
    }}
    不要包含任何 Markdown 代码块标记（如 ```json），只输出纯净的标准 JSON 字符串！
    """
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, 
            temperature=0.7 
        )
        
        dynamic_matrix = json.loads(response.choices[0].message.content)
        
        total_cases = sum(len(v) for v in dynamic_matrix.values())
        logging.info(f" Planner成功生成 {total_cases} 个原子化智能测试点！")
        
        return dynamic_matrix
        
    except Exception as e:
        logging.error(f" Planner 生成失败: {e}")
        sys.exit(1)
if __name__ == "__main__":
    print(json.dumps(generate_dynamic_test_matrix(), indent=2, ensure_ascii=False))