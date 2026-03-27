import time
import os
import logging
import datetime
from run_qa_agent import run_agent
import sys
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CI_PIPELINE] - %(message)s')


PROMPT_MATRIX = {
    "Security_Validation": [
        {
            "name": "PDF_Upload_Spoofing",
            "prompt": "模拟用户上传了一个非图片文件，并尝试将 Content-Type 伪装成 image/jpeg，验证网关的安全拦截能力。"
        },
        {
            "name": "Empty_File_Attack",
            "prompt": "编写测试脚本，模拟用户上传一个 0 字节的空文件，验证网关是否会抛出 422/400 错误正确拒绝，防止后端 OOM。"
        }
    ],
    "Algorithm_Robustness": [
        {
            "name": "1x1_Pixel_Extreme",
            "prompt": "利用 numpy 或 base64 生成并上传一张只有 1x1 像素的极小图片进行评测，观察它在网关中是返回 202，还是在轮询状态时触发底层的计算失败。"
        },
        {
            "name": "Invalid_Base64_Payload",
            "prompt": "不要上传真实图片，故意上传一段损坏的图片二进制流（比如包含乱码），看看 OpenCV 解码时是否会把你的 worker 进程搞崩。注意：如果网关直接返回 422 成功拦截了它，说明网关防御极强，脏数据根本进不到 Worker，请直接认定测试通过，输出 ✅ 防御成功的报告，绝对不要纠结和死磕！"
        }
    ],
    "WhiteBox_Unit_Test": [
        {
            "name": "Worker_NR_Branch_Logic",
           "prompt": "这是一个白盒单元测试任务。请在沙箱中直接 `from worker import process_single_task`。请构造一个合法的 task_data 字典（注意：必须包含 'task_id', 'filename', 'pred_b64' 这三个 key，不要 gt_b64），使用 `unittest.mock.MagicMock` 来 mock Redis 客户端并调用函数。验证代码是否正确进入了 NR-IQA 分支。"
        },
        {
            "name": "Full_Flow_Integration",
            "prompt": "请编写一个完整的端到端异步集成测试。使用 `httpx.AsyncClient` 配合 `main.py` 的 app 对象，模拟上传一张真实图片，拿到 task_id 后，立刻去 Redis 队列里把任务取出来扔给 `worker.py` 执行，最后再次请求 task_status 接口验证状态是否变为了 completed。输出整个链路的代码覆盖率。"
        }
    ],
    "Performance_Load": [
        {
            "name": "Locust_Gateway_Spike",
            "prompt": "这是一项性能打流任务。请编写一个标准的 locustfile.py。在 `HttpUser` 中编写任务，不断向 `/api/v1/submit_eval` 接口发送 multipart/form-data 请求。为了减少网络带宽瓶颈干扰，请在代码中用 base64 生成一个极其微小的合法 10x10 像素 PNG 图片进行上传。写好后调用 execute_locust_load_test 工具执行压测，并根据返回的汇总表格，在报告中分析网关的 QPS 吞吐量和 P99 响应延迟。"
        }
    ]
}

def generate_master_report(results: list, output_dir: str):
    # 将总表也保存在这个专属目录下
    master_file = os.path.join(output_dir, "🏆_VisionGuard_CI_Master_Report.md")
    
    with open(master_file, "w", encoding="utf-8") as f:
        f.write("#  VisionGuard CI/CD 自动化检测\n\n")
        f.write("> **执行状态**: 已完成\n> \n")
        f.write("> **Agent引擎**: DeepSeek QA Agent\n\n")
        
        f.write("##  模块体检结果汇总\n\n")
        f.write("| 测试维度 | 用例名称 | Agent执行结果 | 详细探案报告 |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        
        for res in results:
            filename = os.path.basename(res['report_file'])
            f.write(f"| **{res['category']}** | `{res['test_name']}` | {res['status']} | [点击查看明细](./{filename}) |\n")
            
        f.write("\n---\n*注：如果出现「发现漏洞」，请点击详细报告查看 Agent 的报错堆栈与架构修复建议。*\n")
    
    logging.info(f"\n\n 执行完毕！报告已生成: {master_file}\n")

def send_im_alert(vuln_count: int, report_dir: str, details: list, run_url: str):
    """向群机器人发送报警"""
    webhook_url = os.environ.get("IM_WEBHOOK_URL")
    
    if not webhook_url:
        logging.warning("未配置 IM_WEBHOOK_URL 环境变量，跳过发送报警。")
        return

    content = (
        f" **VisionGuard 安全警报** \n\n"
        f"QA Agent 刚刚在 GitHub CI 检测中打崩了网关！\n"
        f"- **发现漏洞数**: {vuln_count} 个高危漏洞！\n"
        f"- **被攻破模块**: {', '.join(details)}\n\n"
        f" [点击此处下载完整体检报告]({run_url})"
    )
    
    payload = {
        "msg_type": "text",
        "content": {"text": content}
    }
    
    try:
        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            logging.info(" [报警已发送成功！]")
        else:
            logging.error(f"报警发送失败: {response.text}")
    except Exception as e:
        logging.error(f"Webhook 请求异常: {e}")


def run_pipeline():
    logging.info(" 启动 VisionGuard CI 流水线...")
    
    base_report_dir = "test_reports"
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_dir = os.path.join(base_report_dir, f"run_{run_timestamp}")
    
    os.makedirs(current_run_dir, exist_ok=True)
    logging.info(f" 已创建本次构建的专属工件目录: {current_run_dir}")
    
    all_results = []
    vuln_count = 0
    vuln_modules = []
    
    for category, tests in PROMPT_MATRIX.items():
        logging.info(f"\n>>> 开始检测: [{category}] <<<")
        
        for test in tests:
            test_name = test["name"]
            instruction = test["prompt"]
            logging.info(f" 正在唤醒 Agent 执行用例: {test_name}")
            
            try:
                result = run_agent(test_name=test_name, user_instruction=instruction, output_dir=current_run_dir)
                result["category"] = category
                all_results.append(result)
                
                if "漏洞" in result["status"] or "Bug" in result["status"] or "超时" in result["status"]:
                    vuln_count += 1
                    vuln_modules.append(test_name)
                
                time.sleep(3) 
            except Exception as e:
                logging.error(f" Agent 执行 {test_name} 时内部崩溃: {e}")
                all_results.append({
                    "category": category,
                    "test_name": test_name,
                    "status": " 框架级错误",
                    "report_file": "None"
                })
                vuln_count += 1
                vuln_modules.append(f"{test_name}(框架崩溃)")


    generate_master_report(all_results, output_dir=current_run_dir)

    if vuln_count > 0:
        logging.error(f" 检测到 {vuln_count} 个高危漏洞/异常！")
        repo_name = os.environ.get("GITHUB_REPOSITORY", "your-repo")
        run_id = os.environ.get("GITHUB_RUN_ID", "")
        run_url = f"https://github.com/{repo_name}/actions/runs/{run_id}" if run_id else "本地运行，无云端链接"
        
        send_im_alert(vuln_count, current_run_dir, vuln_modules, run_url)
        sys.exit(1)
    else:
        logging.info(" 未发现漏洞！")
        sys.exit(0)
if __name__ == "__main__":
    run_pipeline()