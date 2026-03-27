import time
import os
import logging
import datetime
from run_qa_agent import run_agent
import sys
import requests
from planner_agent import generate_dynamic_test_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CI_PIPELINE] - %(message)s')

def generate_master_report(results: list, output_dir: str):
    master_file = os.path.join(output_dir, "VisionGuard_CI_Master_Report.md")
    
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
        f" VisionGuard 安全警报 \n\n"
        f"QA Agent 刚刚在 GitHub CI 检测中出现问题！\n"
        f"- 发现漏洞数: {vuln_count} 个高危漏洞！\n"
        f"- 被攻破模块: {', '.join(details)}\n\n"
        f" [点击此处下载完整检测报告]({run_url})"
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
    PROMPT_MATRIX = generate_dynamic_test_matrix()
    logging.info(" 启动 VisionGuard CI 流水线...")
    
    base_report_dir = "test_reports"
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_run_dir = os.path.join(base_report_dir, f"run_{run_timestamp}")
    
    os.makedirs(current_run_dir, exist_ok=True)
    logging.info(f" 已创建本次构建的专属工件目录: {current_run_dir}")
    
    all_results = []
    vuln_count = 0
    vuln_modules = []
    FAILED_STATUSES = ["发现漏洞", "Agent 运行超时", "框架级错误", "需人工复核"]
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
                
                current_status = result.get("status", "")
                if current_status in FAILED_STATUSES:
                    vuln_count += 1
                    vuln_modules.append(test_name)
                    logging.warning(f"模块 {test_name} 检出异常: {current_status}")
                
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
        logging.info("未发现漏洞，测试通过！")
        sys.exit(0)
if __name__ == "__main__":
    run_pipeline()