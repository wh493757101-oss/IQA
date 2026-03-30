import os
import time
import requests
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

BASE_URL = "http://127.0.0.1:8000"
PREDS_DIR = "data/preds"
GTS_DIR = "data/gts"
SUPPORTED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')
def submit_single_pair(filename):
    """判断图片是否携带GT"""
    pred_path = os.path.join(PREDS_DIR, filename)
    gt_path = os.path.join(GTS_DIR, filename)
    f_pred = open(pred_path, "rb")
    files = {
        "pred_file": (filename, f_pred, "application/octet-stream")
    }
    
    f_gt = None
    if os.path.exists(gt_path):
        f_gt = open(gt_path, "rb")
        files["gt_file"] = (filename, f_gt, "application/octet-stream")
        
    try:
        res = requests.post(f"{BASE_URL}/api/v1/submit_eval", files=files)
        
        if res.status_code == 202:
            return {"filename": filename, "task_id": res.json()["task_id"], "status": "submitted"}
        else:
            return {"filename": filename, "error": f"Submission failed: {res.status_code}"}
    except Exception as e:
        return {"filename": filename, "error": str(e)}
    finally:
        f_pred.close()
        if f_gt:
            f_gt.close()
def generate_visual_report(tasks_info):
    logging.info("Generating visual reports...")

    data = []
    for task in tasks_info:
        if task.get("status", "").startswith("completed"):
            metrics = task.get("metrics", {})
            data.append({
                "Filename": task["filename"],
                "Mode": "FR" if "FR" in metrics.get("Mode", "") else "NR",
                "PSNR_dB": metrics.get("PSNR_dB", None),
                "BRISQUE_Score": metrics.get("BRISQUE_Score", None)
            })

    if not data:
        logging.warning("Incomplete data for report generation")
        return

    df = pd.DataFrame(data)

    def assert_quality(row):
        PSNR_THRESHOLD = 30.0  
        BRISQUE_THRESHOLD = 50.0
        if row['Mode'] == 'FR' and pd.notnull(row['PSNR_dB']):
            return ' PASS' if row['PSNR_dB'] >= PSNR_THRESHOLD else ' FAIL'
        elif row['Mode'] == 'NR' and pd.notnull(row['BRISQUE_Score']):
            return ' PASS' if row['BRISQUE_Score'] <= BRISQUE_THRESHOLD else ' FAIL'
        return ' UNKNOWN'

    
    df['Test_Result'] = df.apply(assert_quality, axis=1)
    total_cases = len(df)
    failed_cases = len(df[df['Test_Result'] == ' FAIL'])
    defect_rate = (failed_cases / total_cases) * 100 if total_cases > 0 else 0

    logging.info("Automated image quality regression test completed!")
    logging.info(f"Total test cases: {total_cases} images")
    logging.info(f"Defects found: {failed_cases} images (Defect rate: {defect_rate:.2f}%)")

    df.to_csv("iqa_evaluation_report.csv", index=False, encoding="utf-8-sig")
    logging.info("Data exported: iqa_evaluation_report.csv")

    df_fr = df[df["PSNR_dB"].notnull()]
    df_nr = df[df["BRISQUE_Score"].notnull()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if not df_fr.empty:
        axes[0].bar(df_fr["Filename"], df_fr["PSNR_dB"], color='#4C72B0', edgecolor='black')
        axes[0].set_title("FR-IQA: PSNR (Higher is Better)", fontsize=14, fontweight='bold')
        axes[0].set_ylabel("dB", fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].axhline(y=30, color='red', linestyle='--', linewidth=2, label='Baseline (30 dB)')
        axes[0].legend()
        for i, v in enumerate(df_fr["PSNR_dB"]):
            axes[0].text(i, v + 0.5, f"{v:.1f}", ha='center', fontweight='bold')

    if not df_nr.empty:
        axes[1].bar(df_nr["Filename"], df_nr["BRISQUE_Score"], color='#DD8452', edgecolor='black')
        axes[1].set_title("NR-IQA: BRISQUE (Lower is Better)", fontsize=14, fontweight='bold')
        axes[1].set_ylabel("Score", fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].axhline(y=50, color='red', linestyle='--', linewidth=2, label='Warning Line (50)')
        axes[1].legend()
        for i, v in enumerate(df_nr["BRISQUE_Score"]):
            axes[1].text(i, v + 1, f"{v:.1f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("iqa_visual_report.png", dpi=300)
    logging.info("Visual report saved: iqa_visual_report.png")

def run_batch_test():
    if not os.path.exists(PREDS_DIR) or not os.path.exists(GTS_DIR):
        logging.error(f"{PREDS_DIR} or {GTS_DIR} directory not found!")
        return

    valid_files = [
        f for f in os.listdir(PREDS_DIR)
        if f.lower().endswith(SUPPORTED_EXTS)
    ]

    if not valid_files:
        logging.warning(f"{PREDS_DIR} is empty or no supported image formats (jpg/png/bmp)!")
        return

    logging.info(f"Found {len(valid_files)} images to evaluate, submitting in parallel...")
    
    tasks_info = []

    submit_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=200) as executor:
        futures = {executor.submit(submit_single_pair, f): f for f in valid_files}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if "task_id" in result:
                tasks_info.append(result)
                logging.info(f"Submitted: {result['filename']} -> TaskID: {result['task_id'][:8]}...")
            else:
                logging.error(f"Failed: {result['filename']} - {result.get('error')}")

    logging.info(f"All tasks submitted! Time: {time.time() - submit_start:.2f}s")

    logging.info("Monitoring background processing...")
    completed_count = 0
    total_tasks = len(tasks_info)

    while completed_count < total_tasks:
        completed_count = 0
        for task in tasks_info:
            if task["status"].startswith("completed") or task["status"] == "failed":
                completed_count += 1
                continue
            res = requests.get(f"{BASE_URL}/api/v1/task_status/{task['task_id']}").json()
            task["status"] = res["status"]

            if res["status"].startswith("completed"):
                task["metrics"] = res["metrics"]
                mode = res["metrics"].get("Mode", "Unknown")

                if "FR" in mode:
                    score = f"PSNR: {res['metrics'].get('PSNR_dB')} dB"
                else:
                    if "BRISQUE_Score" in res['metrics']:
                        score = f"BRISQUE: {res['metrics']['BRISQUE_Score']} | Result: {res['metrics'].get('Conclusion')}"
                    elif "NIQE_Score" in res['metrics']:
                        score = f"NIQE: {res['metrics']['NIQE_Score']} | Result: {res['metrics'].get('Conclusion')}"
                    else:
                        score = f"Clarity: {res['metrics'].get('Clarity_Score')}"

                logging.info(f"[Done] {task['filename']} | {mode} | {score}")
                completed_count += 1

            elif res["status"] == "failed":
                logging.error(f"[Failed] {task['filename']} - Backend processing crashed.")
                completed_count += 1

        if completed_count < total_tasks:
            time.sleep(1)

    logging.info("Batch evaluation completed!")
    generate_visual_report(tasks_info)

if __name__ == "__main__":
    run_batch_test()