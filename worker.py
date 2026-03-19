import cv2
import numpy as np
import time
import json
import os
import urllib.request
import redis
import logging
import base64

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [WORKER] - %(levelname)s - %(message)s'
)

def ensure_brisque_models():
    os.makedirs("models", exist_ok=True)
    files = {
       "models/brisque_model_live.yml": "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml",
        "models/brisque_range_live.yml": "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml"
    }
    for filename, url in files.items():
        if not os.path.exists(filename):
            urllib.request.urlretrieve(url, filename)

def process_single_task(task_data, redis_client):
    task_id = task_data["task_id"]
    filename = task_data["filename"]
    redis_key = f"iqa:task:{task_id}"
    
    try:
        redis_client.hset(redis_key, "status", "processing")
        logging.info(f"Picked up task: {filename} ({task_id[:8]}...)")
        
        start_time = time.time()
        
        pred_bytes = base64.b64decode(task_data["pred_b64"])
        pred_arr = np.frombuffer(pred_bytes, np.uint8)
        pred_img = cv2.imdecode(pred_arr, cv2.IMREAD_COLOR)
        
        gt_bytes = base64.b64decode(task_data["gt_b64"]) if task_data.get("gt_b64") else b""


        if gt_bytes: 
            gt_arr = np.frombuffer(gt_bytes, np.uint8)
            gt_img = cv2.imdecode(gt_arr, cv2.IMREAD_COLOR)
            if pred_img.shape != gt_img.shape:
                pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
            
            mse = np.mean((pred_img.astype("float") - gt_img.astype("float")) ** 2)
            psnr = cv2.PSNR(pred_img, gt_img)
            
            status_msg = "completed_FR"
            metrics_json = {"Mode": "FR-IQA", "MSE": round(float(mse), 4), "PSNR_dB": round(float(psnr), 2)}
        else:
            brisque_engine = cv2.quality.QualityBRISQUE_create("models/brisque_model_live.yml", "models/brisque_range_live.yml")
            brisque_score = brisque_engine.compute(pred_img)[0]

            status_msg = "completed_NR"
            metrics_json = {
                "Mode": "NR-IQA",
                "BRISQUE_Score": round(float(brisque_score), 2),
                "Conclusion": "Excellent" if brisque_score < 35 else "Degraded" if brisque_score < 60 else "Poor"
            }


        cost_time = (time.time() - start_time) * 1000
        redis_client.hset(redis_key, mapping={
            "status": status_msg,
            "metrics_json": json.dumps(metrics_json),
            "cost_time_ms": str(cost_time)
        })
        logging.info(f"Finished {filename} | Mode: {metrics_json['Mode']} | Cost: {int(cost_time)}ms")
        return True

    except Exception as e:
        logging.error(f"Worker crashed on task: {str(e)}")
        redis_client.hset(redis_key, mapping={"status": "failed", "error": str(e)})
        return False


def run_worker():# pragma: no cover
    ensure_brisque_models()
    redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    logging.info("Worker started! Waiting for tasks in 'iqa:task_queue'...")
    
    while True:
        try:
            result = redis_client.brpop("iqa:task_queue", timeout=0)
            if not result:
                continue
                
            _, task_json = result
            task_data = json.loads(task_json)
            process_single_task(task_data, redis_client)

        except Exception as e:
            logging.error(f"Worker loop error: {str(e)}")
            time.sleep(1) 

if __name__ == "__main__":# pragma: no cover
    run_worker()