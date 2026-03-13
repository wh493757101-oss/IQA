import uvicorn
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional
import cv2
import numpy as np
import uuid
import time
import json
import os
import urllib.request
import redis
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

app = FastAPI(
    title="VisionGuard API ",
    description="图像质量评测平台",
    version="1.0.0"
)
try:
    redis_client = redis.Redis(host='redis-server', port=6379, db=0, decode_responses=True)  #Docker
    redis_client.ping()
    logging.info("Redis connected successfully!")
except Exception as e:
    logging.warning(f"Redis connection failed, ensure Redis is running: {e}")
def ensure_brisque_models():
    os.makedirs("models", exist_ok=True)
    files = {
       "models/brisque_model_live.yml": "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_model_live.yml",
        "models/brisque_range_live.yml": "https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/quality/samples/brisque_range_live.yml"
    }
    for filename, url in files.items():
        if not os.path.exists(filename):
           try:
                urllib.request.urlretrieve(url, filename)
                logging.info(f"Downloaded {filename}")
           except Exception as e:
                logging.warning(f"Failed to download {filename}: {e}")

ensure_brisque_models()

def background_evaluation_worker(task_id: str, filename: str, pred_bytes: bytes, gt_bytes: bytes = b""):
    start_time = time.time()
    
    redis_key = f"iqa:task:{task_id}" 
    
    try:
        redis_client.hset(redis_key, "status", "processing")
        
        pred_arr = np.frombuffer(pred_bytes, np.uint8)
        pred_img = cv2.imdecode(pred_arr, cv2.IMREAD_COLOR)

        if gt_bytes: 
            gt_arr = np.frombuffer(gt_bytes, np.uint8)
            gt_img = cv2.imdecode(gt_arr, cv2.IMREAD_COLOR)
            if pred_img.shape != gt_img.shape:
                pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
            
            mse = np.mean((pred_img.astype("float") - gt_img.astype("float")) ** 2)
            psnr = cv2.PSNR(pred_img, gt_img)
            
            status_msg = "completed_FR"
            metrics_json = {"Mode": "FR-IQA (Full-Reference)", "MSE": round(float(mse), 4), "PSNR_dB": round(float(psnr), 2)}
        else:
            brisque_engine = cv2.quality.QualityBRISQUE_create("models/brisque_model_live.yml", "models/brisque_range_live.yml")
            brisque_score = brisque_engine.compute(pred_img)[0]

            status_msg = "completed_NR"
            metrics_json = {
                "Mode": "NR-IQA (BRISQUE Blind)",
                "BRISQUE_Score": round(float(brisque_score), 2),
                "Conclusion": "Excellent" if brisque_score < 35 else "Degraded" if brisque_score < 60 else "Poor"
            }

        cost_time = (time.time() - start_time) * 1000
        redis_client.hset(redis_key, mapping={
            "status": status_msg,
            "metrics_json": json.dumps(metrics_json),
            "cost_time_ms": str(cost_time)
        })
        logging.info(f"Background task {task_id} completed! Mode: {metrics_json['Mode']}")

    except Exception as e:
        logging.error(f"Background task {task_id} failed: {str(e)}")
        redis_client.hset(redis_key, mapping={"status": "failed", "error": str(e)})

@app.post("/api/v1/submit_eval")
async def submit_evaluation(
    background_tasks: BackgroundTasks,
    pred_file: UploadFile = File(...),
    gt_file: Optional[UploadFile] = File(None)
):
    task_id = str(uuid.uuid4())
    redis_key = f"iqa:task:{task_id}"
    
    pred_bytes = await pred_file.read()
    gt_bytes = await gt_file.read() if gt_file else b""

    redis_client.hset(redis_key, mapping={
        "task_id": task_id,
        "status": "pending",
        "filename": pred_file.filename
    })
    
    redis_client.expire(redis_key, 86400)

    background_tasks.add_task(background_evaluation_worker, task_id, pred_file.filename, pred_bytes, gt_bytes)

    return JSONResponse(status_code=202, content={"task_id": task_id, "status": "pending"})

@app.get("/api/v1/task_status/{task_id}")
async def get_task_status(task_id: str):
    redis_key = f"iqa:task:{task_id}"
    
    task_data = redis_client.hgetall(redis_key)

    if not task_data:
        return JSONResponse(status_code=404, content={"message": "Task not found or expired"})

    status = task_data.get("status", "")
    metrics_str = task_data.get("metrics_json", "{}")
    cost_time = float(task_data.get("cost_time_ms", 0))

    if status.startswith("completed"):
        metrics = json.loads(metrics_str)
        return {"task_id": task_id, "status": status, "metrics": metrics, "cost_time_ms": round(cost_time, 2)}
    
    return {"task_id": task_id, "status": status}

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def serve_webpage():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>Error: index.html not found</h1>"

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)