import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Optional
import uuid
import json
import logging
import base64
from contextlib import asynccontextmanager 
import redis.asyncio as redis 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


redis_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    try:
       
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        await redis_client.ping()
        logging.info("Async Redis connected successfully!")
    except Exception as e:
        logging.warning(f"Async Redis connection failed: {e}")
        
    yield  
    
    if redis_client:
        await redis_client.aclose()
        logging.info("Async Redis connection closed gracefully.")

app = FastAPI(
    title="VisionGuard API Gateway", 
    description="纯 I/O 全异步网关",
    lifespan=lifespan 
)

@app.post("/api/v1/submit_eval")
async def submit_evaluation(
    pred_file: UploadFile = File(...),
    gt_file: Optional[UploadFile] = File(None)
):
    task_id = str(uuid.uuid4())
    redis_key = f"iqa:task:{task_id}"
    
    pred_bytes = await pred_file.read()
    gt_bytes = await gt_file.read() if gt_file else b""

    await redis_client.hset(redis_key, mapping={
        "task_id": task_id,
        "status": "pending",
        "filename": pred_file.filename
    })
    await redis_client.expire(redis_key, 86400)

    task_payload = {
        "task_id": task_id,
        "filename": pred_file.filename,
        "pred_b64": base64.b64encode(pred_bytes).decode('utf-8'),
        "gt_b64": base64.b64encode(gt_bytes).decode('utf-8') if gt_bytes else ""
    }
    

    await redis_client.lpush("iqa:task_queue", json.dumps(task_payload))

    return JSONResponse(status_code=202, content={"task_id": task_id, "status": "pending"})

@app.get("/api/v1/task_status/{task_id}")
async def get_task_status(task_id: str):
    redis_key = f"iqa:task:{task_id}"
    
    task_data = await redis_client.hgetall(redis_key)

    if not task_data:
        return JSONResponse(status_code=404, content={"message": "Task not found"})

    status = task_data.get("status", "")
    metrics_str = task_data.get("metrics_json", "{}")
    cost_time = float(task_data.get("cost_time_ms", 0))

    if status.startswith("completed"):
        metrics = json.loads(metrics_str)
        return {"task_id": task_id, "status": status, "metrics": metrics, "cost_time_ms": round(cost_time, 2)}
    
    return {"task_id": task_id, "status": status}

@app.get("/", response_class=HTMLResponse)
async def serve_webpage():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>VisionGuard Gateway Running</h1>"

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)