import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Optional
import uuid
import json
import logging
import base64
from contextlib import asynccontextmanager 
import redis.asyncio as redis
import io
import os
from PIL import Image, UnidentifiedImageError
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker


db_pwd = os.environ.get("DB_PASSWORD", "visionguard_pwd")
DB_URL = f"mysql+pymysql://root:{db_pwd}@localhost:3306/visionguard_db"
engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class EvalRecord(Base):
    __tablename__ = "eval_records"
    id = Column(Integer, primary_key=True, index=True)
    task_id = Column(String(255), unique=True, index=True)
    filename = Column(String(255))
    mode = Column(String(50))
    score = Column(Float, nullable=True)
    cost_time_ms = Column(Integer)
    created_at = Column(DateTime)

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
    description="I/O 异步网关",
    lifespan=lifespan 
)

MAX_FILE_SIZE = 10 * 1024 * 1024  # 
ALLOWED_EXTS = {'jpg', 'jpeg', 'png', 'bmp', 'gif', 'webp'}


MAGIC_NUMBERS = {
    b'\xff\xd8\xff': 'jpeg',
    b'\x89PNG\r\n\x1a\n': 'png',
    b'BM': 'bmp',
    b'GIF87a': 'gif',
    b'GIF89a': 'gif',
    b'RIFF': 'webp'
}

def is_real_image(content: bytes) -> bool:
    """深度检验：判断文件底层的二进制流是否真的是图片"""
    for magic in MAGIC_NUMBERS:
        if content.startswith(magic):
            return True
    return False

async def secure_read_file(file: UploadFile, is_optional: bool = False) -> bytes:
    """读取文件并进行四层深度防御校验"""
    if not file or not file.filename:
        return b""
        

    ext = file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if ext not in ALLOWED_EXTS:
        raise HTTPException(status_code=422, detail=f"安全拦截：不支持的文件扩展名 '{ext}'")

    content = await file.read()

    if not content and is_optional:
        return b""


    if len(content) == 0:
        raise HTTPException(status_code=422, detail="安全拦截：文件内容不能为空 (0 bytes)")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="安全拦截：文件大小超出 10MB 限制")

    if not is_real_image(content):
        raise HTTPException(status_code=422, detail="安全拦截：检测到文件伪装！底层的二进制流并非图片。")


    try:
        with Image.open(io.BytesIO(content)) as img:
            width, height = img.size
            if width < 10 or height < 10:
                raise HTTPException(status_code=422, detail=f"安全拦截：图片尺寸太小 ({width}x{height})，为保护后端算力，最小要求 10x10 像素。")
    except UnidentifiedImageError:
        raise HTTPException(status_code=422, detail="安全拦截：图片数据已损坏，无法解析图像尺寸。")

    return content

@app.post("/api/v1/submit_eval")
async def submit_evaluation(
    pred_file: UploadFile = File(...),
    gt_file: Optional[UploadFile] = File(None)
):
    pred_bytes = await secure_read_file(pred_file)
    gt_bytes = await secure_read_file(gt_file, is_optional=True)

    task_id = str(uuid.uuid4())
    redis_key = f"iqa:task:{task_id}"

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
@app.get("/api/v1/history")
def get_history(limit: int = 50):
    """获取最近的图像评测历史记录"""
    db = SessionLocal()
    try:
        records = db.query(EvalRecord).order_by(EvalRecord.id.desc()).limit(limit).all()
        results = []
        for r in records:
            results.append({
                "task_id": r.task_id,
                "filename": r.filename,
                "mode": r.mode,
                "score": r.score,
                "cost_time_ms": r.cost_time_ms,
                "created_at": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else None
            })
            
        return {"code": 200, "message": "success", "data": results}
        
    except Exception as e:
        return {"code": 500, "message": f"查询数据库失败: {str(e)}"}
    finally:
        db.close()      

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)