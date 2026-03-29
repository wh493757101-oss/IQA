import cv2
import numpy as np
import time
import json
import os
import urllib.request
import redis
import logging
import base64
import datetime 
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker


DB_URL = "mysql+pymysql://root:visionguard_pwd@localhost:3306/visionguard_db"

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
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine)
# 👆 -------------------------------------- 👆

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

        db = SessionLocal()
        try:
           
            score_val = metrics_json.get("BRISQUE_Score") or metrics_json.get("PSNR_dB")
            
            new_record = EvalRecord(
                task_id=task_id,
                filename=filename,
                mode=metrics_json.get("Mode"),
                score=float(score_val) if score_val else 0.0,
                cost_time_ms=int(cost_time)
            )
            db.add(new_record)
            db.commit() 
            logging.info(f" MySQL 存档成功: Task {task_id[:8]}... 耗时: {int(cost_time)}ms")
        except Exception as db_err:
            logging.error(f" MySQL 写入失败: {db_err}")
            db.rollback()
        finally:
            db.close()
        # 👆 ---------------------------- 👆

        return True

    except Exception as e:
        logging.error(f"Worker crashed on task: {str(e)}")
        redis_client.hset(redis_key, mapping={"status": "failed", "error": str(e)})
        
        db = SessionLocal()
        try:
            error_record = EvalRecord(
                task_id=task_id,
                filename=filename,
                mode="CRASHED",   
                score=-1.0,       
                cost_time_ms=0
            )
            db.add(error_record)
            db.commit()
        except Exception as db_err:
            logging.error(f" 记录失败状态到MySQL时出错: {db_err}")
            db.rollback()
        finally:
            db.close()       
        return False


def run_worker():
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

if __name__ == "__main__":
    run_worker()