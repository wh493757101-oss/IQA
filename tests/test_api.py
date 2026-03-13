import time
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    """验证首页看板加载"""
    response = client.get("/")
    assert response.status_code == 200

def test_submit_invalid_file_type():
    """测试上传非法格式文件 (比如 .txt)"""
    files = {"pred_file": ("test.txt", b"hello world", "text/plain")}
    response = client.post("/api/v1/submit_eval", files=files)
    assert response.status_code in [202, 400, 422] 

def test_get_invalid_task_status():
    """测试查询一个不存在的 Task ID"""
    response = client.get("/api/v1/task_status/999999-invalid-id")
    assert response.status_code == 404

def test_submit_missing_params():
    """测试缺少必填字段的请求"""
    response = client.post("/api/v1/submit_eval")
    assert response.status_code == 422

def test_full_evaluation_flow():
    """测试从提交到结果完成的全过程"""
    payload = b"fake-image-binary-content"
    files = {"pred_file": ("test.jpg", payload, "image/jpeg")}
    submit_res = client.post("/api/v1/submit_eval", files=files)
    assert submit_res.status_code == 202
    task_id = submit_res.json()["task_id"]

    max_retries = 5
    completed = False
    for _ in range(max_retries):
        status_res = client.get(f"/api/v1/task_status/{task_id}")
        assert status_res.status_code == 200
        state = status_res.json()["status"]
        
        if state.startswith("completed"):
            completed = True
            assert "metrics" in status_res.json() 
            break
        time.sleep(1)
