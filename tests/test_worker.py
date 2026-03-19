import pytest
import json
import cv2
import base64
import numpy as np
from unittest.mock import MagicMock
from worker import process_single_task

def get_test_image_b64(width=128, height=128):
    """动态生成一张合法的 JPEG 图像 Base64 字符串"""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img, (10, 10), (width-10, height-10), (255, 0, 0), -1)
    
    _, buffer = cv2.imencode('.jpg', img)
    return base64.b64encode(buffer).decode('utf-8')

VALID_IMAGE_B64 = get_test_image_b64()

def test_worker_nr_iqa_branch():
    """测试无参考 IQA 分支 (NR-IQA)"""
    task_data = {
        "task_id": "test-nr-001",
        "filename": "nr_test.jpg",
        "pred_b64": VALID_IMAGE_B64,
        "gt_b64": "" 
    }
    mock_redis = MagicMock()
    

    result = process_single_task(task_data, mock_redis)
    
    assert result is True
    last_call_mapping = mock_redis.hset.call_args_list[-1][1]['mapping']
    assert "completed_NR" in last_call_mapping['status']
    print("\n[NR 分支] 动态图像测试通过！")

def test_worker_fr_iqa_branch():
    """测试全参考 IQA 分支 (FR-IQA)"""

    pred_b64 = VALID_IMAGE_B64

    gt_b64 = get_test_image_b64() 
    
    task_data = {
        "task_id": "test-fr-001",
        "filename": "fr_test.jpg",
        "pred_b64": pred_b64,
        "gt_b64": gt_b64
    }
    mock_redis = MagicMock()
    
    result = process_single_task(task_data, mock_redis)
    
    assert result is True
    last_call_mapping = mock_redis.hset.call_args_list[-1][1]['mapping']
    assert "completed_FR" in last_call_mapping['status']
    print("\n[FR 分支] 动态图像联调通过！")

def test_worker_error_handling():
    """测试异常捕获逻辑"""
    task_data = {
        "task_id": "test-err-001",
        "filename": "error.png",
        "pred_b64": "invalid_random_string",
        "gt_b64": ""
    }
    mock_redis = MagicMock()
    result = process_single_task(task_data, mock_redis)
    
    assert result is False
    last_call_mapping = mock_redis.hset.call_args_list[-1][1]['mapping']
    assert last_call_mapping['status'] == "failed"
    print("\n[异常分支] 健壮性测试通过！")