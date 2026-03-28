# VisionGuard - 图像质量自动评测平台

VisionGuard 是一款自动化图像质量评估平台。了解决手动评测图像效率低、数据统计易出错的问题，开发了支持自动化执行、多维度指标量化的评测工具。

## 文件说明

| File | Description |
|------|-------------|
| `main.py` | FastAPI 全异步非阻塞 API 网关 (生产者) |
| `worker.py` | 无状态的 OpenCV 图像处理计算节点 (消费者) |
| `run.py` | 自动化批量执行脚本，可以一键处理整个文件夹的图片 |
| `locustfile.py` | 用于进行压力测试，验证系统在高并发下的稳定性 |
| `models/` | 预训练的图像质量评估模型文件 |
| `requirements.txt` | Python 环境依赖文件 |
| `data/` | 存放数据的文件夹 |
| `tests/` |  存放 Pytest 自动化测试脚本的文件夹 |
| `.github/workflows/` |  存放 GitHub Actions 云端自动测试配置 |
| `planner_agent.py` |  根据要求生成测试用例Agent |
| `run_qa_agent.py` |  执行测试用例Agent |
| `qa_tools.py` |  工具函数库 |

## 安装依赖

安装所需的依赖：

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 启动主程序

确保 Redis 在后台运行，然后启动主程序：

```bash
python main.py
```
```bash
python worker.py
```

### 2. 运行批量评估

将要评估的图片放入 `preds/` 文件夹，然后运行：

```bash
python run.py
```

### 3. 自动化报告
根据预设的及格线（如 PSNR > 30dB）自动判断图片是否合格。终端会直接打印出总的及格率和缺陷率，生成的 CSV 报告里也会带有 PASS/FAIL 的标记。

### 4. 访问 Web界面

服务启动后，打开浏览器并访问 `http://127.0.0.1:8000`，即可使用内置的 Web 界面。

### 5. 自动化测试与质量保证
测试框架 (Pytest)：在 tests/ 目录下编写了针对后端的测试用例。可以自动模拟图片上传、查询不存在的任务等异常情况。

测试覆盖率 (Coverage)：使用 pytest-cov 测算了代码覆盖率，目前测试脚本已经覆盖了约 72% 的代码。

CI/CD (GitHub Actions)：每次把代码 push 到 GitHub，云端的服务器就会自动帮我建环境、装依赖，并跑一遍所有的测试用例。

## 测试结果

### 模拟 50 个虚拟用户 (阶梯式加压)
![50 Users Test](assets/image-1.png)  

### 模拟 200 个虚拟用户 (阶梯式加压)
![200 Users Test](assets/image-2.png)  

### 模拟 500 个虚拟用户 (阶梯式加压)
![500 Users Test](assets/image-3.png)  

## 加入消息队列后的测试结果

### 模拟 200 个虚拟用户 (阶梯式加压)
![200 Users Test](assets/image-01.png)

### 模拟 500 个虚拟用户 (阶梯式加压)
![500 Users Test](assets/image-02.png)

### 模拟 1000 个虚拟用户 (阶梯式加压)
![1000 Users Test](assets/image-03.png)

## 部署到云原生

使用 Docker Compose 部署：

```bash
docker compose up --build -d
```
