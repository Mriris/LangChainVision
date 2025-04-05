import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, "data")
UPLOAD_FOLDER = os.path.join(ROOT_DIR, "uploads")
RESULT_FOLDER = os.path.join(ROOT_DIR, "results")
MODEL_WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
MODEL_CACHE_DIR = os.path.join(ROOT_DIR, "models")

# 确保必要的目录存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_WEIGHTS_DIR, exist_ok=True)
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

# Ollama设置
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
ANALYSIS_MODEL = os.environ.get("ANALYSIS_MODEL", "phi4-mini:latest")
VISION_MODEL = os.environ.get("VISION_MODEL", "llava:13b")

# 模型设置
YOLO_MODEL = os.environ.get("YOLO_MODEL", "yolov5s")
CLASSIFICATION_MODEL = "resnet50"
ENHANCEMENT_MODEL = "swinir"

# 结果图像设置
ANNOTATED_IMAGE_PATH = os.path.join(RESULT_FOLDER, "annotated_image.jpg")
ENHANCED_IMAGE_PATH = os.path.join(RESULT_FOLDER, "enhanced_image.jpg")
COMPARISON_IMAGE_PATH = os.path.join(RESULT_FOLDER, "image_comparison.jpg")

# 允许的文件扩展名
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

# 任务类型
TASK_CLASSIFICATION = "classification"
TASK_ANNOTATION = "annotation" 
TASK_INTERPRETATION = "interpretation"
TASK_ENHANCEMENT = "enhancement" 