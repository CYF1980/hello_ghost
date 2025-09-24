# --- AI 推論 ---
import degirum as dg
import degirum_tools

# ============ 參數區 ============
# Hailo/DeGirum 模型
INFERENCE_HOST = "@local"
ZOO_URL = "degirum/hailo"
TOKEN = ""
DEVICE_TYPE = "HAILORT/HAILO8L"
MODEL_NAME = "yolov8n_relu6_face--640x640_quant_hailort_multidevice_1"

# 攝影機設定
CAM_INDEX = 0
FRAME_W = 640
FRAME_H = 640


