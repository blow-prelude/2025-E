import numpy as np

# 相机参数
VIDEO_PATH = 0
VIDEO_HEIGHT = 640
VIDEO_WIDTH = 480

bin_thres = 100
bin_maxval = 255

stm_port = "/dev/AMA4"
ti_port = "/dev/AMA3"
baudrate = 115200
timeout = 0.01

# 绘制离散点
DISCRETE_POINT_COLOR = (0, 255, 255)
# 绘制整圆
CIRCLE_COLOR = (255, 0, 255)

# 光斑和靶心的距离
DISTANCE_DIFF = 10

A4_ASPECT_RATIO = np.sqrt(2)  # 约 1.414
ASPECT_RATIO_TOLERANCE = 0.20  # 允许的高宽比误差，设为20%以应对透视变形

# --- KalmanTracker 配置 ---
PROCESS_NOISE = 1e-3
MEASUREMENT_NOISE = 1e-1
CONFIRMATION_THRESHOLD = 3  # 连续更新3帧后确认
UNSEEN_THRESHOLD = 5  # 连续5帧未见则丢失

# --- 模拟器配置 ---
IMG_WIDTH = 1280
IMG_HEIGHT = 720
