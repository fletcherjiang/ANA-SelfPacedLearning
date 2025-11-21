# config.py
import torch
# ==================================================================
# 固定参数
# ==================================================================
EPOCH = 200
BATCH_SIZE = 32
LEARNING_RATE = 5e-3
WEIGHT_DECAY = 0
N = 224
MOMENTUM = (0.9, 0.999)
TRAIN_LIMIT = 200  # 训练时使用的样本数量
VAL_LIMIT = 200   # 验证时使用的样本数量

GPU_IN_USE = torch.cuda.is_available()
# Update these paths to point to your own data directory
DIR_TRAIN_IMAGES = './data/images/'  # Path to training images
DIR_TEST_IMAGES = './data/images/'   # Path to test/validation images
PATH_TRAIN_ANNFILE = 'all_single_small_82240_19400_19330.csv'
PATH_TEST_ANNFILE = 'all_single_small_82240_19400_19330.csv'
NUM_CATEGORIES = 8  # 8个类别，数据集决定
LOSS_OUTPUT_INTERVAL = 300
PATIENT = 0
STAGE = 1
LAST_STAGE = 6

# ==================================================================
# 全局变量
# ==================================================================
current_training_iteration = torch.tensor([1])
current_test_iteration = torch.tensor([1])
loss_graph_window = 'loss graph'
test_f1_graph_window = 'test OF1 and CF1 graph'
evaluation_window = 'six evaluation metrics'
of1 = 0.
cf1 = 0.
best_mi = 0.
best_ma = 0.
best_acc = 0.
best_mAP = 0.

of1_single = 0.
cf1_single = 0.
best_mi_single = 0.
best_ma_single = 0.
best_acc_single = 0.
best_mAP_single = 0.

of1_multiple = 0.
cf1_multiple = 0.
best_mi_multiple = 0.
best_ma_multiple = 0.
best_acc_multiple = 0.
best_mAP_multiple = 0
