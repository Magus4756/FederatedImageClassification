from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = ''

# -------------------------------------------------- #
# Backbone                                           #
# -------------------------------------------------- #
_C.MODEL.BACKBONE = CN()
_C.MODEL.BACKBONE.NAME = ''
_C.MODEL.BACKBONE.INPUT_RESOLUTION = 32
_C.MODEL.BACKBONE.INPUT_CHENNEL = 3
_C.MODEL.BACKBONE.OUT_CHENNELS = 2048

# -------------------------------------------------- #
# VGG's Conv layers                                  #
# -------------------------------------------------- #
_C.MODEL.CNN = CN()
_C.MODEL.CNN.CONV = []
_C.MODEL.CNN.KERNEL_SIZE = 3
_C.MODEL.CNN.POOLER_SCALE = []
_C.MODEL.CNN.POOLER_SIZE = 2

# -------------------------------------------------- #
# ResNet layers                                      #
# -------------------------------------------------- #
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.INPUT_RESOLUTION = 32
_C.MODEL.RESNET.BLOCK_NUM = [0, 3, 4, 23, 3]
_C.MODEL.RESNET.POOLER_SCALE = [0.5, 0.25, 0.125, 0.0625, 0.03125]
_C.MODEL.RESNET.OUT_CHENNELS = 2048

# -------------------------------------------------- #
# Classifier by FCN                                  #
# -------------------------------------------------- #
_C.MODEL.CLASSIFIER = CN()
_C.MODEL.CLASSIFIER.HIDDEN_LAYERS = []
_C.MODEL.CLASSIFIER.NUM_CLASSES = 10

# -------------------------------------------------- #
# Data set configures                                #
# -------------------------------------------------- #
_C.DATASETS = CN()
_C.DATASETS.NAME = 'CIFAR10'
_C.DATASETS.TRAIN = ('data',)
_C.DATASETS.VAL = ('data',)
_C.DATASETS.TRAIN_BATCH_SIZE = 64

# -------------------------------------------------- #
# Training configures                                #
# -------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.CUDA = True
_C.SOLVER.LEARNING_RATE = 0.1
_C.SOLVER.WARMUP_EPOCHS = 0
_C.SOLVER.WARMUP_FACTOR = 1. / 10
_C.SOLVER.WARMUP_METHOD = 'linear'
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = [300, 400, ]
_C.SOLVER.MAX_EPOCH = 500
_C.SOLVER.STRATEGY = 'MBGD'
_C.SOLVER.CHECKPOINT = 100
_C.SOLVER.OUTPUT_DIR = 'checkpoint'
_C.SOLVER.CLIENT_NUM = 100

cfg = _C
