# PARAMS
PATCH_SIZE = 16  # pixels per side of square patches
VAL_SIZE = 10  # size of the validation set (number of images)
CUTOFF = 0.25 # 0.25 # 0.25  # minimum average brightness for a mask patch to be classified as containing road

# MAGIC NUMBERS
ROOT_PATH = "./data/raw"  # data location
EXTERNAL_PATH = "./data/external"  # external data location

# SETTINGS
VERBOSE = True
SHOW_PLOTS = False
BATCH_SIZE = 4
SAVE_MODEL = True
LOAD_MODEL = True

RESIZE_TO = 384, 384

