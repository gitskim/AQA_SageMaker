# declaring random seed
randomseed = 0

C, H, W = 3,112,112
input_resize = 171,128#
test_batch_size = 1

m1_path = '/opt/program/models/model_CNN_94.pth'
m2_path = '/opt/program/models/model_my_fc6_94.pth'
m3_path = '/opt/program/models/model_score_regressor_94.pth'
m4_path = '/opt/program/models/model_dive_classifier_94.pth'
c3d_path = '/opt/program/models/c3d.pickle'

# m1_path = '/tmp/model_CNN_94.pth'
# m2_path = '/tmp/model_my_fc6_94.pth'
# m3_path = '/tmp/model_score_regressor_94.pth'
# m4_path = '/tmp/model_dive_classifier_94.pth'
# c3d_path = '/tmp/c3d.pickle'

with_dive_classification = False
with_caption = False

max_epochs = 100

model_ckpt_interval = 1  # in epochs

base_learning_rate = 0.0001

temporal_stride = 16

BUCKET_NAME = 'aqa-diving'
BUCKET_WEIGHT_FC6 = 'model_my_fc6_94.pth'
BUCKET_WEIGHT_CNN = 'model_CNN_94.pth'
BUCKET_WEIGHT_SCORE_REG = 'model_score_regressor_94.pth'
BUCKET_WEIGHT_DIV_CLASS = 'model_dive_classifier_94.pth'
