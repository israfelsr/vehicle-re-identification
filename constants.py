import os

# Global Variables
#num_feat = 512 #8192 # we will try this at the begining
workers = 2
batch_size = 16
num_epoch = 2
num_class = 775
nc = 3
img_size = 128
device = 'cpu'

# Info
running_stats_freq = 50
eval_every = 5
save_every = 1

## Paths
gdrive_root = 'content/MyDrive/vehicle-re-identification'
veri_root = os.path.join(gdrive_root, 'data/VeRi/')
checkpoint_path = 'training/weights/checkpoint.pth'

# Hyperparameters
alpha = 0.001
tloss = 0.0001
sloss = 1.0
classifier_weight = 1.0
align_weight = 1.0
beta = 1.0
lr = 0.001
