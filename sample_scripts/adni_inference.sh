# Navigate to the SwiFT directory
cd /mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/SwiFT

# Source micromamba and activate environment
source /mnt/data/iai/micromamba/bin/micromamba
micromamba activate py39

# Set CUDA Visible Devices
export CUDA_VISIBLE_DEVICES=1 # Adjust to your desired GPU index (0-based)

echo "Starting SwiFT inference for ADNI Classification..."
 
TRAINER_ARGS='--accelerator gpu --max_epochs 10 --num_nodes 1 --devices 1 --strategy DDP' # specify the number of gpus as '--devices'
MAIN_ARGS='--batch_size 8 --num_workers 8 --patch_size 7 7 7 1 --input_type rest --loggername tensorboard --image_path /mnt/data/iai/datasets/ADNI_MNI_to_TRs/metadata/metafile.csv'
DATA_ARGS='--dataset_name ADNI --project_name adni_sex --downstream_task sex --downstream_task_type classification'
OPTIONAL_ARGS='--test_only --test_ckpt_path ./output/adni_sex/last.ckpt'
# export NEPTUNE_API_TOKEN="{neptune API token}" # when using neptune as a logger

export CUDA_VISIBLE_DEVICES=2

python project/main.py $TRAINER_ARGS $MAIN_ARGS $DATA_ARGS $OPTIONAL_ARGS \
 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 --sequence_length 20 --first_window_size 4 4 4 4 --window_size 4 4 4 4 --img_size 112 112 112 20

