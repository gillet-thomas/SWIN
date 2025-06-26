# Navigate to the SwiFT directory
cd /mnt/data/iai/Projects/ABCDE/fmris/CLIP_fmris/SwiFT

# Source micromamba and activate environment
source /mnt/data/iai/micromamba/bin/micromamba
micromamba activate py39

# Set CUDA Visible Devices
export CUDA_VISIBLE_DEVICES=2 # Adjust to your desired GPU index (0-based)

echo "Starting SwiFT training for ADNI Classification..."
python project/main.py