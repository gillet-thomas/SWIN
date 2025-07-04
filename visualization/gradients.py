import os
import torch
import warnings
import nibabel as nib
import nilearn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
import glob
import nilearn.image
import time
import torchio as tio
import yaml
from SWIN import ADNISwiFTDataset, Model
from nilearn import plotting

warnings.simplefilter(action='ignore', category=FutureWarning)

cuda_id = 1
target = 0
generate_maps = True
max_iter = 25000
save_dir = os.path.join(os.getcwd(),'visualization/gradients')

# Load dataset
config = yaml.safe_load(open("data/config.yaml", "r"))
best_model_path = config[f'best_swin_{config["map_task"]}']
data_module = ADNISwiFTDataset(config, mode='test')
test_loader = torch.utils.data.DataLoader(data_module.data, batch_size=config['eval_batch_size'], shuffle=False, num_workers=config['num_workers'], pin_memory=True, prefetch_factor=2)
model = Model(config)
model.to(device=cuda_id) if torch.cuda.is_available() else model
model.load_state_dict(torch.load(best_model_path))
model.eval()
print(f"Using model from {best_model_path}")
print(f"Running on {len(test_loader)} samples")

# Load attribution method
integrated_gradients = IntegratedGradients(model)
noise_tunnel = NoiseTunnel(integrated_gradients)   
# gradient_shap = Occlusion(model)

kwargs = {
    "nt_samples": 4,
    "nt_samples_batch_size": 4,
    "nt_type": "smoothgrad_sq", # 1
    #"stdevs": 0.05,
    "internal_batch_size": 4,
}

# Generate IG maps
def generate_ig_maps():
    for idx, (subject, _, _, start_frame_idx) in enumerate(tqdm(test_loader),0):
        subj_name = subject[0]
        data_fmri, data_target = data_module[idx]
        data_fmri = data_fmri.to(device=cuda_id).unsqueeze(0)  # 1, 1, 112, 112, 112, 20
        data_target = int(data_target.item())

        pred = model.forward(data_fmri)
        pred_prob = torch.sigmoid(pred)
        pred_int = (pred_prob>0.5).int().item()

        #only choose corrected samples
        if pred_int == data_target: # Of the subjects that corrected
            if (data_target == 0 and pred_prob <= 0.25) or (data_target == 1 and pred_prob >= 0.75):
                file_dir = os.path.join(save_dir, f'ADNI_{config["map_task"]}_target{target}')
                os.makedirs(file_dir,exist_ok=True)
                file_path = os.path.join(file_dir, f"{subj_name}_{start_frame_idx.item()}.pt")
                if not os.path.exists(file_path):
                    result = noise_tunnel.attribute(data_fmri,baselines=data_fmri[0,0,0,0,0,0].item(),target=None,**kwargs)
                    result = result.squeeze().cpu() # 112 112 112 20

                    torch.save(result, file_path)
        
        if idx >= max_iter:
            print("MAX ITERATION REACHED")
            break

if generate_maps:
    generate_ig_maps()

# Importing template images for visualizations
icbm = tio.datasets.ICBM2009CNonlinearSymmetric()
target_affine = nib.load('/mnt/data/iai/datasets/ADNI_CONN_conversion/corresponding_processed/136_S_4993_I342514/wauI342514_Resting_State_fMRI_136_S_4993.nii').affine
target_shape = (112,112,112)
MNI152 = nilearn.image.resample_img(icbm['t1']['path'],target_affine=target_affine,target_shape=target_shape,interpolation='nearest')

# Make mask
mask_npy = MNI152.get_fdata()
target_affine = MNI152.affine
mask = (mask_npy!=0).astype(int) # Creates binary mask for background
IG_dir = f'{save_dir}/ADNI_{config["map_task"]}_target{target}/*'

# Collect IG maps
maps=[]

for idx,file in enumerate(tqdm(glob.glob(IG_dir))):
    volumes=[]
    masked_volumes=[]
    image = torch.load(file) # 112 112 112 20

    # Segment background
    for i in range(image.shape[3]):
        masked_image = image[:,:,:,i] * mask
        masked_volumes.append(masked_image)
    masked_image = np.stack(masked_volumes, axis=3)

    #global_normalize
    masked_image[masked_image!=0] = (masked_image[masked_image!=0] - np.mean(masked_image[masked_image!=0])) / np.std(masked_image[masked_image!=0])

    # Smooth IG maps
    for i in range(masked_image.shape[3]):
        volumes.append(nilearn.image.smooth_img(nib.Nifti2Image(masked_image[:,:,:,i],affine=target_affine),fwhm=7).get_fdata())

    output = np.stack(volumes, axis=3) # [112 112 112, 20] to [112 112 112 20]
    output = output.mean(axis=3) # [112 112 112 20] to [112 112 112]
    maps.append(output)

    if idx >= max_iter:
        print("MAX ITERATION REACHED")
        break

means_of_maps = np.stack(maps, axis=3)
nifti_mean_ig_map = nib.Nifti2Image(means_of_maps.mean(axis=3),affine=target_affine)

# Visualize IG maps
display = plotting.plot_stat_map(
    nifti_mean_ig_map,
    bg_img=MNI152, # Use the MNI152 template as background for better context
    threshold=1,
    display_mode='ortho', # Typically 'ortho' for 3D brain plots
    cut_coords=(0,0,0), # Let nilearn choose optimal cuts, or specify (x,y,z)
    title=f'ADNI_{config["map_task"]}_target{target}'
)

# Save the generated figure using matplotlib's savefig
timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(f'{save_dir}/ADNI_{config["map_task"]}_target{target}_{timestamp}.png', dpi=300) # dpi for higher resolution
print(f"Saved '{save_dir}/ADNI_{config['map_task']}_target{target}_{timestamp}.png'")

