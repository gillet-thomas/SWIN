import glob
import os
import time
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import nilearn
import nilearn.image
import numpy as np
import torch
import torchio as tio
import yaml
from captum.attr import GradientShap, IntegratedGradients, NoiseTunnel
from matplotlib.colors import LinearSegmentedColormap
from nilearn import plotting
from tqdm import tqdm

from src.data.DatasetADNI import ADNISwiFTDataset
from src.data.DatasetPain import PainDataset
from src.models.SWIN4D import SWIN4D


def generate_ig_maps():
    """
    Compute and save IG attribution maps for the target class
    Pass each sample in model forward method and compute IG attribution maps for correctly classified samples
    """
    for idx, (subject, _, _, start_frame_idx) in enumerate(tqdm(test_loader), 0):
        subj_name = subject[0]
        data_fmri, data_target = data_module[idx]
        data_fmri = data_fmri.to(device=cuda_id).unsqueeze(0)  # 1, 1, 112, 112, 112, 20
        data_target = int(data_target.item())

        pred = model.forward(data_fmri)
        pred_prob = torch.sigmoid(pred)
        pred_int = (pred_prob > 0.5).int().item()

        # only choose corrected samples
        if pred_int == data_target:  # Of the subjects that corrected
            if (data_target == 0 and pred_prob <= 0.25) or (data_target == 1 and pred_prob >= 0.75):
                file_dir = os.path.join(save_dir, f'ADNI_{config["map_task"]}_target{target}')
                os.makedirs(file_dir, exist_ok=True)
                file_path = os.path.join(file_dir, f"{subj_name}_{start_frame_idx.item()}.pt")
                if not os.path.exists(file_path):
                    # baseline = torch.zeros_like(data_fmri)
                    baseline = torch.mean(data_fmri, dim=(2, 3, 4, 5), keepdim=True).expand_as(data_fmri)
                    # result = noise_tunnel.attribute(data_fmri, baselines=baseline, target=None, **kwargs)
                    result = gradient_shap.attribute(data_fmri, baselines=baseline, target=None, **kwargs)
                    result = result.squeeze().cpu()  # 112 112 112 20

                    torch.save(result, file_path)

        if idx >= max_iter:
            print("MAX ITERATION REACHED")
            break


def collect_ig_maps():
    """
    Load IG attribution maps from .pt files and collect them in a list
    """
    maps = []

    for idx, file in enumerate(tqdm(glob.glob(IG_dir))):
        volumes = []
        masked_volumes = []
        image = torch.load(file)  # 112 112 112 20

        # Segment background
        for i in range(image.shape[3]):
            masked_image = image[:, :, :, i] * mask
            masked_volumes.append(masked_image)
        masked_image = np.stack(masked_volumes, axis=3)

        # global_normalize
        masked_image[masked_image != 0] = (
            masked_image[masked_image != 0] - np.mean(masked_image[masked_image != 0])
        ) / np.std(masked_image[masked_image != 0])

        # Smooth IG maps
        for i in range(masked_image.shape[3]):
            volumes.append(
                nilearn.image.smooth_img(
                    nib.Nifti2Image(masked_image[:, :, :, i], affine=target_affine), fwhm=7
                ).get_fdata()
            )

        output = np.stack(volumes, axis=3)  # [112 112 112, 20] to [112 112 112 20]
        output = output.mean(axis=3)  # [112 112 112 20] to [112 112 112]
        maps.append(output)

        if idx >= max_iter:
            print("MAX ITERATION REACHED")
            break

    return maps


def create_custom_colormap():
    colors = ["blue", "lightblue", "white", "lightcoral", "red"]
    n_bins = 256
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    return cmap


# Method 1: Symmetric threshold visualization
def visualize_bidirectional_attributions(nifti_mean_ig_map, MNI152, save_dir, config, target, threshold=0.3):
    """
    Visualize both positive and negative attributions with different colors
    """
    custom_cmap = create_custom_colormap()

    # Get the data and compute symmetric thresholds
    data = nifti_mean_ig_map.get_fdata()
    pos_threshold = np.percentile(data[data > 0], 100 - threshold * 100) if np.any(data > 0) else 0
    neg_threshold = np.percentile(data[data < 0], threshold * 100) if np.any(data < 0) else 0

    print(f"Positive threshold: {pos_threshold:.4f}")
    print(f"Negative threshold: {neg_threshold:.4f}")

    # Create the plot
    display = plotting.plot_stat_map(
        nifti_mean_ig_map,
        bg_img=MNI152,
        threshold=abs(neg_threshold),  # Use absolute value of negative threshold
        display_mode="ortho",
        cut_coords=None,
        title=f'ADNI_{config["map_task"]}_target{target} (Red: Important, Blue: Suppressive)',
        cmap=custom_cmap,
        symmetric_cbar=True,  # This ensures symmetric color scaling
        vmax=max(abs(pos_threshold), abs(neg_threshold)),  # Symmetric range
        colorbar=True,
    )

    return display


# Method 2: Separate positive and negative maps
def visualize_separate_pos_neg(nifti_mean_ig_map, MNI152, save_dir, config, target, threshold=1):
    """
    Create separate visualizations for positive and negative attributions
    """
    data = nifti_mean_ig_map.get_fdata()
    affine = nifti_mean_ig_map.affine

    # Create positive-only map
    pos_data = data.copy()
    pos_data[pos_data < threshold] = 0
    pos_nifti = nib.Nifti2Image(pos_data, affine=affine)

    # Create negative-only map (take absolute value for visualization)
    neg_data = data.copy()
    neg_data[neg_data > threshold] = 0
    neg_data = np.abs(neg_data)  # Make positive for visualization
    neg_nifti = nib.Nifti2Image(neg_data, affine=affine)

    # Plot positive attributions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    plt.subplot(2, 1, 1)
    display_pos = plotting.plot_stat_map(
        pos_nifti,
        bg_img=MNI152,
        threshold="auto",
        display_mode="ortho",
        cut_coords=None,
        title=f'Positive Attributions - {config["map_task"]}_target{target}',
        cmap="Reds",
        axes=ax1,
    )

    plt.subplot(2, 1, 2)
    display_neg = plotting.plot_stat_map(
        neg_nifti,
        bg_img=MNI152,
        threshold="auto",
        display_mode="ortho",
        cut_coords=None,
        title=f'Negative Attributions - {config["map_task"]}_target{target}',
        cmap="Blues",
        axes=ax2,
    )

    return fig


if __name__ == "__main__":
    # Set random seed
    config = yaml.safe_load(open("configs/config.yaml", "r"))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    warnings.simplefilter(action="ignore", category=FutureWarning)

    cuda_id = 0
    target = 0
    generate_maps = True
    max_iter = 50
    save_dir = os.path.join(os.getcwd(), "visualization/gradients")

    # Load dataset
    best_model_path = config[f'best_swin_{config["map_task"]}']
    data_module = ADNISwiFTDataset(config, mode="test")
    test_loader = torch.utils.data.DataLoader(
        data_module.data,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=True,
        prefetch_factor=2,
    )
    model = SWIN4D(config)
    model.to(device=cuda_id) if torch.cuda.is_available() else model
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    print(f"Using model from {best_model_path}")
    print(f"Running on {len(test_loader)} samples")

    # Load attribution method
    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)
    gradient_shap = GradientShap(model)

    kwargs = {
        "nt_samples": 20,
        "nt_samples_batch_size": 4,
        "nt_type": "smoothgrad_sq",  # 1
        "stdevs": 0.05,
        "internal_batch_size": 4,
    }

    # Generate IG maps
    if generate_maps:
        generate_ig_maps()

    # Importing template images for visualizations
    icbm = tio.datasets.ICBM2009CNonlinearSymmetric()
    target_affine = nib.load(
        "/mnt/data/iai/datasets/ADNI_CONN_conversion/corresponding_processed/136_S_4993_I342514/wauI342514_Resting_State_fMRI_136_S_4993.nii"
    ).affine
    target_shape = (112, 112, 112)
    MNI152 = nilearn.image.resample_img(
        icbm["t1"]["path"],
        target_affine=target_affine,
        target_shape=target_shape,
        interpolation="nearest",
    )

    # Make mask
    mask_npy = MNI152.get_fdata()
    target_affine = MNI152.affine
    mask = (mask_npy != 0).astype(int)  # Creates binary mask for background
    IG_dir = f'{save_dir}/ADNI_{config["map_task"]}_target{target}/*'

    # Collect IG maps
    maps = collect_ig_maps()
    means_of_maps = np.stack(maps, axis=3)
    nifti_mean_ig_map = nib.Nifti2Image(means_of_maps.mean(axis=3), affine=target_affine)

    # Visualize IG maps
    display = plotting.plot_stat_map(
        nifti_mean_ig_map,
        bg_img=MNI152,  # Use the MNI152 template as background for better context
        threshold=1,
        display_mode="ortho",  # Typically 'ortho' for 3D brain plots
        cut_coords=None,  # Let nilearn choose optimal cuts, or specify (x,y,z)
        title=f'ADNI_{config["map_task"]}_target{target}',
    )

    # Save the generated figure using matplotlib's savefig
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(
        f'{save_dir}/ADNI_{config["map_task"]}_target{target}_{timestamp}.png', dpi=300
    )  # dpi for higher resolution
    print(f"Saved '{save_dir}/ADNI_{config['map_task']}_target{target}_{timestamp}.png'")

    # Method 1: Bidirectional with custom colormap
    display = visualize_bidirectional_attributions(nifti_mean_ig_map, MNI152, save_dir, config, target, threshold=0.2)
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(
        f'{save_dir}/AADNI_{config["map_task"]}_target{target}_bidirectional_{timestamp}.png',
        dpi=300,
        bbox_inches="tight",
    )

    # Method 2: Separate positive and negative plots
    fig = visualize_separate_pos_neg(nifti_mean_ig_map, MNI152, save_dir, config, target, threshold=2)
    plt.savefig(
        f'{save_dir}/AADNI_{config["map_task"]}_target{target}_separate_{timestamp}.png', dpi=300, bbox_inches="tight"
    )

    print(f"Saved visualizations with timestamp {timestamp}")
