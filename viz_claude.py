import glob
import os
import time
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import nilearn
import nilearn.image
import numpy as np
import seaborn as sns
import torch
import torchio as tio
import yaml
from captum.attr import (GradientShap, GuidedBackprop, IntegratedGradients,
                         NoiseTunnel, Occlusion, Saliency)
from nilearn import plotting
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from SWIN import ADNISwiFTDataset, Model


class fMRIAttributionAnalyzer:
    def __init__(self, config, cuda_id=0):
        self.config = config
        self.cuda_id = cuda_id
        self.device = torch.device(
            f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu"
        )
        self.setup_model()
        self.setup_attribution_methods()

    def setup_model(self):
        """Initialize model and data loaders"""
        best_model_path = self.config[f'best_swin_{self.config["map_task"]}']
        self.data_module = ADNISwiFTDataset(self.config, mode="test")
        self.test_loader = torch.utils.data.DataLoader(
            self.data_module.data,
            batch_size=self.config["eval_batch_size"],
            shuffle=False,
            num_workers=self.config["num_workers"],
            pin_memory=True,
            prefetch_factor=2,
        )

        self.model = Model(self.config)
        self.model.to(self.device)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        print(f"Using model from {best_model_path}")

    def setup_attribution_methods(self):
        """Initialize multiple attribution methods for comparison"""
        self.attribution_methods = {
            "integrated_gradients": IntegratedGradients(self.model),
            "gradient_shap": GradientShap(self.model),
            "occlusion": Occlusion(self.model),
            "saliency": Saliency(self.model),
            "guided_backprop": GuidedBackprop(self.model),
        }

    def get_better_baseline(self, data_fmri):
        """Generate more meaningful baselines for fMRI data"""
        baselines = {
            "zero": torch.zeros_like(data_fmri),
            "mean": torch.mean(data_fmri, dim=(2, 3, 4, 5), keepdim=True).expand_as(
                data_fmri
            ),
            "gaussian_noise": torch.randn_like(data_fmri) * 0.1,
            "temporal_mean": torch.mean(data_fmri, dim=5, keepdim=True).expand_as(
                data_fmri
            ),
        }
        return baselines

    def compute_attributions(
        self, data_fmri, method_name="integrated_gradients", baseline_type="zero"
    ):
        """Compute attributions with different methods and baselines"""
        baselines = self.get_better_baseline(data_fmri)
        baseline = baselines[baseline_type]

        if method_name == "integrated_gradients":
            # Use NoiseTunnel for more stable IG
            noise_tunnel = NoiseTunnel(self.attribution_methods[method_name])
            attribution = noise_tunnel.attribute(
                data_fmri,
                baselines=baseline,
                target=None,
                nt_samples=10,  # Increased for better stability
                nt_samples_batch_size=5,
                nt_type="smoothgrad_sq",
                stdevs=0.05,  # Added explicit noise level
                internal_batch_size=2,
            )
        elif method_name == "gradient_shap":
            attribution = self.attribution_methods[method_name].attribute(
                data_fmri, baselines=baseline, target=None, n_samples=50, stdevs=0.05
            )
        elif method_name == "occlusion":
            attribution = self.attribution_methods[method_name].attribute(
                data_fmri,
                target=None,
                sliding_window_shapes=(1, 8, 8, 8, 4),  # Adapted for fMRI dimensions
                strides=(1, 4, 4, 4, 2),
            )
        else:
            attribution = self.attribution_methods[method_name].attribute(
                data_fmri, target=None
            )

        return attribution.squeeze().cpu()

    def sanity_checks(self, data_fmri, attributions):
        """Perform sanity checks on attributions"""
        checks = {}

        # 1. Model Parameter Randomization Test
        original_state = self.model.state_dict().copy()

        # Randomize model parameters
        for param in self.model.parameters():
            param.data = torch.randn_like(param.data)

        random_attr = self.compute_attributions(data_fmri, "saliency", "zero")

        # Restore original parameters
        self.model.load_state_dict(original_state)

        # Compare attributions
        correlation = pearsonr(
            attributions.flatten().numpy(), random_attr.flatten().numpy()
        )[0]
        checks["parameter_randomization"] = correlation

        # 2. Input Invariance Test
        shifted_input = torch.roll(data_fmri, shifts=1, dims=2)
        shifted_attr = self.compute_attributions(shifted_input, "saliency", "zero")

        # Should be different for meaningful attributions
        input_invariance = pearsonr(
            attributions.flatten().numpy(), shifted_attr.flatten().numpy()
        )[0]
        checks["input_invariance"] = input_invariance

        # 3. Sensitivity Test (small perturbations)
        noise = torch.randn_like(data_fmri) * 0.01
        noisy_input = data_fmri + noise
        noisy_attr = self.compute_attributions(noisy_input, "saliency", "zero")

        sensitivity = pearsonr(
            attributions.flatten().numpy(), noisy_attr.flatten().numpy()
        )[0]
        checks["sensitivity"] = sensitivity

        return checks

    def cross_method_validation(self, data_fmri):
        """Compare attributions across different methods"""
        methods = ["integrated_gradients", "gradient_shap", "saliency"]
        attributions = {}

        for method in methods:
            try:
                attr = self.compute_attributions(data_fmri, method, "zero")
                attributions[method] = attr
            except Exception as e:
                print(f"Error with {method}: {e}")
                continue

        # Compute correlations between methods
        correlations = {}
        method_names = list(attributions.keys())

        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names[i + 1 :], i + 1):
                corr = pearsonr(
                    attributions[method1].flatten().numpy(),
                    attributions[method2].flatten().numpy(),
                )[0]
                correlations[f"{method1}_vs_{method2}"] = corr

        return attributions, correlations

    def improved_preprocessing(self, attribution_map, mask, target_affine):
        """Improved preprocessing with better normalization and smoothing"""
        # Handle different dimensionalities
        if attribution_map.ndim == 4 and mask.ndim == 3:
            # Expand mask to match temporal dimension
            mask_expanded = np.expand_dims(mask, axis=3)
            masked_attr = attribution_map * mask_expanded
        else:
            masked_attr = attribution_map * mask

        # Robust normalization (using median and MAD instead of mean/std)
        non_zero_values = masked_attr[masked_attr != 0]
        if len(non_zero_values) > 0:
            median_val = np.median(non_zero_values)
            mad_val = np.median(np.abs(non_zero_values - median_val))
            if mad_val > 0:
                masked_attr[masked_attr != 0] = (
                    masked_attr[masked_attr != 0] - median_val
                ) / mad_val

        # Adaptive smoothing based on signal strength
        if attribution_map.ndim == 4:  # Temporal dimension
            smoothed_volumes = []
            for t in range(attribution_map.shape[3]):
                # Adaptive FWHM based on signal strength
                signal_strength = np.std(masked_attr[:, :, :, t])
                adaptive_fwhm = max(3, min(7, 5 / (signal_strength + 1e-6)))

                smoothed = nilearn.image.smooth_img(
                    nib.Nifti2Image(masked_attr[:, :, :, t], affine=target_affine),
                    fwhm=adaptive_fwhm,
                ).get_fdata()
                smoothed_volumes.append(smoothed)

            output = np.stack(smoothed_volumes, axis=3)
            # Temporal aggregation with weighted average (higher weights for stronger signals)
            weights = np.std(output, axis=(0, 1, 2))
            weights = weights / np.sum(weights)
            output = np.average(output, axis=3, weights=weights)
        else:
            output = nilearn.image.smooth_img(
                nib.Nifti2Image(masked_attr, affine=target_affine), fwhm=5
            ).get_fdata()

        return output

    def generate_reliability_report(self, subject_name, data_fmri, target_dir):
        """Generate comprehensive reliability report"""
        print(f"\n=== Reliability Analysis for {subject_name} ===")

        # 1. Cross-method validation
        attributions, correlations = self.cross_method_validation(data_fmri)

        print("\nCross-method correlations:")
        for pair, corr in correlations.items():
            print(f"{pair}: {corr:.3f}")

        # 2. Sanity checks on best attribution
        best_method = max(correlations.items(), key=lambda x: abs(x[1]))[0].split(
            "_vs_"
        )[0]
        best_attribution = attributions[best_method]

        sanity_results = self.sanity_checks(data_fmri, best_attribution)
        print(f"\nSanity checks for {best_method}:")
        for check, result in sanity_results.items():
            print(f"{check}: {result:.3f}")

        # 3. Generate visualization
        self.create_reliability_visualization(
            attributions, correlations, sanity_results, subject_name, target_dir
        )

        return {
            "attributions": attributions,
            "correlations": correlations,
            "sanity_checks": sanity_results,
            "recommended_method": best_method,
        }

    def create_reliability_visualization(
        self, attributions, correlations, sanity_results, subject_name, target_dir
    ):
        """Create comprehensive visualization of reliability metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Cross-method correlations
        methods = list(correlations.keys())
        corr_values = list(correlations.values())

        axes[0, 0].bar(range(len(methods)), corr_values)
        axes[0, 0].set_xticks(range(len(methods)))
        axes[0, 0].set_xticklabels(methods, rotation=45, ha="right")
        axes[0, 0].set_ylabel("Correlation")
        axes[0, 0].set_title("Cross-Method Correlations")
        axes[0, 0].set_ylim(-1, 1)

        # Plot 2: Sanity check results
        sanity_names = list(sanity_results.keys())
        sanity_values = list(sanity_results.values())

        axes[0, 1].bar(range(len(sanity_names)), sanity_values)
        axes[0, 1].set_xticks(range(len(sanity_names)))
        axes[0, 1].set_xticklabels(sanity_names, rotation=45, ha="right")
        axes[0, 1].set_ylabel("Correlation")
        axes[0, 1].set_title("Sanity Check Results")
        axes[0, 1].set_ylim(-1, 1)

        # Plot 3: Attribution distribution comparison
        for i, (method, attr) in enumerate(attributions.items()):
            axes[1, 0].hist(attr.flatten().numpy(), bins=50, alpha=0.7, label=method)
        axes[1, 0].set_xlabel("Attribution Value")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Attribution Value Distributions")
        axes[1, 0].legend()
        axes[1, 0].set_yscale("log")

        # Plot 4: Method agreement heatmap
        method_names = list(attributions.keys())
        agreement_matrix = np.zeros((len(method_names), len(method_names)))

        for i, method1 in enumerate(method_names):
            for j, method2 in enumerate(method_names):
                if i != j:
                    corr = pearsonr(
                        attributions[method1].flatten().numpy(),
                        attributions[method2].flatten().numpy(),
                    )[0]
                    agreement_matrix[i, j] = corr
                else:
                    agreement_matrix[i, j] = 1.0

        im = axes[1, 1].imshow(agreement_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        axes[1, 1].set_xticks(range(len(method_names)))
        axes[1, 1].set_yticks(range(len(method_names)))
        axes[1, 1].set_xticklabels(method_names, rotation=45, ha="right")
        axes[1, 1].set_yticklabels(method_names)
        axes[1, 1].set_title("Method Agreement Matrix")
        plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig(
            os.path.join(target_dir, f"{subject_name}_reliability_report.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()


def main():
    """Main execution function with reliability analysis"""
    config = yaml.safe_load(open("configs/config.yaml", "r"))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    warnings.simplefilter(action="ignore", category=FutureWarning)

    # Initialize analyzer
    analyzer = fMRIAttributionAnalyzer(config, cuda_id=0)

    # Setup directories and templates
    save_dir = os.path.join(os.getcwd(), "visualization/gradients_reliable")
    os.makedirs(save_dir, exist_ok=True)

    # Load template and mask
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
    mask = (MNI152.get_fdata() != 0).astype(int)

    # Process subjects with reliability analysis
    reliable_maps = []
    reliability_reports = []

    for idx, (subject, _, _, start_frame_idx) in enumerate(tqdm(analyzer.test_loader)):
        if idx >= 5:  # Process fewer subjects for detailed analysis
            break

        subj_name = subject[0]
        data_fmri, data_target = analyzer.data_module[idx]
        data_fmri = data_fmri.to(analyzer.device).unsqueeze(0)
        data_target = int(data_target.item())

        # Check prediction accuracy
        pred = analyzer.model.forward(data_fmri)
        pred_prob = torch.sigmoid(pred)
        pred_int = (pred_prob > 0.5).int().item()

        if pred_int == data_target:
            if (data_target == 0 and pred_prob <= 0.25) or (
                data_target == 1 and pred_prob >= 0.75
            ):
                # Generate reliability report
                reliability_report = analyzer.generate_reliability_report(
                    subj_name, data_fmri, save_dir
                )

                # Use most reliable method
                best_method = reliability_report["recommended_method"]
                best_attribution = reliability_report["attributions"][best_method]

                # Improved preprocessing
                processed_map = analyzer.improved_preprocessing(
                    best_attribution.numpy(), mask, target_affine
                )

                reliable_maps.append(processed_map)
                reliability_reports.append(reliability_report)

                print(f"Processed {subj_name} with {best_method} method")

    # Create final visualization with reliability metrics
    if reliable_maps:
        mean_reliable_map = np.mean(np.stack(reliable_maps), axis=0)
        nifti_mean_map = nib.Nifti2Image(mean_reliable_map, affine=target_affine)

        # Generate final plot
        display = plotting.plot_stat_map(
            nifti_mean_map,
            bg_img=MNI152,
            threshold=0.5,
            display_mode="ortho",
            title=f"Reliable Attribution Map (n={len(reliable_maps)})",
        )

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f"{save_dir}/reliable_attribution_map_{timestamp}.png", dpi=300)
        print(f"Saved reliable attribution map to {save_dir}")

        # Save reliability summary
        with open(f"{save_dir}/reliability_summary_{timestamp}.txt", "w") as f:
            f.write("=== Reliability Summary ===\n\n")
            f.write(f"Number of subjects analyzed: {len(reliability_reports)}\n")

            # Aggregate reliability metrics
            all_correlations = {}
            all_sanity_checks = {}

            for report in reliability_reports:
                for pair, corr in report["correlations"].items():
                    if pair not in all_correlations:
                        all_correlations[pair] = []
                    all_correlations[pair].append(corr)

                for check, result in report["sanity_checks"].items():
                    if check not in all_sanity_checks:
                        all_sanity_checks[check] = []
                    all_sanity_checks[check].append(result)

            f.write("\nAverage Cross-Method Correlations:\n")
            for pair, corrs in all_correlations.items():
                f.write(f"{pair}: {np.mean(corrs):.3f} ± {np.std(corrs):.3f}\n")

            f.write("\nAverage Sanity Check Results:\n")
            for check, results in all_sanity_checks.items():
                f.write(f"{check}: {np.mean(results):.3f} ± {np.std(results):.3f}\n")

        print(f"Saved reliability summary to {save_dir}")


if __name__ == "__main__":
    main()
