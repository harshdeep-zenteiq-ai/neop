"""Batch comparison script - runs code comparison on multiple JAX-PyTorch pairs."""

from code_comparison_agent import CodeComparisonAgent
from pathlib import Path
import json


# Add your arrays here (copy-pasted from find_jax_pytorch_pairs.py output)
jax_files = ['/home/deepan/neuraloperator/neuralop/utils_jax.py', '/home/deepan/neuraloperator/neuralop/training/patching_jax.py', '/home/deepan/neuraloperator/neuralop/training/tensor_galore_projector_jax.py', '/home/deepan/neuraloperator/neuralop/training/adamw_jax.py', '/home/deepan/neuraloperator/neuralop/training/training_state_jax.py', '/home/deepan/neuraloperator/neuralop/training/trainer_jax.py', '/home/deepan/neuraloperator/neuralop/data/datasets/dict_dataset_jax.py', '/home/deepan/neuraloperator/neuralop/data/datasets/car_cfd_dataset_jax.py', '/home/deepan/neuraloperator/neuralop/data/datasets/mesh_datamodule_jax.py', '/home/deepan/neuraloperator/neuralop/data/transforms/data_processors_jax.py', '/home/deepan/neuraloperator/neuralop/data/transforms/normalizers_jax.py', '/home/deepan/neuraloperator/neuralop/data/transforms/base_transforms_jax.py', '/home/deepan/neuraloperator/neuralop/models/base_model_jax.py', '/home/deepan/neuraloperator/neuralop/models/gino_jax.py', '/home/deepan/neuraloperator/neuralop/layers/gno_weighting_functions_jax.py', '/home/deepan/neuraloperator/neuralop/layers/skip_connections_jax.py', '/home/deepan/neuraloperator/neuralop/layers/gno_block_jax.py', '/home/deepan/neuraloperator/neuralop/layers/spectral_convolution_jax.py', '/home/deepan/neuraloperator/neuralop/layers/channel_mlp_jax.py', '/home/deepan/neuraloperator/neuralop/layers/neighbor_search_jax.py', '/home/deepan/neuraloperator/neuralop/layers/resample_jax.py', '/home/deepan/neuraloperator/neuralop/layers/segment_csr_jax.py', '/home/deepan/neuraloperator/neuralop/layers/einsum_utils_jax.py', '/home/deepan/neuraloperator/neuralop/layers/integral_transform_jax.py', '/home/deepan/neuraloperator/neuralop/layers/complex_jax.py', '/home/deepan/neuraloperator/neuralop/layers/base_spectral_conv_jax.py', '/home/deepan/neuraloperator/neuralop/layers/fno_block_jax.py', '/home/deepan/neuraloperator/neuralop/layers/normalization_layers_jax.py', '/home/deepan/neuraloperator/neuralop/layers/embeddings_jax.py', '/home/deepan/neuraloperator/neuralop/losses/data_losses_jax.py', '/home/deepan/neuraloperator/scripts/train_gino_carcfd_jax.py']

pytorch_files = ['/home/deepan/neuraloperator/neuralop/utils.py', '/home/deepan/neuraloperator/neuralop/training/patching.py', '/home/deepan/neuraloperator/neuralop/training/tensor_galore_projector.py', '/home/deepan/neuraloperator/neuralop/training/adamw.py', '/home/deepan/neuraloperator/neuralop/training/training_state.py', '/home/deepan/neuraloperator/neuralop/training/trainer.py', '/home/deepan/neuraloperator/neuralop/data/datasets/dict_dataset.py', '/home/deepan/neuraloperator/neuralop/data/datasets/car_cfd_dataset.py', '/home/deepan/neuraloperator/neuralop/data/datasets/mesh_datamodule.py', '/home/deepan/neuraloperator/neuralop/data/transforms/data_processors.py', '/home/deepan/neuraloperator/neuralop/data/transforms/normalizers.py', '/home/deepan/neuraloperator/neuralop/data/transforms/base_transforms.py', '/home/deepan/neuraloperator/neuralop/models/base_model.py', '/home/deepan/neuraloperator/neuralop/models/gino.py', '/home/deepan/neuraloperator/neuralop/layers/gno_weighting_functions.py', '/home/deepan/neuraloperator/neuralop/layers/skip_connections.py', '/home/deepan/neuraloperator/neuralop/layers/gno_block.py', '/home/deepan/neuraloperator/neuralop/layers/spectral_convolution.py', '/home/deepan/neuraloperator/neuralop/layers/channel_mlp.py', '/home/deepan/neuraloperator/neuralop/layers/neighbor_search.py', '/home/deepan/neuraloperator/neuralop/layers/resample.py', '/home/deepan/neuraloperator/neuralop/layers/segment_csr.py', '/home/deepan/neuraloperator/neuralop/layers/einsum_utils.py', '/home/deepan/neuraloperator/neuralop/layers/integral_transform.py', '/home/deepan/neuraloperator/neuralop/layers/complex.py', '/home/deepan/neuraloperator/neuralop/layers/base_spectral_conv.py', '/home/deepan/neuraloperator/neuralop/layers/fno_block.py', '/home/deepan/neuraloperator/neuralop/layers/normalization_layers.py', '/home/deepan/neuraloperator/neuralop/layers/embeddings.py', '/home/deepan/neuraloperator/neuralop/losses/data_losses.py', '/home/deepan/neuraloperator/scripts/train_gino_carcfd.py']

output_dirs = ['utils_comparison', 'patching_comparison', 'tensor_galore_projector_comparison', 'adamw_comparison', 'training_state_comparison', 'trainer_comparison', 'dict_dataset_comparison', 'car_cfd_dataset_comparison', 'mesh_datamodule_comparison', 'data_processors_comparison', 'normalizers_comparison', 'base_transforms_comparison', 'base_model_comparison', 'gino_comparison', 'gno_weighting_functions_comparison', 'skip_connections_comparison', 'gno_block_comparison', 'spectral_convolution_comparison', 'channel_mlp_comparison', 'neighbor_search_comparison', 'resample_comparison', 'segment_csr_comparison', 'einsum_utils_comparison', 'integral_transform_comparison', 'complex_comparison', 'base_spectral_conv_comparison', 'fno_block_comparison', 'normalization_layers_comparison', 'embeddings_comparison', 'data_losses_comparison', 'train_gino_carcfd_comparison']


def run_batch_comparison(jax_files, pytorch_files, output_dirs, base_output_dir="./batch_comparison_results"):
    """Run code comparison on multiple JAX-PyTorch pairs.

    Parameters
    ----------
    jax_files : list
        List of JAX file paths
    pytorch_files : list
        List of PyTorch file paths
    output_dirs : list
        List of output directory names
    base_output_dir : str
        Base directory to save all results
    """
    # Validate arrays
    if not (len(jax_files) == len(pytorch_files) == len(output_dirs)):
        print("✗ Error: Arrays have different lengths!")
        print(f"  jax_files: {len(jax_files)}")
        print(f"  pytorch_files: {len(pytorch_files)}")
        print(f"  output_dirs: {len(output_dirs)}")
        return

    if len(jax_files) == 0:
        print("✗ No pairs to compare. Arrays are empty.")
        return

    # Create base output directory
    Path(base_output_dir).mkdir(parents=True, exist_ok=True)

    # Track results
    results_summary = {
        "total_pairs": len(jax_files),
        "completed": 0,
        "failed": 0,
        "results": []
    }

    print("="*70)
    print(f"BATCH COMPARISON - {len(jax_files)} pairs")
    print("="*70 + "\n")

    # Run comparison for each pair
    for i, (jax_file, pytorch_file, output_dir) in enumerate(zip(jax_files, pytorch_files, output_dirs), 1):
        print(f"\n[{i}/{len(jax_files)}] Comparing: {output_dir}")
        print("-" * 70)

        try:
            # Create agent
            agent = CodeComparisonAgent(verbose=False)

            # Create output directory for this comparison
            pair_output_dir = Path(base_output_dir) / output_dir
            pair_output_dir.mkdir(parents=True, exist_ok=True)

            # Run comparison
            result = agent.run(
                jax_file=jax_file,
                pytorch_file=pytorch_file,
                output_dir=str(pair_output_dir)
            )

            if result.get("status") == "completed":
                print(f"✓ Comparison complete")
                analysis = result.get("analysis", {})

                # Extract key info
                parity = "✓ YES" if analysis.get("overall_parity") else "✗ NO"
                score = analysis.get("compatibility_score", "N/A")
                print(f"  Parity: {parity}")
                print(f"  Compatibility Score: {score}/100")
                print(f"  Report: {result.get('report_path')}")

                results_summary["completed"] += 1
                results_summary["results"].append({
                    "pair": output_dir,
                    "status": "completed",
                    "parity": analysis.get("overall_parity"),
                    "compatibility_score": score,
                    "report_path": result.get("report_path")
                })
            else:
                print(f"✗ Comparison failed: {result.get('error', 'Unknown error')}")
                results_summary["failed"] += 1
                results_summary["results"].append({
                    "pair": output_dir,
                    "status": "failed",
                    "error": result.get("error")
                })

        except Exception as e:
            print(f"✗ Error: {e}")
            results_summary["failed"] += 1
            results_summary["results"].append({
                "pair": output_dir,
                "status": "error",
                "error": str(e)
            })

    # Print summary
    print("\n" + "="*70)
    print("BATCH COMPARISON SUMMARY")
    print("="*70)
    print(f"Total pairs: {results_summary['total_pairs']}")
    print(f"Completed: {results_summary['completed']}")
    print(f"Failed: {results_summary['failed']}")
    print(f"\nResults saved in: {base_output_dir}")

    # Save summary report
    summary_file = Path(base_output_dir) / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"Summary report: {summary_file}\n")

    # Print details
    print("Comparison Details:")
    for result in results_summary["results"]:
        status_icon = "✓" if result["status"] == "completed" else "✗"
        parity = result.get("parity", "N/A")
        score = result.get("compatibility_score", "N/A")
        print(f"  {status_icon} {result['pair']}: Parity={parity}, Score={score}")


if __name__ == "__main__":
    # Example: Uncomment and fill with your actual data
    # jax_files = ["/path/to/file1_jax.py", "/path/to/file2_jax.py"]
    # pytorch_files = ["/path/to/file1.py", "/path/to/file2.py"]
    # output_dirs = ["file1_comparison", "file2_comparison"]

    print("Batch Comparison Script")
    print("="*70)
    print("\nTo use this script:")
    print("1. Copy your arrays from find_jax_pytorch_pairs.py output")
    print("2. Paste them into the arrays at the top of this file")
    print("3. Run: python3 batch_comparison.py")
    print("\nExample:")
    print("  jax_files = ['/path/to/module1_jax.py', '/path/to/module2_jax.py']")
    print("  pytorch_files = ['/path/to/module1.py', '/path/to/module2.py']")
    print("  output_dirs = ['module1_comparison', 'module2_comparison']")
    print("\nThen call: run_batch_comparison(jax_files, pytorch_files, output_dirs)")

    # Check if arrays are populated
    if jax_files and pytorch_files and output_dirs:
        run_batch_comparison(jax_files, pytorch_files, output_dirs)
    else:
        print("\n⚠ Arrays are empty. Please populate them with your data.")
