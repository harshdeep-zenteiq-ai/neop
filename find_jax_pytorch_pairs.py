"""Find all JAX-PyTorch module pairs in the directory tree."""

import os


def find_jax_pytorch_pairs(root_dir="/home/deepan/neuraloperator"):
    """Find all JAX-PyTorch module pairs in the directory tree.

    For each _jax.py file, looks for corresponding .py file (without _jax).
    Only includes complete pairs.

    Parameters
    ----------
    root_dir : str
        Root directory to search

    Returns
    -------
    dict with arrays:
        - jax_files: List of JAX file paths
        - pytorch_files: List of corresponding PyTorch file paths
        - output_dirs: List of output directory names for comparisons
    """
    jax_files = []
    pytorch_files = []
    output_dirs = []

    # Recursively search for _jax.py files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith("_jax.py"):
                jax_file_path = os.path.join(dirpath, filename)

                # Extract base name (remove _jax.py)
                # e.g., spectral_convolution_jax.py -> spectral_convolution
                base_name = filename.replace("_jax.py", "")
                pytorch_filename = base_name + ".py"
                pytorch_file_path = os.path.join(dirpath, pytorch_filename)

                # Check if PyTorch version exists
                if os.path.exists(pytorch_file_path):
                    jax_files.append(jax_file_path)
                    pytorch_files.append(pytorch_file_path)
                    output_dirs.append(f"{base_name}_comparison")

                    print(f"✓ Found pair:")
                    print(f"  JAX: {jax_file_path}")
                    print(f"  PyTorch: {pytorch_file_path}")
                    print(f"  Output dir: {base_name}_comparison\n")
                else:
                    print(f"✗ No PyTorch pair for: {jax_file_path}")
                    print(f"  Looking for: {pytorch_file_path}\n")

    return {
        "jax_files": jax_files,
        "pytorch_files": pytorch_files,
        "output_dirs": output_dirs
    }


if __name__ == "__main__":
    print("="*70)
    print("SEARCHING FOR JAX-PYTORCH MODULE PAIRS")
    print("="*70 + "\n")

    # Find all pairs
    pairs = find_jax_pytorch_pairs()

    # Print summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total pairs found: {len(pairs['jax_files'])}\n")

    if pairs['jax_files']:
        print("JAX Files:")
        for f in pairs['jax_files']:
            print(f"  {f}")

        print("\nPyTorch Files:")
        for f in pairs['pytorch_files']:
            print(f"  {f}")

        print("\nOutput Directories:")
        for d in pairs['output_dirs']:
            print(f"  {d}")

        # Print as Python arrays for easy copying
        print("\n" + "="*70)
        print("PYTHON ARRAYS (for copy-paste):")
        print("="*70)
        print(f"\njax_files = {pairs['jax_files']}")
        print(f"\npytorch_files = {pairs['pytorch_files']}")
        print(f"\noutput_dirs = {pairs['output_dirs']}")
    else:
        print("No JAX-PyTorch pairs found.")
