#!/usr/bin/env python3
"""
Verify and inspect generated mixture datasets
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def verify_mixture_files(base_path):
    """
    Verify the integrity and properties of generated mixture files
    
    Args:
        base_path: Base path without extension (e.g., 'dataset/training_mixtures/demodtrain_QPSK_CommSignal2')
    """
    print(f"\n{'='*70}")
    print(f"Verifying: {Path(base_path).name}")
    print(f"{'='*70}\n")
    
    # Load files
    try:
        mixtures = np.load(f"{base_path}_mixtures.npy")
        soi_gt = np.load(f"{base_path}_soi_groundtruth.npy")
        with open(f"{base_path}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        print("✓ All files loaded successfully")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        return False
    
    # Check shapes
    print(f"\nData Shapes:")
    print(f"  Mixtures:       {mixtures.shape}")
    print(f"  SOI GT:         {soi_gt.shape}")
    print(f"  Metadata:       {len(metadata)} entries")
    
    assert mixtures.shape == soi_gt.shape, "Shape mismatch between mixtures and ground truth!"
    assert len(metadata) == len(mixtures), "Metadata count doesn't match number of mixtures!"
    print("✓ Shape consistency check passed")
    
    # Check data types
    print(f"\nData Types:")
    print(f"  Mixtures:       {mixtures.dtype}")
    print(f"  SOI GT:         {soi_gt.dtype}")
    
    assert mixtures.dtype == np.complex64, f"Expected complex64, got {mixtures.dtype}"
    print("✓ Data type check passed")
    
    # Analyze SINR distribution
    target_sinrs = np.array([m['target_sinr_db'] for m in metadata])
    actual_sinrs = np.array([m['actual_sinr_db'] for m in metadata])
    
    unique_target_sinrs = np.unique(target_sinrs)
    
    print(f"\nSINR Levels:")
    print(f"  Number of levels:    {len(unique_target_sinrs)}")
    print(f"  SINR range:          {target_sinrs.min():.2f} to {target_sinrs.max():.2f} dB")
    print(f"  Samples per level:   {len(target_sinrs) // len(unique_target_sinrs)}")
    
    print(f"\nSINR Accuracy:")
    sinr_errors = actual_sinrs - target_sinrs
    print(f"  Mean error:          {sinr_errors.mean():.4f} dB")
    print(f"  Max error:           {sinr_errors.max():.4f} dB")
    print(f"  Std error:           {sinr_errors.std():.4f} dB")
    
    # Check for any NaN or Inf values
    has_nan = np.any(np.isnan(mixtures)) or np.any(np.isnan(soi_gt))
    has_inf = np.any(np.isinf(mixtures)) or np.any(np.isinf(soi_gt))
    
    print(f"\nData Quality:")
    print(f"  Contains NaN:        {'✗ YES' if has_nan else '✓ No'}")
    print(f"  Contains Inf:        {'✗ YES' if has_inf else '✓ No'}")
    
    # Power statistics
    mixture_powers_db = 10 * np.log10(np.mean(np.abs(mixtures)**2, axis=1))
    soi_powers_db = 10 * np.log10(np.mean(np.abs(soi_gt)**2, axis=1))
    
    print(f"\nPower Statistics:")
    print(f"  Mixture power:       {mixture_powers_db.mean():.2f} ± {mixture_powers_db.std():.2f} dB")
    print(f"  SOI power:           {soi_powers_db.mean():.2f} ± {soi_powers_db.std():.2f} dB")
    
    # Check metadata consistency
    soi_types = set(m['soi_type'] for m in metadata)
    int_types = set(m['interference_type'] for m in metadata)
    
    print(f"\nMetadata:")
    print(f"  SOI types:           {', '.join(soi_types)}")
    print(f"  Interference types:  {', '.join(int_types)}")
    print(f"  Unique SOI files:    {len(set(m['soi_file'] for m in metadata))}")
    print(f"  Unique Int files:    {len(set(m['int_file'] for m in metadata))}")
    
    print(f"\n{'='*70}")
    print("✅ Verification complete - All checks passed!")
    print(f"{'='*70}\n")
    
    return True


def plot_mixture_examples(base_path, n_examples=3, output_dir='verification_plots'):
    """
    Plot example mixtures and their components
    
    Args:
        base_path: Base path to mixture files
        n_examples: Number of examples to plot
        output_dir: Directory to save plots
    """
    # Load data
    mixtures = np.load(f"{base_path}_mixtures.npy")
    soi_gt = np.load(f"{base_path}_soi_groundtruth.npy")
    with open(f"{base_path}_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Select examples from different SINR levels
    target_sinrs = np.array([m['target_sinr_db'] for m in metadata])
    unique_sinrs = np.unique(target_sinrs)
    
    # Pick low, medium, and high SINR examples
    selected_sinrs = [unique_sinrs[0], unique_sinrs[len(unique_sinrs)//2], unique_sinrs[-1]]
    
    for sinr_val in selected_sinrs:
        # Find an example at this SINR
        idx = np.where(target_sinrs == sinr_val)[0][0]
        
        mixture = mixtures[idx]
        soi = soi_gt[idx]
        interference = mixture - soi
        meta = metadata[idx]
        
        # Create figure
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        fig.suptitle(f"{meta['soi_type']} vs {meta['interference_type']} | "
                    f"SINR = {meta['actual_sinr_db']:.2f} dB", fontsize=14, fontweight='bold')
        
        # Plot mixture
        axes[0, 0].plot(np.real(mixture[:1000]), label='Real', linewidth=0.8)
        axes[0, 0].plot(np.imag(mixture[:1000]), label='Imag', linewidth=0.8)
        axes[0, 0].set_title('Mixture (time domain)')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot mixture spectrum
        mixture_fft = np.fft.fftshift(np.fft.fft(mixture))
        axes[0, 1].plot(10 * np.log10(np.abs(mixture_fft)**2 + 1e-10), linewidth=0.8)
        axes[0, 1].set_title('Mixture (frequency domain)')
        axes[0, 1].set_xlabel('FFT Bin')
        axes[0, 1].set_ylabel('Power (dB)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot SOI
        axes[1, 0].plot(np.real(soi[:1000]), label='Real', linewidth=0.8)
        axes[1, 0].plot(np.imag(soi[:1000]), label='Imag', linewidth=0.8)
        axes[1, 0].set_title('SOI (time domain)')
        axes[1, 0].set_xlabel('Sample')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot SOI spectrum
        soi_fft = np.fft.fftshift(np.fft.fft(soi))
        axes[1, 1].plot(10 * np.log10(np.abs(soi_fft)**2 + 1e-10), linewidth=0.8)
        axes[1, 1].set_title('SOI (frequency domain)')
        axes[1, 1].set_xlabel('FFT Bin')
        axes[1, 1].set_ylabel('Power (dB)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot interference
        axes[2, 0].plot(np.real(interference[:1000]), label='Real', linewidth=0.8)
        axes[2, 0].plot(np.imag(interference[:1000]), label='Imag', linewidth=0.8)
        axes[2, 0].set_title('Interference (time domain)')
        axes[2, 0].set_xlabel('Sample')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # Plot interference spectrum
        int_fft = np.fft.fftshift(np.fft.fft(interference))
        axes[2, 1].plot(10 * np.log10(np.abs(int_fft)**2 + 1e-10), linewidth=0.8)
        axes[2, 1].set_title('Interference (frequency domain)')
        axes[2, 1].set_xlabel('FFT Bin')
        axes[2, 1].set_ylabel('Power (dB)')
        axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        filename = f"{Path(base_path).name}_SINR_{int(sinr_val):+03d}dB.png"
        output_path = Path(output_dir) / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved plot: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Verify generated mixture datasets')
    
    parser.add_argument('--mixture_dir', type=str, default='dataset/training_mixtures',
                       help='Directory containing mixture files')
    parser.add_argument('--verify_all', action='store_true',
                       help='Verify all mixture files in directory')
    parser.add_argument('--file_prefix', type=str,
                       help='Specific file prefix to verify (e.g., demodtrain_QPSK_CommSignal2)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--plot_dir', type=str, default='verification_plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    
    mixture_dir = Path(args.mixture_dir)
    
    if not mixture_dir.exists():
        print(f"Error: Directory not found: {mixture_dir}")
        return
    
    # Find all mixture files
    if args.verify_all:
        mixture_files = sorted(mixture_dir.glob("*_mixtures.npy"))
        file_bases = [str(f).replace('_mixtures.npy', '') for f in mixture_files]
        
        if not file_bases:
            print(f"No mixture files found in {mixture_dir}")
            return
        
        print(f"Found {len(file_bases)} mixture datasets to verify\n")
        
        for base_path in file_bases:
            verify_mixture_files(base_path)
            
            if args.plot:
                print("Generating plots...")
                plot_mixture_examples(base_path, output_dir=args.plot_dir)
                print()
    
    elif args.file_prefix:
        base_path = mixture_dir / args.file_prefix
        verify_mixture_files(str(base_path))
        
        if args.plot:
            print("\nGenerating plots...")
            plot_mixture_examples(str(base_path), output_dir=args.plot_dir)
    
    else:
        print("Please specify either --verify_all or --file_prefix")
        print("\nExample usage:")
        print("  python verify_mixtures.py --verify_all")
        print("  python verify_mixtures.py --file_prefix demodtrain_QPSK_CommSignal2 --plot")


if __name__ == '__main__':
    main()
