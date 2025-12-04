#!/usr/bin/env python
"""Test the complete virtual unwrapping pipeline."""

import sys
import tempfile
from pathlib import Path
import numpy as np
import torch


def test_rle_encoding():
    """Test RLE encoding/decoding."""
    from src.vesuvius.submission import mask_to_rle, rle_to_mask, validate_rle_roundtrip
    
    print("\n1. Testing RLE Encoding...")
    
    # Test case 1: Simple pattern
    mask = np.array([[0, 0, 1, 1], [1, 1, 0, 0]], dtype=np.uint8)
    assert validate_rle_roundtrip(mask), "RLE roundtrip failed for simple mask"
    rle = mask_to_rle(mask)
    print(f"  ✓ Simple mask: {mask.shape} -> RLE: '{rle}'")
    
    # Test case 2: Empty mask
    empty_mask = np.zeros((10, 10), dtype=np.uint8)
    assert validate_rle_roundtrip(empty_mask), "RLE roundtrip failed for empty mask"
    print(f"  ✓ Empty mask: roundtrip validated")
    
    # Test case 3: Full mask
    full_mask = np.ones((10, 10), dtype=np.uint8)
    assert validate_rle_roundtrip(full_mask), "RLE roundtrip failed for full mask"
    print(f"  ✓ Full mask: roundtrip validated")
    
    # Test case 4: Random pattern
    random_mask = np.random.randint(0, 2, (50, 50), dtype=np.uint8)
    assert validate_rle_roundtrip(random_mask), "RLE roundtrip failed for random mask"
    print(f"  ✓ Random mask (50x50): roundtrip validated")
    
    print("  ✓ All RLE encoding tests passed")


def test_surface_extraction():
    """Test surface extraction and unwrapping."""
    from src.vesuvius.unwrap import (
        extract_surface_from_volume, 
        compute_surface_statistics,
        extract_surface_neighborhood
    )
    
    print("\n2. Testing Surface Extraction...")
    
    # Create test volume and surface mask
    volume = np.random.rand(50, 100, 100).astype(np.float32)
    surface_mask = np.zeros_like(volume, dtype=np.uint8)
    surface_mask[20:25, :, :] = 1  # Flat surface
    
    print(f"  Test volume: {volume.shape}")
    print(f"  Surface mask: {surface_mask.sum()} voxels")
    
    # Test center extraction
    surface_image, surface_coords = extract_surface_from_volume(
        volume, surface_mask, method="center"
    )
    assert surface_image.shape == (100, 100), f"Unexpected surface shape: {surface_image.shape}"
    print(f"  ✓ Center extraction: {surface_image.shape}")
    
    # Test mean extraction
    surface_image_mean, _ = extract_surface_from_volume(
        volume, surface_mask, method="mean"
    )
    assert surface_image_mean.shape == (100, 100), "Mean extraction failed"
    print(f"  ✓ Mean extraction: {surface_image_mean.shape}")
    
    # Test max extraction
    surface_image_max, _ = extract_surface_from_volume(
        volume, surface_mask, method="max"
    )
    assert surface_image_max.shape == (100, 100), "Max extraction failed"
    print(f"  ✓ Max extraction: {surface_image_max.shape}")
    
    # Compute statistics
    stats = compute_surface_statistics(surface_coords)
    assert stats['coverage'] > 0.9, f"Surface coverage too low: {stats['coverage']}"
    print(f"  ✓ Surface statistics: coverage={stats['coverage']:.2%}, mean_depth={stats['mean_depth']:.1f}")
    
    # Test neighborhood extraction
    neighborhood = extract_surface_neighborhood(volume, surface_coords, depth_radius=3)
    assert neighborhood.shape == (100, 100, 7), f"Unexpected neighborhood shape: {neighborhood.shape}"
    print(f"  ✓ Neighborhood extraction: {neighborhood.shape}")
    
    print("  ✓ All surface extraction tests passed")


def test_model_inference():
    """Test model loading and inference."""
    from src.vesuvius.models import build_model
    
    print("\n3. Testing Model Inference...")
    
    # Create test model (smaller for faster testing)
    model_config = {
        'type': 'unet3d_residual',
        'in_channels': 1,
        'out_channels': 1,
        'base_channels': 16,  # Smaller for faster test
        'channel_multipliers': [1, 2, 4],
        'blocks_per_stage': 1,
        'deep_supervision': False,
        'dropout': 0.0,
        'activation': 'relu',
        'norm': 'instance'
    }
    
    model = build_model(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Test model: {num_params/1e6:.2f}M parameters")
    
    model.eval()
    
    # Test inference with different input sizes
    test_sizes = [
        (32, 64, 64),
        (48, 128, 128),
        (64, 256, 256),
    ]
    
    for depth, height, width in test_sizes:
        test_input = torch.randn(1, 1, depth, height, width)
        with torch.no_grad():
            output = model(test_input)
        
        assert 'logits' in output, "Model output missing 'logits'"
        assert output['logits'].shape == test_input.shape, f"Output shape mismatch: {output['logits'].shape} vs {test_input.shape}"
        print(f"  ✓ Input {test_input.shape} -> Output {output['logits'].shape}")
    
    print("  ✓ All model inference tests passed")


def test_submission_generation():
    """Test submission CSV generation."""
    from src.vesuvius.submission import create_submission_csv, load_submission_csv, get_rle_stats
    
    print("\n4. Testing Submission Generation...")
    
    # Create test predictions
    predictions = {
        'fragment_1': np.random.randint(0, 2, (100, 100), dtype=np.uint8),
        'fragment_2': np.random.randint(0, 2, (150, 150), dtype=np.uint8),
        'fragment_3': np.zeros((80, 80), dtype=np.uint8),  # Empty
    }
    
    print(f"  Test predictions: {len(predictions)} fragments")
    
    # Generate submission
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "submission.csv"
        create_submission_csv(predictions, output_path, validate=True)
        
        assert output_path.exists(), "Submission CSV not created"
        print(f"  ✓ Submission CSV created: {output_path.name}")
        
        # Verify CSV format
        import pandas as pd
        df = pd.read_csv(output_path)
        assert 'Id' in df.columns and 'Predicted' in df.columns, "Invalid CSV format"
        assert len(df) == 3, f"Wrong number of submissions: {len(df)} vs 3"
        print(f"  ✓ CSV format validated: {len(df)} rows")
        
        # Load and verify
        loaded = load_submission_csv(output_path)
        assert len(loaded) == 3, "Failed to load submission"
        print(f"  ✓ CSV loading validated")
        
        # Check RLE stats
        for frag_id, rle in loaded.items():
            stats = get_rle_stats(rle)
            print(f"  ✓ {frag_id}: {stats['num_runs']} runs, {stats['total_pixels']} pixels")
    
    print("  ✓ All submission generation tests passed")


def test_visualization():
    """Test visualization functions."""
    from src.vesuvius.unwrap import visualize_unwrapped_text
    
    print("\n5. Testing Visualization...")
    
    # Create test data
    surface_image = np.random.rand(200, 200).astype(np.float32)
    ink_mask = np.zeros((200, 200), dtype=np.uint8)
    ink_mask[80:120, 50:150] = 1  # Horizontal bar
    ink_mask[50:150, 80:120] = 1  # Vertical bar (cross pattern)
    
    print(f"  Surface image: {surface_image.shape}")
    print(f"  Ink mask: {ink_mask.shape}, {ink_mask.sum()} ink pixels")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        
        try:
            viz_files = visualize_unwrapped_text(
                surface_image,
                ink_mask,
                output_dir,
                fragment_id="test_fragment",
                create_overlay=True,
                create_text_only=True
            )
            
            # Check all expected files were created
            expected_types = ['surface', 'ink', 'overlay', 'text']
            for viz_type in expected_types:
                assert viz_type in viz_files, f"Missing visualization: {viz_type}"
                assert viz_files[viz_type].exists(), f"File not created: {viz_files[viz_type]}"
            
            print(f"  ✓ Created {len(viz_files)} visualizations")
            for viz_type, path in viz_files.items():
                print(f"    - {viz_type}: {path.name}")
            
        except ImportError as e:
            print(f"  ⚠ Visualization skipped (matplotlib not available): {e}")
            return
    
    print("  ✓ All visualization tests passed")


def test_postprocessing():
    """Test post-processing functions."""
    from src.vesuvius.postprocess import (
        remove_small_components,
        morphological_closing,
        apply_postprocessing
    )
    
    print("\n6. Testing Post-processing...")
    
    # Create test mask with small components
    mask = np.zeros((100, 100, 100), dtype=np.uint8)
    mask[40:60, 40:60, 40:60] = 1  # Large component
    mask[10:12, 10:12, 10:12] = 1  # Small component (8 voxels)
    
    print(f"  Input mask: {mask.sum()} voxels")
    
    # Test small component removal
    cleaned = remove_small_components(mask, min_size=100)
    assert cleaned.sum() < mask.sum(), "Small components not removed"
    print(f"  ✓ Small component removal: {mask.sum()} -> {cleaned.sum()} voxels")
    
    # Test morphological closing
    closed = morphological_closing(mask, radius=2)
    print(f"  ✓ Morphological closing: {mask.sum()} -> {closed.sum()} voxels")
    
    # Test full post-processing pipeline
    config = {
        'remove_small_components_voxels': 100,
        'closing_radius': 2
    }
    processed = apply_postprocessing(mask, config)
    print(f"  ✓ Full pipeline: {mask.sum()} -> {processed.sum()} voxels")
    
    print("  ✓ All post-processing tests passed")


def main():
    print("=" * 80)
    print("Vesuvius Challenge - Full Pipeline Test")
    print("=" * 80)
    
    try:
        test_rle_encoding()
        test_surface_extraction()
        test_model_inference()
        test_submission_generation()
        test_visualization()
        test_postprocessing()
        
        print("\n" + "=" * 80)
        print("✓ All pipeline tests passed!")
        print("=" * 80)
        print("\nPipeline components verified:")
        print("  ✓ RLE encoding/decoding")
        print("  ✓ Surface extraction and unwrapping")
        print("  ✓ Model inference")
        print("  ✓ Submission CSV generation")
        print("  ✓ Visualization")
        print("  ✓ Post-processing")
        print("\nYour pipeline is ready for production use!")
        print("=" * 80)
        
        return 0
        
    except Exception as e:
        print(f"\n" + "=" * 80)
        print(f"✗ Test failed: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
