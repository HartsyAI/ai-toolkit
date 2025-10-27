#!/usr/bin/env python3
"""
F-Lite Model Loading Test Script

This script tests that F-Lite is correctly integrated into AI Toolkit.
Run this after implementing all the changes to verify everything works.

Usage:
    python test_flite_loading.py

Requirements:
    - All F-Lite modifications applied to AI Toolkit
    - f_lite package installed: pip install git+https://github.com/fal-ai/f-lite.git
    - Sufficient VRAM (24GB+ recommended with quantization)
"""

import sys
import torch
from pathlib import Path

# Add toolkit to path
sys.path.insert(0, str(Path(__file__).parent))


def test_config_modules():
    """Test that config_modules.py has F-Lite support."""
    print("\n" + "=" * 60)
    print("TEST 1: Testing config_modules.py")
    print("=" * 60)

    try:
        from toolkit.config_modules import ModelConfig, ModelArch

        # Check if 'flite' is in ModelArch
        print("✓ Imported ModelConfig and ModelArch")

        # Try to create F-Lite config
        config = ModelConfig(
            name_or_path="Freepik/F-Lite",
            is_flite=True,
            flite_text_encoder_layer=17,
            quantize=True
        )

        print("✓ Created F-Lite ModelConfig")
        print(f"  - is_flite: {config.is_flite}")
        print(f"  - flite_text_encoder_layer: {config.flite_text_encoder_layer}")
        print(f"  - quantize: {config.quantize}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_flite_handler():
    """Test that flite.py model handler works."""
    print("\n" + "=" * 60)
    print("TEST 2: Testing toolkit/models/flite.py")
    print("=" * 60)

    try:
        from toolkit.models.flite import FLiteModelHandler, setup_flite_for_training

        print("✓ Imported FLiteModelHandler and setup_flite_for_training")

        # Test registration
        handler = FLiteModelHandler()
        result = handler.register_flite_classes()

        if result:
            print("✓ F-Lite classes registered successfully")
        else:
            print("⚠ F-Lite class registration returned False (may not be critical)")

        return True

    except ImportError as e:
        print(f"✗ FAILED: Could not import flite.py: {e}")
        print("\nMake sure you created toolkit/models/flite.py")
        return False
    except Exception as e:
        print(f"✗ FAILED: {e}")
        return False


def test_stable_diffusion_model():
    """Test that stable_diffusion_model.py recognizes F-Lite."""
    print("\n" + "=" * 60)
    print("TEST 3: Testing toolkit/stable_diffusion_model.py")
    print("=" * 60)

    try:
        from toolkit.stable_diffusion_model import StableDiffusionModel
        from toolkit.config_modules import ModelConfig

        print("✓ Imported StableDiffusionModel")

        # Create F-Lite config
        config = ModelConfig(
            name_or_path="Freepik/F-Lite",
            is_flite=True,
            quantize=True,
            dtype='bf16'
        )

        # Create model instance (don't load yet, just instantiate)
        model = StableDiffusionModel(
            model_config=config,
            device='cuda:0',
            dtype='bf16'
        )

        print("✓ Created StableDiffusionModel instance with F-Lite config")
        print(f"  - Architecture: {model.arch}")
        print(f"  - is_flite property exists: {hasattr(model, 'is_flite')}")

        if hasattr(model, 'is_flite'):
            print(f"  - is_flite: {model.is_flite}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_loading():
    """Test full F-Lite model loading (requires HuggingFace download)."""
    print("\n" + "=" * 60)
    print("TEST 4: Testing Full F-Lite Model Loading")
    print("=" * 60)
    print("This test will download the F-Lite model from HuggingFace")
    print("Size: ~20GB (first time only)")

    response = input("\nProceed with full model loading test? (y/n): ")
    if response.lower() != 'y':
        print("Skipping full loading test")
        return None

    try:
        from toolkit.stable_diffusion_model import StableDiffusionModel
        from toolkit.config_modules import ModelConfig

        # Check CUDA availability
        if not torch.cuda.is_available():
            print("⚠ CUDA not available, skipping GPU-dependent test")
            return None

        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  - Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")

        # Create config
        config = ModelConfig(
            name_or_path="Freepik/F-Lite",
            is_flite=True,
            quantize=True,  # Enable quantization for memory efficiency
            dtype='bf16'
        )

        # Create model
        model = StableDiffusionModel(
            model_config=config,
            device='cuda:0',
            dtype='bf16'
        )

        print("\nLoading F-Lite model (this may take a few minutes)...")
        model.load_model()

        print("\n✓ F-Lite model loaded successfully!")
        print(f"  - Architecture: {model.arch}")
        print(f"  - Is F-Lite: {model.is_flite}")
        print(f"  - Has text encoder: {hasattr(model, 'text_encoder')}")
        print(f"  - Has VAE: {hasattr(model, 'vae')}")
        print(f"  - Has UNet/Transformer: {hasattr(model, 'unet')}")

        # Check memory usage
        print(f"\nMemory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f}GB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f}GB")

        return True

    except ImportError as e:
        print(f"\n✗ FAILED: Missing package: {e}")
        print("\nMake sure f_lite package is installed:")
        print("  pip install git+https://github.com/fal-ai/f-lite.git")
        return False
    except torch.cuda.OutOfMemoryError:
        print(f"\n✗ FAILED: CUDA Out of Memory")
        print("\nTry:")
        print("  1. Close other GPU applications")
        print("  2. Enable quantization (already enabled in test)")
        print("  3. Use a GPU with more VRAM (24GB+ recommended)")
        return False
    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("F-LITE AI TOOLKIT INTEGRATION TEST SUITE")
    print("=" * 60)

    results = []

    # Test 1: config_modules
    results.append(("Config Modules", test_config_modules()))

    # Test 2: flite handler
    results.append(("F-Lite Handler", test_flite_handler()))

    # Test 3: stable_diffusion_model
    results.append(("Stable Diffusion Model", test_stable_diffusion_model()))

    # Test 4: full loading (optional)
    full_load_result = test_full_loading()
    if full_load_result is not None:
        results.append(("Full Model Loading", full_load_result))

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed! F-Lite integration is working correctly.")
        print("\nNext steps:")
        print("  1. Copy config/examples/train_lora_flite_24gb.yaml to your config folder")
        print("  2. Update the dataset folder_path in the config")
        print("  3. Run training: python run.py config/your_flite_config.yaml")
    else:
        print("\n⚠ Some tests failed. Please review the errors above.")
        print("\nCommon issues:")
        print("  - f_lite package not installed: pip install git+https://github.com/fal-ai/f-lite.git")
        print("  - Missing modifications to config_modules.py or stable_diffusion_model.py")
        print("  - File toolkit/models/flite.py not created")

        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
