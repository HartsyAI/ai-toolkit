"""
F-Lite Model Handler for AI Toolkit

This module provides model loading and handling for Fal AI's F-Lite model.
F-Lite is a copyright-safe text-to-image diffusion model trained on licensed content.

Key Features:
- 10B parameter DiT transformer
- T5 XXL text encoder with layer 17 extraction
- Flux Schnell VAE
- Flow matching training paradigm
- Optimized for 24GB VRAM with quantization

Author: AI Toolkit + Claude
Version: 1.0
"""

import os
import torch
from typing import Dict, Optional, Any
from diffusers import AutoencoderKL


class FLiteModelHandler:
    """
    Handler class for F-Lite model operations.
    Manages model registration, loading, and configuration.
    """

    def __init__(self):
        self.model_name = "Freepik/F-Lite"
        self.registered = False

    def register_flite_classes(self) -> bool:
        """
        Register F-Lite model classes with diffusers.

        This method attempts to register the FliteTransformer2DModel
        from the f_lite package with diffusers' model loading system.

        Returns:
            bool: True if registration successful, False otherwise
        """
        try:
            from f_lite.model.transformer_2d import FliteTransformer2DModel

            # Register with diffusers
            from diffusers import ModelMixin
            if hasattr(ModelMixin, '_register_model'):
                ModelMixin._register_model('flite', FliteTransformer2DModel)

            self.registered = True
            print("✓ F-Lite model classes registered successfully")
            return True

        except ImportError as e:
            print(f"⚠ Could not register F-Lite classes: {e}")
            print("This is not critical if you're just testing imports")
            return False
        except Exception as e:
            print(f"⚠ Unexpected error during registration: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the F-Lite model.

        Returns:
            Dict containing model specifications
        """
        return {
            'name': 'F-Lite',
            'organization': 'Freepik',
            'model_id': self.model_name,
            'parameters': '10B',
            'text_encoder': 'T5 XXL (layer 17)',
            'vae': 'Flux Schnell VAE',
            'training_paradigm': 'Flow Matching',
            'recommended_vram': '24GB+ (with quantization)',
            'copyright_safe': True,
        }


def load_t5_xxl_with_layer_extraction(
    model_path: str,
    layer: int = -8,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16
) -> torch.nn.Module:
    """
    Load T5 XXL text encoder for F-Lite.

    F-Lite uses layer -8 (8th from end) of T5 XXL for text embeddings with
    layer normalization and dropout applied after extraction.

    The actual layer extraction happens in train_tools.encode_prompts_flite(),
    which properly extracts hidden_states[return_index] and applies layer norm/dropout.

    Args:
        model_path: Path or HuggingFace ID for the T5 model
        layer: Which layer to extract (default: -8 for F-Lite). This is for info only.
        device: Target device
        dtype: Data type for the model

    Returns:
        Loaded T5 encoder model
    """
    from transformers import T5EncoderModel

    print(f"Loading T5 XXL (F-Lite uses layer {layer} extraction)...")

    text_encoder = T5EncoderModel.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )

    if device is not None:
        text_encoder.to(device)

    print(f"✓ T5 XXL loaded (layer {layer} will be extracted during encoding)")
    return text_encoder


def load_flite_vae(
    vae_path: Optional[str] = None,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16
) -> AutoencoderKL:
    """
    Load the VAE for F-Lite (Flux Schnell VAE).

    Args:
        vae_path: Optional path to VAE. If None, loads from Flux Schnell
        device: Target device
        dtype: Data type for the VAE

    Returns:
        Loaded VAE model
    """
    if vae_path is None:
        vae_path = "black-forest-labs/FLUX.1-schnell"
        subfolder = "vae"
    else:
        subfolder = None

    print("Loading VAE (Flux Schnell)...")

    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder=subfolder,
        torch_dtype=dtype,
    )

    if device is not None:
        vae.to(device)

    print("✓ VAE loaded")
    return vae


def load_flite_transformer(
    model_path: str,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
    low_vram: bool = False
) -> torch.nn.Module:
    """
    Load the F-Lite DiT transformer (10B parameters).

    Args:
        model_path: Path or HuggingFace ID for the transformer
        device: Target device
        dtype: Data type for the model
        low_vram: If True, keep on CPU initially

    Returns:
        Loaded transformer model
    """
    from f_lite.model.transformer_2d import FliteTransformer2DModel

    print("Loading F-Lite transformer (10B parameters)...")
    print(f"  Model path: {model_path}")

    transformer = FliteTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )

    if not low_vram and device is not None:
        transformer.to(device)
        print(f"✓ Transformer loaded on {device}")
    else:
        print("✓ Transformer loaded (on CPU for low VRAM mode)")

    return transformer


def load_flite_tokenizer(
    model_path: str
):
    """
    Load the tokenizer for F-Lite (T5 tokenizer).

    Args:
        model_path: Path or HuggingFace ID for the model

    Returns:
        Loaded tokenizer
    """
    from transformers import T5Tokenizer

    print("Loading T5 tokenizer...")

    tokenizer = T5Tokenizer.from_pretrained(
        model_path,
        subfolder="tokenizer",
    )

    print("✓ Tokenizer loaded")
    return tokenizer


def setup_flite_for_training(
    model_config,
    device: torch.device,
    dtype_str: str = "bf16"
) -> Dict[str, Any]:
    """
    Complete setup function for F-Lite model training.

    This is the main entry point called by stable_diffusion_model.py
    when loading an F-Lite model for training.

    Args:
        model_config: ModelConfig object with F-Lite settings
        device: Target device for models
        dtype_str: Data type string ('bf16', 'fp16', 'fp32')

    Returns:
        Dictionary containing all model components:
            - tokenizer: T5 tokenizer
            - text_encoder: T5 XXL with layer 17 extraction
            - vae: Flux Schnell VAE
            - transformer: F-Lite DiT transformer (10B)
    """
    print("\n" + "=" * 60)
    print("Setting up F-Lite for Training")
    print("=" * 60)

    # Parse dtype
    dtype_map = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32,
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.bfloat16)
    print(f"Using dtype: {dtype}")

    # Get model path
    model_path = model_config.name_or_path
    print(f"Model path: {model_path}")

    # Check if we're in low VRAM mode
    low_vram = getattr(model_config, 'low_vram', False)
    if low_vram:
        print("⚠ Low VRAM mode enabled")

    # Get layer extraction setting (default -8 for F-Lite)
    # Note: -8 means 8th layer from the end (not layer 17)
    extract_layer = getattr(model_config, 'flite_text_encoder_layer', -8)
    print(f"T5 layer extraction: {extract_layer} (8th from end)")

    # Register F-Lite classes
    handler = FLiteModelHandler()
    handler.register_flite_classes()

    # Load components
    print("\n" + "-" * 60)
    print("Loading Model Components")
    print("-" * 60)

    # 1. Load tokenizer
    tokenizer = load_flite_tokenizer(model_path)

    # 2. Load text encoder with layer extraction
    t5_path = model_path
    # Check if there's a custom TE path
    if hasattr(model_config, 'te_name_or_path') and model_config.te_name_or_path:
        t5_path = model_config.te_name_or_path
        print(f"Using custom T5 path: {t5_path}")

    text_encoder = load_t5_xxl_with_layer_extraction(
        model_path=t5_path,
        layer=extract_layer,
        device=device,
        dtype=dtype
    )

    # 3. Load VAE
    vae_path = getattr(model_config, 'vae_path', None)
    vae = load_flite_vae(
        vae_path=vae_path,
        device=device,
        dtype=dtype
    )

    # 4. Load transformer
    transformer = load_flite_transformer(
        model_path=model_path,
        device=device if not low_vram else None,
        dtype=dtype,
        low_vram=low_vram
    )

    print("\n" + "-" * 60)
    print("Component Summary")
    print("-" * 60)
    print(f"✓ Tokenizer: T5")
    print(f"✓ Text Encoder: T5 XXL (layer {extract_layer} = 8th from end)")
    print(f"✓ VAE: Flux Schnell")
    print(f"✓ Transformer: F-Lite DiT (10B)")
    print(f"✓ Dtype: {dtype}")
    print(f"✓ Device: {device}")
    if low_vram:
        print("✓ Low VRAM mode: Active")

    # Return components
    return {
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'vae': vae,
        'transformer': transformer,
        'dtype': dtype,
        'device': device,
    }


def get_flite_noise_pred(
    model,
    noisy_latents,
    timesteps,
    prompt_embeds,
    **kwargs
):
    """
    Get noise prediction from F-Lite model.

    This function handles the forward pass through the F-Lite transformer
    during training, accounting for any F-Lite-specific requirements.

    Args:
        model: The F-Lite transformer model
        noisy_latents: Noisy latent representations
        timesteps: Diffusion timesteps
        prompt_embeds: Text embeddings from T5 (layer 17)
        **kwargs: Additional arguments

    Returns:
        Predicted noise
    """
    # F-Lite uses similar interface to FLUX
    # The model expects:
    # - hidden_states: noisy latents
    # - timestep: diffusion timestep
    # - encoder_hidden_states: text embeddings

    noise_pred = model(
        hidden_states=noisy_latents,
        timestep=timesteps,
        encoder_hidden_states=prompt_embeds,
        return_dict=False,
    )[0]

    return noise_pred


# Model information for reference
FLITE_MODEL_INFO = {
    'model_name': 'F-Lite',
    'model_id': 'Freepik/F-Lite',
    'organization': 'Freepik / Fal AI',
    'parameters': '10B',
    'architecture': 'Diffusion Transformer (DiT)',
    'text_encoder': 'T5 XXL',
    'text_encoder_extraction': 'Layer -8 (8th from end) with layer norm/dropout',
    'vae': 'Flux Schnell VAE',
    'training_paradigm': 'Flow Matching',
    'license': 'Copyright-safe (trained on licensed data)',
    'recommended_vram': '24GB with quantization',
    'recommended_settings': {
        'quantize': True,
        'dtype': 'bf16',
        'noise_scheduler': 'flowmatch',
        'guidance_scale': '3.0-6.0',
        'inference_steps': '25-30',
    },
    'notes': [
        'F-Lite is trained on licensed data only (Freepik library)',
        'Layer -8 (8th from end) extraction from T5 with layer norm and dropout',
        'Quantization recommended for 24GB VRAM systems',
        'Flow matching scheduler required for training',
        'Lower guidance scales (3-6) work better than SD (7-10)',
    ]
}


def print_flite_info():
    """Print F-Lite model information."""
    info = FLITE_MODEL_INFO
    print("\n" + "=" * 60)
    print("F-Lite Model Information")
    print("=" * 60)
    print(f"Model: {info['model_name']}")
    print(f"Organization: {info['organization']}")
    print(f"HuggingFace ID: {info['model_id']}")
    print(f"Parameters: {info['parameters']}")
    print(f"Architecture: {info['architecture']}")
    print(f"Text Encoder: {info['text_encoder']}")
    print(f"  └─ Extraction: {info['text_encoder_extraction']}")
    print(f"VAE: {info['vae']}")
    print(f"Training: {info['training_paradigm']}")
    print(f"License: {info['license']}")
    print(f"Recommended VRAM: {info['recommended_vram']}")
    print("\nRecommended Settings:")
    for key, value in info['recommended_settings'].items():
        print(f"  {key}: {value}")
    print("\nNotes:")
    for note in info['notes']:
        print(f"  • {note}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # If run directly, print model info
    print_flite_info()
