import torch
import torch.nn as nn
from dpipe.layers.conv import PreActivationND
from peft import LoraConfig, TaskType, get_peft_model

def apply_image_lora(image_encoder, r=2, lora_alpha=2, lora_dropout=0.0):
    """
    Wrap an image encoder's ResNet backbone with ConvLoRA for efficient fine-tuning.

    Args:
        image_encoder: The image encoder module containing an i3_resnet attribute.
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout rate.

    Returns:
        The modified image encoder with ConvLoRA applied to the i3_resnet backbone.
    """
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        # Regex matches 'conv1', 'conv2', 'conv3', or the downsample Conv3d blocks
        target_modules=r".*\.conv[123]|.*\.downsample\.0", 
        # target_modules=['conv1', 'conv2', 'conv3', 'downsample.0'],
        lora_dropout=lora_dropout,
        bias="none",
    )

    # Pass your top-level model container
    wrapped = get_peft_model(image_encoder, lora_config)
    return wrapped

def conv_merge_and_unload(top_peft_model):
    """
    Manually merges LoRA weights into Conv3d base layers and completely 
    strips out the PEFT structural wrappers (PeftModel, LoraModel, lora.Conv3d).
    """
    if hasattr(top_peft_model, "base_model") and hasattr(top_peft_model.base_model, "model"):
        native_base_model = top_peft_model.base_model.model
    else:
        raise ValueError("The provided model does not appear to be a standard PEFT wrapper container.")

    for name, module in top_peft_model.named_modules():
        # Check if the module is a PEFT LoRA Conv3d layer
        if hasattr(module, "lora_A") and hasattr(module, "lora_B") and "default" in module.lora_A:
            
            weight_A = module.lora_A["default"].weight  # (r, in_channels, k_d, k_h, k_w)
            weight_B = module.lora_B["default"].weight  # (out_channels, r, 1, 1, 1)
            
            r = module.r["default"]
            alpha = module.lora_alpha["default"]
            scaling = alpha / r
            
            with torch.no_grad():
                base_shape = module.base_layer.weight.shape
                out_ch, in_ch, k_d, k_h, k_w = base_shape
                
                # Reshape matrix math based on which side holds the kernel
                if weight_A.shape[2:] == (1, 1, 1):
                    B_flat = weight_B.view(out_ch, r * k_d * k_h * k_w)
                    A_flat = weight_A.view(r, in_ch)
                    delta_weight = torch.matmul(B_flat, A_flat).view(out_ch, k_d, k_h, k_w, in_ch)
                    delta_weight = delta_weight.permute(0, 4, 1, 2, 3)
                else:
                    B_flat = weight_B.view(out_ch, r)
                    A_flat = weight_A.view(r, in_ch * k_d * k_h * k_w)
                    delta_weight = torch.matmul(B_flat, A_flat).view(base_shape)
                
                # Permanently add the adapter weights directly to the base layer
                module.base_layer.weight.data += delta_weight * scaling

    def recursive_strip(module):
        for name, child in module.named_children():
            # If a child is a lora wrapper container, swap it out for its raw base_layer
            if child.__class__.__name__ == "Conv3d" and hasattr(child, "base_layer"):
                setattr(module, name, child.base_layer)
            else:
                recursive_strip(child)

    # Clean the extracted core model structure completely
    recursive_strip(native_base_model)
    
    return native_base_model


def apply_text_lora(text_encoder, r=16, lora_alpha=32, lora_dropout=0.2):
    """
    Wrap a text encoder with PEFT LoRA targeting Q/K/V projection layers.

    Args:
        text_encoder: The Hugging Face transformer encoder to wrap.
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: LoRA dropout rate.

    Returns:
        The PEFT-wrapped text encoder.
    """
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["query", "key", "value"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    wrapped = get_peft_model(text_encoder, lora_config)
    wrapped.config.gradient_checkpointing = True
    return wrapped


def unfreeze_module(module: nn.Module):
    """Enable gradient computation for all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = True


def freeze_module(module: nn.Module):
    """Disable gradient computation for all parameters in a module."""
    for param in module.parameters():
        param.requires_grad = False