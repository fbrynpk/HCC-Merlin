import re

import torch
import torch.nn as nn

_LAYER_RE = re.compile(r"^text_encoder\.encoder\.layer\.(\d+)\.")


def text_group_key(name: str) -> str:
    """
    Map a text-encoder parameter name to its LLDR group key.

    Examples:
        text_encoder.embeddings.word_embeddings.weight -> text_encoder.embeddings
        text_encoder.encoder.layer.3.attention.self.query.weight -> text_encoder.encoder.layer.3
        text_encoder.pooler.dense.weight -> text_encoder.pooler
        linear_layer.weight -> linear_layer
    """
    if name.startswith("text_encoder.embeddings."):
        return "text_encoder.embeddings"
    m = _LAYER_RE.match(name)
    if m:
        return f"text_encoder.encoder.layer.{m.group(1)}"
    if name.startswith("text_encoder.pooler."):
        return "text_encoder.pooler"
    if name.startswith("linear_layer."):
        return "linear_layer"
    parts = name.split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else name


def image_group_key(name: str) -> str:
    """
    Map an image-encoder parameter name to its LLDR group key.

    Example:
        i3_resnet.layer4.2.conv1.weight -> layer4
    """
    parts = name.split(".")
    return parts[1] if len(parts) > 1 else name


def _should_exclude(name: str, param: torch.Tensor) -> bool:
    """Return True for params that should have zero weight decay."""
    return param.ndim < 2 or any(
        tag in name
        for tag in ("bn", "ln", "LayerNorm", "bias", "logit_scale", "temperature")
    )


def build_lldr_groups(
    layer_names,
    named_params,
    base_lr: float,
    lr_mult: float,
    weight_decay: float,
    used_param_ids: set,
    prefix: str,
    group_key_fn,
):
    """
    Build per-parameter optimizer groups with layer-wise learning rate decay (LLDR).

    Parameters deeper in the network (earlier in layer_names after reversing) receive
    progressively lower learning rates, decaying by lr_mult each time the group key changes.

    Args:
        layer_names: Ordered list of parameter names (outer → inner, i.e. reversed).
        named_params: Dict mapping name → Parameter for the relevant sub-module.
        base_lr: Starting learning rate (applied to the first group encountered).
        lr_mult: Multiplicative decay applied each time the layer group changes.
        weight_decay: Weight decay for eligible parameters.
        used_param_ids: Set of already-assigned parameter ids (mutated in place).
        prefix: String prefix added to each group's 'name' field for debugging.
        group_key_fn: Callable(name) -> str grouping parameters into LLDR buckets.

    Returns:
        List of optimizer param-group dicts.
    """
    if not layer_names:
        return []

    param_groups = []
    lr = base_lr
    prev_group = group_key_fn(layer_names[0])

    for name in layer_names:
        if name not in named_params:
            continue
        p = named_params[name]
        if not p.requires_grad or id(p) in used_param_ids:
            continue

        cur_group = group_key_fn(name)
        if cur_group != prev_group:
            lr *= lr_mult
            prev_group = cur_group

        used_param_ids.add(id(p))
        wd = 0.0 if _should_exclude(name, p) else weight_decay

        param_groups.append(
            {
                "name": f"{prefix}:{name}",
                "params": [p],
                "lr": lr,
                "weight_decay": wd,
            }
        )

    return param_groups


def build_param_groups_image_text_lldr(
    model,
    image_layer_names,
    text_layer_names,
    base_lr_img: float,
    base_lr_txt: float = None,
    lr_mult_img: float = 0.7,
    lr_mult_txt: float = 0.7,
    weight_decay: float = 0.01,
):
    """
    Combine image and text encoder LLDR param groups into a single list.

    Args:
        model: Merlin model wrapper.
        image_layer_names: Reversed list of image-encoder parameter names.
        text_layer_names: Reversed list of text-encoder parameter names.
        base_lr_img: Base LR for the image encoder.
        base_lr_txt: Base LR for the text encoder (defaults to base_lr_img).
        lr_mult_img: Per-group LR decay multiplier for image encoder.
        lr_mult_txt: Per-group LR decay multiplier for text encoder.
        weight_decay: Weight decay for eligible parameters.

    Returns:
        Combined list of param-group dicts ready for an optimizer.
    """
    if base_lr_txt is None:
        base_lr_txt = base_lr_img

    used_param_ids = set()

    img_groups = build_lldr_groups(
        layer_names=image_layer_names,
        named_params=dict(model.model.encode_image.named_parameters()),
        base_lr=base_lr_img,
        lr_mult=lr_mult_img,
        weight_decay=weight_decay,
        used_param_ids=used_param_ids,
        prefix="img",
        group_key_fn=image_group_key,
    )

    txt_groups = build_lldr_groups(
        layer_names=text_layer_names,
        named_params=dict(model.model.encode_text.named_parameters()),
        base_lr=base_lr_txt,
        lr_mult=lr_mult_txt,
        weight_decay=weight_decay,
        used_param_ids=used_param_ids,
        prefix="txt",
        group_key_fn=text_group_key,
    )

    prompt_groups = []
    if getattr(model.model, "use_coop", False) and hasattr(
        model.model, "prompt_learner"
    ):
        prompt_params = []
        for name, param in model.model.prompt_learner.named_parameters():
            if not param.requires_grad or id(param) in used_param_ids:
                continue
            used_param_ids.add(id(param))
            prompt_params.append(param)

        if prompt_params:
            prompt_groups.append(
                {
                    "name": "txt:prompt_learner",
                    "params": prompt_params,
                    "lr": base_lr_txt,
                    "weight_decay": 0.0,
                }
            )

    return img_groups + txt_groups + prompt_groups


def build_optimizer_and_scheduler(
    model, param_groups, args, total_steps, warmup_steps=0
):
    """
    Construct AdamW optimizer with a warmup → cosine decay schedule.

    Args:
        model: Merlin model (used to append temperature param group).
        param_groups: List of param-group dicts from build_param_groups_image_text_lldr.
        args: Parsed argument namespace (learning_rate used for temperature).
        total_steps: Total number of optimizer steps across all epochs.
        warmup_steps: Number of linear warmup steps before cosine decay begins.

    Returns:
        (optimizer, scheduler, scaler)
    """
    # Append temperature if learnable
    if (
        isinstance(model.model.temperature, nn.Parameter)
        and model.model.temperature.requires_grad
    ):
        param_groups.append(
            {
                "name": "scalar:temperature",
                "params": [model.model.temperature],
                "lr": args.learning_rate,
                "weight_decay": 0.0,
            }
        )
    # if args.use_coop:
    optimizer = torch.optim.SGD(param_groups, weight_decay=1e-4)
    print(optimizer.param_groups[0]["params"][0].shape)
    print(optimizer.param_groups[0]["lr"])

    # else:
    # optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: (
            float(step) / float(max(1, warmup_steps)) if step < warmup_steps else 1.0
        ),
    )
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=0
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, decay_scheduler],
        milestones=[warmup_steps],
    )

    from torch.cuda.amp import GradScaler

    scaler = GradScaler()

    return optimizer, scheduler, scaler
