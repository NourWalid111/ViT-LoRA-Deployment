import copy

import torch
import torch.nn as nn

from transformers import ViTForImageClassification


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "google/vit-base-patch16-224"

CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# Manual LoRA Linear Layer
# ============================================================

class LoRALinear(nn.Module):
    """
    Frozen pretrained Linear layer + trainable low-rank update.

    W' = W + scaling * B @ A

    A: [rank, in_features]
    B: [out_features, rank]
    """

    def __init__(
        self,
        original_layer,
        rank=16,
        alpha=16,
        dropout=0.05,
    ):
        super().__init__()

        if not isinstance(original_layer, nn.Linear):
            raise TypeError(
                "LoRALinear requires an nn.Linear layer."
            )

        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features

        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Keep a frozen copy of the pretrained layer.
        self.base = copy.deepcopy(original_layer)

        for parameter in self.base.parameters():
            parameter.requires_grad = False

        # LoRA A
        self.lora_A = nn.Parameter(
            torch.empty(
                rank,
                self.in_features,
            )
        )

        # LoRA B
        self.lora_B = nn.Parameter(
            torch.zeros(
                self.out_features,
                rank,
            )
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize A randomly.
        nn.init.kaiming_uniform_(
            self.lora_A,
            a=5 ** 0.5,
        )

        # B remains zero.
        # Therefore the initial LoRA update is zero.

    def forward(self, x):

        # Frozen pretrained projection.
        base_output = self.base(x)

        # Low-rank LoRA update.
        lora_output = (
            self.dropout(x)
            @ self.lora_A.T
            @ self.lora_B.T
        )

        return (
            base_output
            + self.scaling * lora_output
        )


# ============================================================
# Load ViT
# ============================================================

def load_vit():

    model = ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        id2label={
            index: name
            for index, name in enumerate(CLASS_NAMES)
        },
        label2id={
            name: index
            for index, name in enumerate(CLASS_NAMES)
        },
    )

    return model


# ============================================================
# Get ViT Transformer blocks
# ============================================================

def get_vit_blocks(model):

    """
    Transformers versions use different internal names.

    Older versions:
        model.vit.encoder.layer

    Newer versions:
        model.vit.layers
    """

    if hasattr(model.vit, "layers"):

        return model.vit.layers

    if hasattr(model.vit, "encoder"):

        return model.vit.encoder.layer

    raise AttributeError(
        "Could not find ViT transformer blocks."
    )


# ============================================================
# Get Query / Value projections
# ============================================================

def get_attention_projections(block):

    """
    Handles both older and newer Transformers ViT layouts.
    """

    attention = block.attention

    # Older structure:
    #
    # block.attention.attention.query
    # block.attention.attention.value
    #
    if hasattr(attention, "attention"):

        attention_module = attention.attention

    else:

        # Newer structure.
        attention_module = attention

    # Query
    if hasattr(attention_module, "q_proj"):

        query = attention_module.q_proj
        query_name = "q_proj"

    elif hasattr(attention_module, "query"):

        query = attention_module.query
        query_name = "query"

    else:

        raise AttributeError(
            "Could not find Query projection."
        )

    # Value
    if hasattr(attention_module, "v_proj"):

        value = attention_module.v_proj
        value_name = "v_proj"

    elif hasattr(attention_module, "value"):

        value = attention_module.value
        value_name = "value"

    else:

        raise AttributeError(
            "Could not find Value projection."
        )

    return (
        attention_module,
        query,
        query_name,
        value,
        value_name,
    )


# ============================================================
# Apply LoRA
# ============================================================

def apply_lora(
    model,
    rank=16,
    target_layers=None,
):

    blocks = get_vit_blocks(model)

    num_layers = len(blocks)

    # Last 3 Transformer blocks.
    if target_layers is None:

        target_layers = list(
            range(
                num_layers - 3,
                num_layers,
            )
        )

    print(
        "Applying LoRA to Transformer blocks:",
        target_layers,
    )

    for layer_index in target_layers:

        block = blocks[layer_index]

        (
            attention_module,
            query,
            query_name,
            value,
            value_name,
        ) = get_attention_projections(
            block
        )

        print(
            f"Block {layer_index}: "
            f"Query={query_name}, "
            f"Value={value_name}"
        )

        # Replace Query projection.
        setattr(
            attention_module,
            query_name,
            LoRALinear(
                query,
                rank=rank,
                alpha=rank,
            ),
        )

        # Replace Value projection.
        setattr(
            attention_module,
            value_name,
            LoRALinear(
                value,
                rank=rank,
                alpha=rank,
            ),
        )

    # Freeze EVERYTHING.
    for parameter in model.parameters():

        parameter.requires_grad = False

    # Enable ONLY LoRA parameters.
    for module in model.modules():

        if isinstance(module, LoRALinear):

            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True

    return model, target_layers


# ============================================================
# Parameter utilities
# ============================================================

def count_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable_parameters(model):

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def print_parameter_summary(model):

    total = count_parameters(model)

    trainable = count_trainable_parameters(model)

    percentage = (
        100.0
        * trainable
        / total
    )

    print()
    print("=" * 60)
    print("MODEL PARAMETER SUMMARY")
    print("=" * 60)

    print(
        f"Total parameters:      {total:,}"
    )

    print(
        f"Trainable parameters:  {trainable:,}"
    )

    print(
        f"Trainable percentage:  {percentage:.4f}%"
    )

    print("=" * 60)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("Loading pretrained ViT...")

    model = load_vit()

    blocks = get_vit_blocks(model)

    print(
        f"ViT Transformer blocks: {len(blocks)}"
    )

    print(
        f"Block type: {type(blocks[0]).__name__}"
    )

    # Assignment requirement:
    # Query + Value in the last 3 blocks.
    model, target_layers = apply_lora(
        model,
        rank=16,
    )

    print_parameter_summary(model)

    print()
    print("Trainable LoRA parameters:")
    print("-" * 100)

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            print(
                f"{name:<90}"
                f"{parameter.numel():,}"
            )

    print()
    print(
        "LoRA model created successfully."
    )