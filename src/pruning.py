import torch.nn.utils.prune as prune
import torch
from transformer import TimeSeriesTransformer

# Magnitude-based pruning
def magnitude_pruning(model, amount=0.2):
    parameters_to_prune = [(name, module) for name, module in model.named_parameters()]
    for name, module in parameters_to_prune:
        prune.l1_unstructured(module, name='weight', amount=amount)


# L1 norm-based pruning
def l1_norm_pruning(model, amount=0.2):
    parameters_to_prune = [(name, module) for name, module in model.named_parameters()]
    for name, module in parameters_to_prune:
        prune.l1_unstructured(module, name='weight', amount=amount)


# L2 norm-based pruning
def l2_norm_pruning(model, amount=0.2):
    parameters_to_prune = [(name, module) for name, module in model.named_parameters()]
    for name, module in parameters_to_prune:
        prune.l2_unstructured(module, name='weight', amount=amount)


# Global pruning
def global_pruning(model, amount=0.2):
    parameters_to_prune = [(name, module) for name, module in model.named_parameters()]
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )


# Structured pruning
def structured_pruning(model, amount=0.2):
    parameters_to_prune = [(name, module) for name, module in model.named_parameters()]
    for name, module in parameters_to_prune:
        prune.ln_structured(module, name='weight', amount=amount, n=2, dim=0)


# Iterative pruning
def iterative_pruning(model, num_iterations=3, amount=0.2):
    for i in range(num_iterations):
        parameters_to_prune = [(name, module) for name, module in model.named_parameters()]
        for name, module in parameters_to_prune:
            prune.l1_unstructured(module, name='weight', amount=amount)