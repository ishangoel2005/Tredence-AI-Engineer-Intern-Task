"""
Self-Pruning Neural Network on CIFAR-10
Key Components:
  - PrunableLinear: Custom linear layer with learnable gate_scores
  - SelfPruningNet: The full network built from PrunableLinear layers
  - Sparsity Loss: L1 penalty on sigmoid(gate_scores) to drive gates to 0
  - Training + Evaluation loop for 3 different lambda values
  - Final gate distribution plot via matplotlib
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# CONFIGURATION

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE    = 128
EPOCHS        = 25
LEARNING_RATE = 1e-3
SPARSITY_THRESHOLD = 1e-2            # gates below this are considered "pruned"
LAMBDA_VALUES = [0.0001, 0.001, 0.01]  # low, medium, high sparsity pressure

print(f"Using device: {DEVICE}")


# PART 1: PrunableLinear Layer

class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that augments each weight
    with a learnable scalar gate.

    Forward pass logic:
        gates        = sigmoid(gate_scores)          # values in (0, 1)
        pruned_weights = weight * gates              # element-wise
        output       = input @ pruned_weights.T + bias

    When a gate collapses to ~0, the corresponding weight is effectively
    removed from the network, no gradient flows through it and it
    contributes nothing to the output.

    Both `weight` and `gate_scores` are registered as nn.Parameter so
    PyTorch's autograd handles gradient flow automatically through both.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias (same as nn.Linear)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )
        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        # Gate scores: one scalar per weight element.
        # Initialized near 0 so initial gates ≈ sigmoid(0) = 0.5
        # (neutral start, neither pruned nor fully active).
        self.gate_scores = nn.Parameter(
            torch.zeros(out_features, in_features)
        )

        # Kaiming initialization for weights (good default for ReLU nets)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Turn gate_scores into values strictly in (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Element-wise multiply weights by their gates
        # If a gate → 0, that weight contributes nothing to output
        pruned_weights = self.weight * gates

        # Step 3: Standard linear transform with pruned weights
        # Using F.linear equivalent: output = x @ W^T + b
        return nn.functional.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return the actual gate values (after sigmoid) as a flat tensor."""
        return torch.sigmoid(self.gate_scores).detach().flatten()

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}"


# PART 1 (cont.): The Full Network

class SelfPruningNet(nn.Module):
    """
    A 3-layer feed-forward network for CIFAR-10 classification.
    All linear layers are PrunableLinear, they carry gate parameters
    that get pushed toward 0 by the sparsity loss during training.

    Architecture:
        Input  : 32 x 32 x 3 = 3072 features (flattened)
        Layer 1: PrunableLinear(3072 → 512)  + ReLU
        Layer 2: PrunableLinear(512  → 256)  + ReLU
        Layer 3: PrunableLinear(256  →  10)  (logits, no activation)
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.ReLU(),
            PrunableLinear(256, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.net(x)

    def prunable_layers(self):
        """Iterate over all PrunableLinear sub-modules."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

# PART 2: Sparsity Loss (L1 on gate values)

def compute_sparsity_loss(model: SelfPruningNet) -> torch.Tensor:
    """
    Compute the L1 sparsity penalty across all PrunableLinear layers.

    For each layer:
        gates = sigmoid(gate_scores)    → values in (0, 1)
        penalty = sum(gates)            → L1 norm (all positive, so |x| = x)

    Total sparsity loss = sum of penalties across all layers.

    WHY L1 ENCOURAGES SPARSITY:
    - The gradient of |x| w.r.t. x is sign(x), a constant ±1.
    - For positive gates, this gradient is always -λ (pushing toward 0).
    - Unlike L2 (gradient proportional to value), L1 applies equal pressure
      on large and small gates. Small gates get the same push toward 0
      as large ones, so they actually reach 0, not just shrink.
    - Gates important for accuracy get a counteracting gradient from the
      classification loss. Gates that don't help accuracy get no pushback
      and collapse to 0 → effective pruning.
    """
    sparsity_loss = torch.tensor(0.0, device=DEVICE)
    for layer in model.prunable_layers():
        gates = torch.sigmoid(layer.gate_scores)
        sparsity_loss = sparsity_loss + gates.sum()
    return sparsity_loss


# PART 3: Sparsity Metric (post-training evaluation)

def compute_sparsity_level(model: SelfPruningNet,
                           threshold: float = SPARSITY_THRESHOLD) -> float:
    """
    Compute what fraction of weights have been effectively pruned.

    A weight is considered 'pruned' if its gate value < threshold.
    Returns a percentage (0–100).
    """
    total_weights = 0
    pruned_weights = 0
    for layer in model.prunable_layers():
        gates = layer.get_gates()           # flat tensor of gate values
        total_weights  += gates.numel()
        pruned_weights += (gates < threshold).sum().item()
    return 100.0 * pruned_weights / total_weights if total_weights > 0 else 0.0


def get_all_gate_values(model: SelfPruningNet) -> np.ndarray:
    """Return all gate values from all PrunableLinear layers as a numpy array."""
    all_gates = []
    for layer in model.prunable_layers():
        all_gates.append(layer.get_gates().cpu().numpy())
    return np.concatenate(all_gates) if all_gates else np.array([])


# DATA LOADING

def get_cifar10_loaders(batch_size: int = BATCH_SIZE):
    """
    Load CIFAR-10 with standard normalization.
    Training set: 50,000 images  |  Test set: 10,000 images
    """
    # Normalize using CIFAR-10 channel means and stds
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std =(0.2023, 0.1994, 0.2010),
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=transform
    )
    test_dataset  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True
    )
    test_loader  = DataLoader(
        test_dataset,  batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    return train_loader, test_loader

# TRAINING LOOP

def train_one_epoch(model:     SelfPruningNet,
                    loader:    DataLoader,
                    optimizer: optim.Optimizer,
                    criterion: nn.Module,
                    lam:       float) -> tuple[float, float, float]:
    """
    Run a single training epoch.

    Returns:
        avg_total_loss     : mean of (classification + sparsity) loss
        avg_cls_loss       : mean of classification loss alone
        avg_sparsity_loss  : mean of sparsity loss alone
    """
    model.train()
    total_loss_sum    = 0.0
    cls_loss_sum      = 0.0
    sparsity_loss_sum = 0.0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)

        # ── Classification loss (Cross-Entropy) ──────────────────────────────
        cls_loss = criterion(logits, labels)

        # ── Sparsity loss (L1 on sigmoid gates) ──────────────────────────────
        sparsity_loss = compute_sparsity_loss(model)

        # ── Total loss ────────────────────────────────────────────────────────
        total_loss = cls_loss + lam * sparsity_loss

        # Backward pass: gradients flow through cls_loss (into weights AND
        # gate_scores) AND through sparsity_loss (into gate_scores only).
        total_loss.backward()
        optimizer.step()

        total_loss_sum    += total_loss.item()
        cls_loss_sum      += cls_loss.item()
        sparsity_loss_sum += sparsity_loss.item()

    n = len(loader)
    return total_loss_sum / n, cls_loss_sum / n, sparsity_loss_sum / n


# EVALUATION

@torch.no_grad()
def evaluate(model: SelfPruningNet, loader: DataLoader) -> float:
    """
    Evaluate the model on the given data loader.
    Returns accuracy as a percentage.
    """
    model.eval()
    correct = 0
    total   = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        logits  = model(images)
        preds   = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return 100.0 * correct / total


# FULL TRAINING RUN FOR ONE LAMBDA

def checkpoint_path(lam: float) -> str:
    """Return the checkpoint filename for a given lambda."""
    return f"checkpoint_lambda_{lam}.pt"


def run_experiment(lam:          float,
                   train_loader: DataLoader,
                   test_loader:  DataLoader) -> dict:
    """
    Train a fresh SelfPruningNet with sparsity weight `lam`.
    If a checkpoint already exists for this lambda, skip training
    and load results directly from disk.
    Returns a dict with test accuracy, sparsity level, and gate values.
    """
    ckpt_file    = checkpoint_path(lam)
    results_file = f"results_lambda_{lam}.json"

    # ── If checkpoint exists, skip retraining entirely ────────────────────
    if os.path.exists(ckpt_file) and os.path.exists(results_file):
        print(f"\n{'='*60}")
        print(f"  Checkpoint found for λ = {lam} — skipping retraining.")
        print(f"{'='*60}")

        # Load model weights
        model = SelfPruningNet().to(DEVICE)
        model.load_state_dict(torch.load(ckpt_file, map_location=DEVICE))
        model.eval()

        # Load saved scalar results
        with open(results_file, "r", encoding="utf-8") as f:
            saved = json.load(f)

        # Re-extract gate values from the loaded model (numpy arrays
        # can't be stored in JSON, so we recompute them)
        gate_values = get_all_gate_values(model)

        print(f"  Loaded  ->  Accuracy: {saved['accuracy']:.2f}%  |  "
              f"Sparsity: {saved['sparsity']:.1f}%")

        return {
            "lambda":      lam,
            "accuracy":    saved["accuracy"],
            "sparsity":    saved["sparsity"],
            "gate_values": gate_values,
        }

    # ── No checkpoint — train from scratch ───────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Training with λ = {lam}")
    print(f"{'='*60}")

    model     = SelfPruningNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        total_l, cls_l, sparse_l = train_one_epoch(
            model, train_loader, optimizer, criterion, lam
        )
        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader)
            sparsity = compute_sparsity_level(model)
            print(
                f"  Epoch {epoch:3d}/{EPOCHS} | "
                f"Total Loss: {total_l:.4f} | "
                f"CE Loss: {cls_l:.4f} | "
                f"Sparsity Loss: {sparse_l:.2f} | "
                f"Test Acc: {acc:.2f}% | "
                f"Sparsity: {sparsity:.1f}%"
            )

    final_acc      = evaluate(model, test_loader)
    final_sparsity = compute_sparsity_level(model)
    gate_values    = get_all_gate_values(model)

    print(f"\n  ── Final Results (λ={lam}) ──────────────────────────")
    print(f"  Test Accuracy : {final_acc:.2f}%")
    print(f"  Sparsity Level: {final_sparsity:.1f}%")

    # ── Save checkpoint + results immediately after this lambda finishes ──
    torch.save(model.state_dict(), ckpt_file)
    print(f"  Checkpoint saved -> {ckpt_file}")

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({"lambda": lam, "accuracy": final_acc, "sparsity": final_sparsity}, f)
    print(f"  Results saved   -> {results_file}")

    return {
        "lambda":       lam,
        "accuracy":     final_acc,
        "sparsity":     final_sparsity,
        "gate_values":  gate_values,
    }

# PLOTTING

def plot_gate_distribution(results: list[dict]) -> None:
    """
    Plot the distribution of gate values for each lambda in a single figure.
    A successful pruning shows a large spike at 0 (pruned weights)
    and a cluster near 1 (active weights), a bimodal distribution.
    """
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4),
                             sharey=False)
    fig.suptitle("Gate Value Distributions After Training", fontsize=14,
                 fontweight="bold", y=1.02)

    colors = ["#e74c3c", "#f39c12", "#2ecc71"]   # red, orange, green

    for ax, result, color in zip(axes, results, colors):
        gates    = result["gate_values"]
        lam      = result["lambda"]
        acc      = result["accuracy"]
        sparsity = result["sparsity"]

        ax.hist(gates, bins=80, color=color, alpha=0.85, edgecolor="white",
                linewidth=0.3)
        ax.set_title(
            f"λ = {lam}\nAcc: {acc:.1f}%  |  Sparsity: {sparsity:.1f}%",
            fontsize=11
        )
        ax.set_xlabel("Gate Value (sigmoid output)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        ax.axvline(x=0.01, color="black", linestyle="--", linewidth=1,
                   label="Prune threshold (0.01)")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig("gate_distributions.png", dpi=150, bbox_inches="tight")
    print("\nGate distribution plot saved → gate_distributions.png")
    plt.show()


def plot_tradeoff_summary(results: list[dict]) -> None:
    """
    Bar chart comparing accuracy and sparsity across lambda values.
    """
    lambdas   = [str(r["lambda"]) for r in results]
    accuracies = [r["accuracy"]  for r in results]
    sparsities = [r["sparsity"]  for r in results]

    x = np.arange(len(lambdas))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, accuracies, width, label="Test Accuracy (%)",
                    color="#3498db", alpha=0.85)
    bars2 = ax2.bar(x + width/2, sparsities, width, label="Sparsity Level (%)",
                    color="#e74c3c", alpha=0.85)

    ax1.set_xlabel("Lambda (λ)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", color="#3498db", fontsize=11)
    ax2.set_ylabel("Sparsity Level (%)", color="#e74c3c", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(lambdas, fontsize=11)
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    # Add value labels on bars
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 fontsize=9, color="#3498db")
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{bar.get_height():.1f}%", ha="center", va="bottom",
                 fontsize=9, color="#e74c3c")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=2, fontsize=10)
    plt.title("Accuracy vs. Sparsity Trade-off for Different λ Values",
              fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig("tradeoff_summary.png", dpi=150, bbox_inches="tight")
    print("Trade-off summary plot saved → tradeoff_summary.png")
    plt.show()

# REPORT GENERATION (runs after all experiments complete)

def generate_report(results: list[dict]) -> None:
    """
    Generate report.md using the REAL results from the training runs.
    Called only after all epochs and evaluations are complete.
    """

    # Build the results table rows from actual numbers
    table_rows = ""
    for r in results:
        table_rows += f"| {r['lambda']} | {r['accuracy']:.2f}% | {r['sparsity']:.1f}% |\n"

    # Find best model (highest accuracy)
    best = max(results, key=lambda r: r["accuracy"])
    worst_acc = min(results, key=lambda r: r["accuracy"])
    most_sparse = max(results, key=lambda r: r["sparsity"])

    report_content = f"""# Self-Pruning Neural Network — Report

{table_rows}
### Interpretation

- **Best accuracy**: λ = {best['lambda']} → {best['accuracy']:.2f}% test accuracy
- **Most sparse**: λ = {most_sparse['lambda']} → {most_sparse['sparsity']:.1f}% of weights pruned
- **Lowest accuracy**: λ = {worst_acc['lambda']} → {worst_acc['accuracy']:.2f}% (heavy pruning cost)


"""

    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("Report saved → report.md")


# MAIN

def main():
    # ── Load data (downloaded once, cached in ./data) ─────────────────────
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(BATCH_SIZE)
    print(f"Train batches: {len(train_loader)} | Test batches: {len(test_loader)}")

    # ── Run experiments for each lambda ───────────────────────────────────
    all_results = []
    for lam in LAMBDA_VALUES:
        result = run_experiment(lam, train_loader, test_loader)
        all_results.append(result)

    # ── Print summary table ───────────────────────────────────────────────
    print("\n\n" + "="*55)
    print("  SUMMARY TABLE")
    print("="*55)
    print(f"  {'Lambda':<12} {'Test Accuracy':<18} {'Sparsity Level'}")
    print(f"  {'-'*12} {'-'*18} {'-'*14}")
    for r in all_results:
        print(f"  {r['lambda']:<12} {r['accuracy']:<18.2f} {r['sparsity']:.1f}%")
    print("="*55)

    # ── Plot gate distributions for all lambda values ─────────────────────
    plot_gate_distribution(all_results)

    # ── Plot accuracy vs sparsity trade-off ──────────────────────────────
    plot_tradeoff_summary(all_results)

    # ── Generate report with REAL results ────────────────────────────────
    generate_report(all_results)

    print("\nAll done! Check gate_distributions.png, tradeoff_summary.png, and report.md")


if __name__ == "__main__":
    main()
