import torch
import torch.nn as nn
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype
import copy


class Probe(nn.Module):
    def __init__(self, activation_dim: int, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Linear(activation_dim, 1, bias=False, dtype=dtype)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@beartype
def train_sklearn_probe(
    train_inputs: Float[torch.Tensor, "train_dataset_size d_model"],
    train_labels: Int[torch.Tensor, "train_dataset_size"],
    test_inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    test_labels: Int[torch.Tensor, "test_dataset_size"],
    max_iter: int = 100,
    C: float = 1.0,  # default sklearn value
    verbose: bool = False,
    l1_ratio: float | None = None,
) -> tuple[torch.Tensor, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    train_inputs = train_inputs.to(dtype=torch.float32)
    test_inputs = test_inputs.to(dtype=torch.float32)

    # Convert torch tensors to numpy arrays
    train_inputs_np = train_inputs.cpu().numpy()
    train_labels_np = train_labels.cpu().numpy()
    test_inputs_np = test_inputs.cpu().numpy()
    test_labels_np = test_labels.cpu().numpy()

    # Initialize the LogisticRegression model
    if l1_ratio is not None:
        # Use Elastic Net regularization
        probe = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=C,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            verbose=int(verbose),
        )
    else:
        # Use L2 regularization
        probe = LogisticRegression(
            penalty="l2", C=C, max_iter=max_iter, verbose=int(verbose)
        )

    # Train the model
    probe.fit(train_inputs_np, train_labels_np)

    # Compute accuracies
    train_accuracy = accuracy_score(train_labels_np, probe.predict(train_inputs_np))
    test_accuracy = accuracy_score(test_labels_np, probe.predict(test_inputs_np))

    if verbose:
        print("\nTraining completed.")
        print(f"Train accuracy: {train_accuracy}, Test accuracy: {test_accuracy}\n")

    return probe, test_accuracy


# Helper function to test the probe
@beartype
def test_sklearn_probe(
    inputs: Float[torch.Tensor, "dataset_size d_model"],
    labels: Int[torch.Tensor, "dataset_size"],
    probe: torch.Tensor,
) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    inputs = inputs.to(dtype=torch.float32)
    inputs_np = inputs.cpu().numpy()
    labels_np = labels.cpu().numpy()
    predictions = probe.predict(inputs_np)
    return accuracy_score(labels_np, predictions)  # type: ignore


@jaxtyped(typechecker=beartype)
@torch.no_grad
def test_probe_gpu(
    inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    labels: Int[torch.Tensor, "test_dataset_size"],
    batch_size: int,
    probe: Probe,
) -> float:
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        corrects_0 = []
        corrects_1 = []
        all_corrects = []
        losses = []

        for i in range(0, len(labels), batch_size):
            acts_BD = inputs[i : i + batch_size]
            labels_B = labels[i : i + batch_size]
            logits_B = probe(acts_BD)
            preds_B = (logits_B > 0.0).long()
            correct_B = (preds_B == labels_B).float()

            all_corrects.append(correct_B)
            corrects_0.append(correct_B[labels_B == 0])
            corrects_1.append(correct_B[labels_B == 1])

            loss = criterion(logits_B, labels_B.to(dtype=probe.net.weight.dtype))
            losses.append(loss)

        accuracy_all = torch.cat(all_corrects).mean().item()

    return accuracy_all


def _standardise(train_x: torch.Tensor, test_x: torch.Tensor, eps: float = 1e-6):
    """
    Z-score the activations on the training set and reuse μ,σ on the test set.
    Returns (train_std, test_std, mean, std).
    """
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0).clamp_min(eps)  # avoid divide-by-zero
    return (train_x - mean) / std, (test_x - mean) / std, mean, std


def _scale_only(train_x: torch.Tensor, test_x: torch.Tensor, eps: float = 1e-6):
    """
    Divide each dimension by its σ (computed on the *training* set).
    No centring, so a bias-free probe survives folding unchanged.
    """
    sigma = train_x.std(dim=0).clamp_min(eps)
    return train_x / sigma, test_x / sigma, sigma


@jaxtyped(typechecker=beartype)
def train_probe_gpu(
    train_inputs: Float[torch.Tensor, "train_dataset_size d_model"],
    train_labels: Int[torch.Tensor, "train_dataset_size"],
    test_inputs: Float[torch.Tensor, "test_dataset_size d_model"],
    test_labels: Int[torch.Tensor, "test_dataset_size"],
    dim: int,
    batch_size: int,
    epochs: int,
    lr: float,
    verbose: bool = False,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 10,
) -> tuple[Probe, float]:
    """
    Trains a *bias-free* linear probe on z-scored activations and folds the
    transform back into the weight vector so the returned probe consumes **raw
    activations**.
    """
    device = train_inputs.device
    model_dtype = train_inputs.dtype
    print(f"Training probe with dim={dim}, device={device}, dtype={model_dtype}")

    # ---------- 1. Standardise activations ----------
    train_z, test_z, sigma_D = _scale_only(train_inputs, test_inputs)

    probe = Probe(dim, model_dtype).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_acc, best_probe, patience = 0.0, None, 0
    for epoch in range(epochs):
        perm = torch.randperm(len(train_z), device=device)
        for i in range(0, len(train_z), batch_size):
            idx = perm[i : i + batch_size]
            acts_BD = train_z[idx]
            labels_B = train_labels[idx]
            logits_B = probe(acts_BD)
            loss = criterion(logits_B, labels_B.to(dtype=model_dtype))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # quick helper avoids recomputing σ twice inside loop
        train_acc = test_probe_gpu(train_z, train_labels, batch_size, probe)
        test_acc = test_probe_gpu(test_z, test_labels, batch_size, probe)

        if test_acc > best_acc:
            best_acc, best_probe, patience = test_acc, copy.deepcopy(probe), 0
        else:
            patience += 1

        if verbose:
            print(
                f"Epoch {epoch + 1}/{epochs}   "
                f"loss={loss.item():.4f}  train_acc={train_acc:.3f}  "
                f"test_acc={test_acc:.3f}"
            )
        if patience >= early_stopping_patience:
            if verbose:
                print(f"Early-stopped at epoch {epoch + 1}")
            break

    assert best_probe is not None

    # ---------- 3. Fold μ,σ back into weight ----------
    with torch.no_grad():
        w = best_probe.net.weight.squeeze()  # (D,)
        w_folded = w / sigma_D  # bring to raw space
        best_probe.net.weight.copy_(w_folded.unsqueeze(0))

    test_acc = test_probe_gpu(test_inputs, test_labels, batch_size, best_probe)
    print(f"Test accuracy after folding: {test_acc:.3f}")

    # Probe now consumes raw activations; no bias term needed.
    return best_probe, best_acc
