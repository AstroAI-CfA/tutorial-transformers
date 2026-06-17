"""Training loop and evaluation utilities for the AstroAI tutorial."""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def _run_epoch(model, loader, optimizer, criterion, device, train: bool):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n


def fit(model, train_loader, val_loader, n_epochs: int, lr: float, device,
        print_every: int = 10) -> dict:
    """Train model and return loss/accuracy history."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history = {k: [] for k in ("train_loss", "val_loss", "train_acc", "val_acc")}
    for epoch in range(1, n_epochs + 1):
        tl, ta = _run_epoch(model, train_loader, optimizer, criterion, device, train=True)
        vl, va = _run_epoch(model, val_loader,   None,      criterion, device, train=False)
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        history["train_acc"].append(ta);  history["val_acc"].append(va)
        if epoch % print_every == 0:
            print(f"  [{epoch:3d}/{n_epochs}]  "
                  f"train  loss {tl:.3f}  acc {ta:.2%}  |  "
                  f"val  loss {vl:.3f}  acc {va:.2%}")
    return history


def plot_history(history: dict, title: str = ""):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["train_loss"], label="train"); ax1.plot(history["val_loss"], label="val")
    ax1.set(xlabel="Epoch", ylabel="Loss",     title="Loss");     ax1.legend()
    ax2.plot(history["train_acc"], label="train"); ax2.plot(history["val_acc"], label="val")
    ax2.set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy"); ax2.legend()
    if title: fig.suptitle(title, fontsize=13)
    plt.tight_layout(); plt.show()


@torch.no_grad()
def get_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def plot_confusion_matrix(preds, labels, class_names, title=""):
    cm = confusion_matrix(labels, preds, normalize="true")
    fig, ax = plt.subplots(figsize=(len(class_names) * 1.5 + 1, len(class_names) * 1.5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=ax, colorbar=False, cmap="Blues", values_format=".2f"
    )
    if title: ax.set_title(title, fontsize=12)
    plt.tight_layout(); plt.show()
