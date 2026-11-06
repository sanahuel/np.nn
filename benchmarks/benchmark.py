"""MLP on MNIST: np.nn vs PyTorch (CPU). Writes results/benchmark_mlp_mnist.{csv,md}.

    python benchmarks/benchmark.py [--epochs 5] [--batch-size 128] [--quick]
"""
import argparse
import csv
import platform
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def run_npnn(data, epochs, batch_size):
    from nn import Sequential, Linear, ReLU, Adam, set_seed
    (Xtr, ytr), (Xte, yte) = data
    set_seed(0)
    model = Sequential([Linear(784, 256), ReLU(), Linear(256, 128), ReLU(), Linear(128, 10)])
    opt, times = Adam(lr=1e-3), []
    for _ in range(epochs):
        t0 = time.perf_counter()
        model.fit(Xtr, ytr, epochs=1, batch_size=batch_size, optimizer=opt, verbose=False)
        times.append(time.perf_counter() - t0)
    _, acc = model.evaluate(Xte, yte)
    return {"framework": "np.nn", "s_per_epoch": float(np.mean(times)), "test_acc": acc,
            "torch_version": "", "torch_threads": ""}


def run_torch(data, epochs, batch_size):
    try:
        import torch
        from torch import nn as tnn
    except ImportError:
        print("PyTorch is not installed: skipped (pip install -e '.[bench]')")
        return None
    (Xtr, ytr), (Xte, yte) = data
    torch.manual_seed(0)
    model = tnn.Sequential(tnn.Linear(784, 256), tnn.ReLU(), tnn.Linear(256, 128),
                           tnn.ReLU(), tnn.Linear(128, 10))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = tnn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    times = []
    for _ in range(epochs):
        t0 = time.perf_counter()
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        times.append(time.perf_counter() - t0)
    model.eval()
    with torch.no_grad():
        acc = (model(torch.from_numpy(Xte)).argmax(1).numpy() == yte).mean()
    return {"framework": "PyTorch", "s_per_epoch": float(np.mean(times)), "test_acc": float(acc),
            "torch_version": torch.__version__, "torch_threads": torch.get_num_threads()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--quick", action="store_true", help="10k train samples and 1 epoch")
    args = ap.parse_args()

    from nn import set_default_dtype
    from nn.utils import load_mnist
    set_default_dtype(np.float32)
    (Xtr, ytr), (Xte, yte) = load_mnist()
    epochs = args.epochs
    if args.quick:
        Xtr, ytr, epochs = Xtr[:10000], ytr[:10000], 1
    data = ((Xtr, ytr), (Xte, yte))

    rows = [r for r in (run_npnn(data, epochs, args.batch_size),
                        run_torch(data, epochs, args.batch_size)) if r]
    cpu = platform.processor() or platform.machine()
    for r in rows:
        r.update(epochs=epochs, batch_size=args.batch_size, cpu=cpu, numpy_version=np.__version__)

    table = ["| Framework | s / epoch | Test accuracy |", "|---|---|---|"]
    table += [f"| {r['framework']} | {r['s_per_epoch']:.2f} | {r['test_acc'] * 100:.2f}% |" for r in rows]
    table = "\n".join(table)
    print(table)

    if not args.quick:
        RESULTS.mkdir(exist_ok=True)
        fields = ["framework", "epochs", "batch_size", "s_per_epoch", "test_acc", "cpu",
                  "numpy_version", "torch_version", "torch_threads"]
        with open(RESULTS / "benchmark_mlp_mnist.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        (RESULTS / "benchmark_mlp_mnist.md").write_text(table + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
