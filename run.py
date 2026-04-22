"""
Run one experiment: build model, train, evaluate, log result.

Usage:
    python run.py "description"              # logs as status=keep
    python run.py "description" --baseline   # logs as status=baseline
    python run.py "description" --discard    # logs as status=discard
"""

import sys
import time
import subprocess
from prepare import load_data, evaluate, log_result


def get_git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def main():
    args = sys.argv[1:]
    status = "keep"
    description_parts = []

    for a in args:
        if a == "--baseline":
            status = "baseline"
        elif a == "--discard":
            status = "discard"
        else:
            description_parts.append(a)

    description = " ".join(description_parts) if description_parts else "experiment"

    # 1. Load data (frozen)
    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_data()

    # 2. Build model (editable)
    from model import build_model
    model = build_model()
    print(f"\nModel: {model.named_steps['model'].__class__.__name__}")

    # 3. Train
    t0 = time.time()
    model.fit(X_train, y_train)
    runtime = time.time() - t0
    print(f"Training time: {runtime:.2f}s")

    # 4. Evaluate on validation set only (test set stays locked)
    val_auc, val_f1, val_acc = evaluate(model, X_val, y_val, verbose=True)
    print(f"\nval_auc: {val_auc:.6f}")
    print(f"val_f1:  {val_f1:.6f}")
    print(f"val_acc: {val_acc:.6f}")
    print(f"runtime: {runtime:.2f}s")
    print(f"\n*** Test set is LOCKED — do not call evaluate(model, X_test, y_test) during the loop ***")

    # 5. Log
    commit = get_git_hash()
    log_result(commit, val_auc, val_f1, val_acc, status, description, runtime)
    print(f"\nResult logged to results.tsv (status={status})")


if __name__ == "__main__":
    main()
