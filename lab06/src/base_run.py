"""Step 1: the reference run - the best Lab 3 model (CNN, Adam 1e-3, batch 128, 5 epochs) trained by the
YAML-driven pipeline with seed, checkpoints and MLflow logging. Also the uninterrupted reference for step 7.

Output: results/runs/base-cnn-adam.json (+ MLflow run in experiment lab06-base)
"""

from runner import train

if __name__ == "__main__":
    s = train("base-cnn-adam")
    print(f"test accuracy {s['test_acc']:.4f}")
