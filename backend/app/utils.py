#for visualization

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import io
import base64
import numpy as np


def generate_decision_boundary(model, X, y):
    # Define grid limits
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Create mesh grid
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    # Predict for each grid point
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.contourf(
        xx, yy, Z,
        alpha=0.3,
        cmap="coolwarm"
    )

    ax.scatter(
        X[:, 0], X[:, 1],
        c=y,
        cmap="coolwarm",
        edgecolor="k",
        s=30
    )

    ax.set_xlabel("Col1")
    ax.set_ylabel("Col2")
    ax.set_title("Decision Boundary")

    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


def generate_tree_image(model):
    fig, ax = plt.subplots(figsize=(20, 10))
    
    plot_tree(
        model,
        filled=True,
        feature_names=["x1", "x2"],
        class_names=["Class 0", "Class 1"],
        ax=ax
    )

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    tree_img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return tree_img_base64
