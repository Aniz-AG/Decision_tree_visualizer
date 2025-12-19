from fastapi import FastAPI
from app.schemas import TreeParams
from app.model import train_decision_tree
from app.utils import generate_tree_image,generate_decision_boundary

app=FastAPI(title="Decision Tree Visualiser API")

@app.post("/train")
def train_tree(params:TreeParams):
    model, metrics,X_train,y_train=train_decision_tree(params)
    tree_image=generate_tree_image(model)
    boundary_image=generate_decision_boundary(model, X_train, y_train)
    return {
        "metrics": metrics,
        "tree_image_base64": tree_image,
        "decision_boundary_image_base64":boundary_image
    }