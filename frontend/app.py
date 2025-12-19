import streamlit as st
import requests
import base64
import json

st.set_page_config(page_title="Decision Tree Visualizer", layout="wide")

st.title("🌳 Decision Tree Visualizer")
st.markdown("""
Interactively explore how **Decision Tree hyperparameters**
affect model structure and performance on a **non-linear dataset**.
""")

## Backend API URL
BACKEND_URL = "https://decision-tree-visualizer-7gpt.onrender.com/train"

st.sidebar.header("Split Criteria")

criterion = st.sidebar.selectbox(
    "criterion", ["gini", "entropy", "log_loss"]
)

splitter = st.sidebar.selectbox(
    "splitter", ["best", "random"]
)

st.sidebar.header("Tree Complexity")

max_depth_ui = st.sidebar.slider(
    "max_depth (0 = None / unlimited)",
    min_value=0,
    max_value=20,
    value=5
)


min_samples_split = st.sidebar.slider(
    "min_samples_split",
    min_value=2,
    max_value=300,
    value=2
)

min_samples_leaf = st.sidebar.slider(
    "min_samples_leaf",
    min_value=1,
    max_value=200,
    value=1
)

max_leaf_nodes = st.sidebar.slider(
    "max_leaf_nodes (0 = None)",
    min_value=0,
    max_value=500,
    value=0
)

st.sidebar.header("Feature Selection")

max_features_option = st.sidebar.selectbox(
    "max_features",
    ["None", "sqrt", "log2"]
)

st.sidebar.header("Pruning")

ccp_alpha = st.sidebar.slider(
    "ccp_alpha (pruning strength)",
    min_value=0.0,
    max_value=0.1,
    value=0.0,
    step=0.005
)

random_state = st.sidebar.number_input(
    "random_state",
    min_value=0,
    max_value=9999,
    value=69
)

def build_payload():
    return {
        "criterion": criterion,
        "splitter": splitter,
        "max_depth": None if max_depth_ui == 0 else max_depth_ui,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "min_weight_fraction_leaf": 0.0,
        "max_features": None if max_features_option == "None" else max_features_option,
        "max_leaf_nodes": None if max_leaf_nodes == 0 else max_leaf_nodes,
        "min_impurity_decrease": 0.0,
        "class_weight": None,
        "ccp_alpha": ccp_alpha,
        "random_state": random_state
    }

if st.button("🚀 Train Decision Tree"):
    with st.spinner("Training model..."):
        response = requests.post(
            BACKEND_URL,
            headers={"Content-Type": "application/json"},
            json=build_payload(),
            timeout=90
        )

    if response.status_code != 200:
        st.error(f"Backend error {response.status_code}")
        st.text(response.text)
        st.stop()

    result = response.json()

    # ✅ EVERYTHING that uses `result` goes INSIDE
    st.subheader("🟦 Decision Boundary")

    boundary_img = base64.b64decode(result["decision_boundary_image_base64"])
    st.image(boundary_img, width=800)

    st.subheader("📊 Model Metrics")

    metrics = result["metrics"]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
    col2.metric("Precision", f"{metrics['precision']:.3f}")
    col3.metric("Recall", f"{metrics['recall']:.3f}")
    col4.metric("F1 Score", f"{metrics['f1']:.3f}")

    st.write("Confusion Matrix")
    st.table(metrics["confusion_matrix"])

    st.subheader("🌳 Decision Tree")
    img_bytes = base64.b64decode(result["tree_image_base64"])
    st.image(img_bytes, width=800)
