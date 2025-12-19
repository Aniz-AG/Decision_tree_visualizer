# 🌳 Decision Tree Visualizer (Full-Stack ML App)

An interactive full-stack machine learning application to **understand Decision Trees visually** by exploring how **hyperparameters affect model behavior, metrics, and decision boundaries** on non-linearly separable data.

🔗 **Live App:**  
👉 https://decisiontreevisualizer-isuh7ercfbrw2vun7cxywt.streamlit.app/

---

## ✨ Features

- Interactive **Decision Tree hyperparameter tuning**
- Real-time model training using **scikit-learn**
- Visualizations:
  - Decision Tree structure
  - Decision boundary over data
  - Classification metrics (Accuracy, Precision, Recall, F1)
- Non-linearly separable dataset (`make_moons`)
- Clean separation of **frontend** and **backend**
- Fully deployable and reproducible using **Docker**

---

## 🧠 Tech Stack

### Backend
- FastAPI
- scikit-learn
- NumPy
- Matplotlib
- Docker
- Hosted on **Render**

### Frontend
- Streamlit
- Requests
- Hosted on **Streamlit Cloud**

---

## 🏗 Architecture

```text
User Browser
     ↓
Streamlit Frontend (UI)
     ↓ HTTP
FastAPI Backend (Dockerized)
     ↓
Decision Tree Training + Visualization
🚀 Live Deployment
Frontend (Streamlit):
https://decisiontreevisualizer-isuh7ercfbrw2vun7cxywt.streamlit.app/

Backend (FastAPI on Render):
Public API serving model training and visualizations

🖥 Run Locally
The entire project can be run locally by cloning the repository.

1️⃣ Clone the repository
bash
Copy code
git clone <YOUR_GITHUB_REPO_URL>
cd Decision-tree-visualiser
2️⃣ Backend (FastAPI + Docker)
Navigate to backend directory:

bash
Copy code
cd backend
Build the Docker image:

bash
Copy code
docker build -t decision-tree-backend .
Run the container:

bash
Copy code
docker run -p 8000:8000 decision-tree-backend
Open in browser:

bash
Copy code
http://localhost:8000/docs
3️⃣ Frontend (Streamlit)
Open a new terminal and navigate to frontend:

bash
Copy code
cd frontend
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Update backend URL in app.py if running locally:

python
Copy code
BACKEND_URL = "http://localhost:8000/train"
Run Streamlit:

bash
Copy code
streamlit run app.py
Open in browser:

arduino
Copy code
http://localhost:8501
🎯 Learning Goals
This project is designed to help understand:

How Decision Trees create axis-aligned decision boundaries

Overfitting vs underfitting via depth and pruning

Impact of hyperparameters like:

max_depth

min_samples_leaf

ccp_alpha

Why trees naturally handle non-linear data

How ML models are exposed via APIs

Real-world deployment using Docker and cloud platforms

📌 Why This Project Matters
Demonstrates ML + Backend + Frontend + Deployment

Mirrors real industry architecture

Goes beyond notebooks into production-style systems

Fully reproducible and cloud-hosted

🧑‍💻 Author
Built by Aniz AG
Undergraduate | Machine Learning & Systems Enthusiast