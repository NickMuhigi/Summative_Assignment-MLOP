# ML Image Classifier Pipeline

This project demonstrates the full **Machine Learning lifecycle**, from data acquisition to model deployment and scaling, using **image data**. It includes model retraining triggers, cloud deployment, API access, and stress testing using **Locust**.

## Project Description

This project aims to classify fruit images using a CNN-based classification model. It demonstrates the end-to-end ML cycle including data preprocessing, model training, evaluation, deployment with FastAPI, and monitoring on a cloud platform.

The system supports:
- Real-time and batch predictions
- Uploading new training data
- Triggering retraining via a UI
- Visualization of dataset features
- Flood testing with **Locust**
- Scaling using Docker containers

---

## Directory Structure
## 🗂️ Project Directory Structure

| Folder/File               | Description                                    |
|---------------------------|------------------------------------------------|
| `README.md`               | Project overview and setup guide               |
| `notebook/`               | Jupyter notebooks for model training/testing   |
| └── `Fruits_Classifier.ipynb` | ML training and evaluation notebook        |
| `src/`                    | Source code and project logic                  |
| ├── `app.py`              | Main FastAPI app runner                        |
| ├── `api.py`              | API route and logic handler                    |
| ├── `model.py`            | Model training and saving logic                |
| ├── `prediction.py`       | Model loading and prediction functions         |
| ├── `preprocessing.py`    | Data loading and preprocessing utilities       |
| ├── `retrain.py`          | Model retraining logic                         |
| ├── `render.yaml`         | Render deployment configuration                |
| └── `locustfile.py`       | Locust script for load testing                 |
| `data/`                   | Training and testing image datasets            |
| ├── `train/`              | Training image data                            |
| └── `test/`               | Testing image data                             |
| `models/`                 | Saved trained model(s)                         |
| └── `best_fruit_classifier.pkl`, `fruit_classifier.pkl`      | Trained model files                             |



---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/NickMuhigi/Summative_Assignment-MLOP.git
cd Fruit Classifier
```
### 2. Run the Jupyter Notebook

```bash
cd notebook
jupyter notebook Fruit_Classifier.ipynb
```
### 3. Run the FastAPI Server

```bash
cd src
uvicorn src.api:app --reload
```
### 4. Run the Streamlit UI

```bash
cd src
streamlit run streamlit_app.py
```
### 5. Run Locust for Load Testing

```bash
cd src
locust -f locustfile.py
```

## Model Evaluation Metrics
All evaluations are performed in notebook/fruit_classifier.ipynb and include:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Training and validation loss curves

## Retraining Workflow
- Upload new data (multiple images) through the UI
- Press a button to trigger retraining
- The model is re-evaluated and saved
- Predictions automatically use the updated model

## Flood Request Simulation (Locust Results)
| **Metric**                    | **Value**         |
|------------------------------|-------------------|
| Endpoint Tested              | POST /predict     |
| Total Requests Made          | 5                 |
| Failed Requests              | 0                 |
| Median Response Time         | 23 ms             |
| 95th Percentile Response     | 2000 ms           |
| 99th Percentile Response     | 2000 ms           |
| Average Response Time        | 428.49 ms         |
| Min / Max Response Time      | 22 ms / 2049 ms   |
| Average Response Size        | 128 bytes         |
| Current RPS                  | 0                 |
| Current Failures/sec         | 0                 |


More: https://docs.google.com/document/d/1_0D8DbsV9w6LPHBx9nXPjyQly3tBeKBzt6IsvDTlGV8/edit?usp=sharing


## Demo 
**Video:** https://youtu.be/DqVadEqKAvU

