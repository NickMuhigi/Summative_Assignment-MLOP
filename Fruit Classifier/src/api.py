from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import joblib
from PIL import Image
import numpy as np
import io
import os
import time
import logging
from datetime import datetime
from typing import List, Dict
import threading
from src.preprocessing import load_images_from_folder, encode_labels
from src.model import train_model, evaluate_model
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Fruit Classifier API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for monitoring
prediction_count = 0
model_load_time = None
training_status = {"is_training": False, "progress": 0, "message": "Ready"}

# Load the model and label encoder once on startup
try:
    model, le = joblib.load('models/fruit_classifier.pkl')
    model_load_time = datetime.now()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model, le = None, None

def preprocess_image(image_bytes, image_size=(64, 64)):
    """Preprocess uploaded image for prediction"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize(image_size)
        img_array = np.array(img).flatten().reshape(1, -1)
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

def save_uploaded_file(upload_file: UploadFile, destination: str):
    """Save uploaded file to destination"""
    try:
        with open(destination, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        return True
    except Exception as e:
        logger.error(f"Failed to save file {destination}: {e}")
        return False

@app.get("/")
def root():
    """Root endpoint with API information"""
    uptime = datetime.now() - model_load_time if model_load_time else None
    return {
        "message": "Welcome to Fruit Classifier API! 🍎",
        "endpoints": {
            "predict": "POST /predict - Upload an image for prediction",
            "retrain": "POST /retrain - Upload multiple images to retrain model",
            "status": "GET /status - Get API and model status",
            "health": "GET /health - Health check endpoint"
        },
        "uptime_seconds": uptime.total_seconds() if uptime else None,
        "total_predictions": prediction_count
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "api_version": "1.0.0"
    }

@app.get("/status")
def get_status():
    """Get detailed API and model status"""
    global prediction_count, model_load_time, training_status
    
    uptime = datetime.now() - model_load_time if model_load_time else None
    
    return {
        "api_status": "online",
        "model_loaded": model is not None,
        "model_classes": list(le.classes_) if le else [],
        "total_predictions": prediction_count,
        "uptime_seconds": uptime.total_seconds() if uptime else None,
        "training_status": training_status,
        "last_model_update": model_load_time.isoformat() if model_load_time else None
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict fruit class from uploaded image"""
    global prediction_count
    
    if not model or not le:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        processed_img = preprocess_image(image_bytes)
        
        # Make prediction
        start_time = time.time()
        prediction_encoded = model.predict(processed_img)
        prediction_proba = model.predict_proba(processed_img)
        inference_time = time.time() - start_time
        
        # Get results
        predicted_label = le.inverse_transform(prediction_encoded)[0]
        confidence = float(np.max(prediction_proba))
        
        # Update metrics
        prediction_count += 1
        
        logger.info(f"Prediction made: {predicted_label} (confidence: {confidence:.3f})")
        
        return JSONResponse(content={
            "prediction": predicted_label,
            "confidence": confidence,
            "inference_time_ms": round(inference_time * 1000, 2),
            "all_probabilities": {
                class_name: float(prob) 
                for class_name, prob in zip(le.classes_, prediction_proba[0])
            }
        })
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

def retrain_model_background(uploaded_files: List[UploadFile], class_labels: List[str]):
    """Background task for model retraining"""
    global model, le, model_load_time, training_status
    
    try:
        training_status["is_training"] = True
        training_status["message"] = "Starting retraining..."
        training_status["progress"] = 10
        
        # Create temporary directory for new data
        temp_dir = "temp_training_data"
        os.makedirs(temp_dir, exist_ok=True)
        
        training_status["message"] = "Saving uploaded files..."
        training_status["progress"] = 20
        
        # Save uploaded files with their labels
        saved_files = []
        for file, label in zip(uploaded_files, class_labels):
            label_dir = os.path.join(temp_dir, label)
            os.makedirs(label_dir, exist_ok=True)
            
            file_path = os.path.join(label_dir, file.filename)
            if save_uploaded_file(file, file_path):
                saved_files.append(file_path)
        
        training_status["message"] = f"Saved {len(saved_files)} files. Loading existing data..."
        training_status["progress"] = 40
        
        # Combine with existing training data
        existing_train_dir = "data/train"
        if os.path.exists(existing_train_dir):
            # Copy existing data to temp directory
            for class_name in os.listdir(existing_train_dir):
                src_class_dir = os.path.join(existing_train_dir, class_name)
                dst_class_dir = os.path.join(temp_dir, class_name)
                
                if os.path.isdir(src_class_dir):
                    os.makedirs(dst_class_dir, exist_ok=True)
                    for filename in os.listdir(src_class_dir):
                        src_file = os.path.join(src_class_dir, filename)
                        dst_file = os.path.join(dst_class_dir, filename)
                        shutil.copy2(src_file, dst_file)
        
        training_status["message"] = "Training new model..."
        training_status["progress"] = 60
        
        # Train new model
        train_model(temp_dir, save_path='models/fruit_classifier_new.pkl')
        
        training_status["message"] = "Evaluating model..."
        training_status["progress"] = 80
        
        # Load and validate new model
        new_model, new_le = joblib.load('models/fruit_classifier_new.pkl')
        
        # Replace old model with new one
        model, le = new_model, new_le
        
        # Move new model to production
        shutil.move('models/fruit_classifier_new.pkl', 'models/fruit_classifier.pkl')
        
        # Update existing training data
        if os.path.exists(existing_train_dir):
            shutil.rmtree(existing_train_dir)
        shutil.move(temp_dir, existing_train_dir)
        
        training_status["message"] = "Retraining completed successfully!"
        training_status["progress"] = 100
        training_status["is_training"] = False
        
        model_load_time = datetime.now()
        logger.info("Model retrained successfully")
        
    except Exception as e:
        training_status["message"] = f"Retraining failed: {str(e)}"
        training_status["is_training"] = False
        training_status["progress"] = 0
        logger.error(f"Retraining failed: {e}")
        
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

@app.post("/retrain")
async def retrain(files: List[UploadFile] = File(...)):
    """Retrain model with new data"""
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=409, detail="Model is already being retrained")
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    # Validate files
    for file in files:
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not an image")
    
    try:
        # For simplicity, we'll use the filename to determine the class
        # In a real scenario, you'd want to pass class labels separately
        class_labels = []
        for file in files:
            # Extract class from filename (e.g., "apple_001.jpg" -> "apple")
            filename_parts = file.filename.lower().split('_')
            if len(filename_parts) > 0:
                class_name = filename_parts[0]
                class_labels.append(class_name)
            else:
                class_labels.append("unknown")
        
        # Start retraining in background
        training_thread = threading.Thread(
            target=retrain_model_background, 
            args=(files, class_labels)
        )
        training_thread.start()
        
        return JSONResponse(content={
            "message": "Retraining started",
            "files_count": len(files),
            "status": "training_initiated"
        })
        
    except Exception as e:
        logger.error(f"Failed to start retraining: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start retraining: {str(e)}")

@app.get("/training-status")
def get_training_status():
    """Get current training status"""
    return training_status

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)