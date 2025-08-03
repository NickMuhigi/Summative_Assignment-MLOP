#!/usr/bin/env python3
"""
Standalone retraining script for the fruit classifier
Can be called directly or triggered by the API
"""

import os
import sys
import joblib
import shutil
import json
import time
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from preprocessing import load_images_from_folder, encode_labels
except ImportError:
    print("❌ Could not import preprocessing functions from src/")
    print("Make sure src/preprocessing.py exists with load_images_from_folder and encode_labels functions")
    sys.exit(1)

def update_training_status(is_training=False, progress=0, message="Ready", start_time=None, end_time=None):
    """Update training status to JSON file for Streamlit to read"""
    status_file = "training_status.json"
    
    status = {
        "is_training": is_training,
        "progress": progress,
        "message": message,
        "start_time": start_time,
        "end_time": end_time,
        "last_updated": time.time()
    }
    
    try:
        with open(status_file, 'w') as f:
            json.dump(status, f, indent=2)
        print(f"📝 Status updated: {message} ({progress}%)")
    except Exception as e:
        print(f"⚠️ Could not update status file: {e}")

def retrain_model(train_dir='../data/train', test_dir='../data/test', model_path='../models/fruit_classifier.pkl'):
    """
    Retrain the fruit classifier model with proper status updates
    
    Args:
        train_dir: Directory containing training data
        test_dir: Directory containing test data (optional)
        model_path: Path to save the new model
    """
    
    start_time = time.time()
    
    try:
        print("🚀 Starting model retraining...")
        print(f"Training data directory: {train_dir}")
        print(f"Model will be saved to: {model_path}")
        
        update_training_status(True, 5, "Starting retraining process...", start_time)
        
        # Check if training directory exists
        if not os.path.exists(train_dir):
            error_msg = f"Training directory not found: {train_dir}"
            update_training_status(False, 0, f"Error: {error_msg}")
            raise FileNotFoundError(error_msg)
        
        update_training_status(True, 10, "Loading training data...")
        
        # Load training data
        print("📁 Loading training data...")
        try:
            X_train, y_train = load_images_from_folder(train_dir)
            print(f"Loaded {len(X_train)} training images")
            print(f"Classes found: {set(y_train)}")
            
            if len(X_train) == 0:
                error_msg = "No training data found!"
                update_training_status(False, 0, f"Error: {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            error_msg = f"Failed to load training data: {str(e)}"
            update_training_status(False, 0, f"Error: {error_msg}")
            raise Exception(error_msg)
        
        update_training_status(True, 25, f"Loaded {len(X_train)} images from {len(set(y_train))} classes")
        
        # Encode labels
        print("🏷️ Encoding labels...")
        update_training_status(True, 30, "Encoding labels...")
        
        y_train_encoded, le = encode_labels(y_train)
        print(f"Label encoder classes: {list(le.classes_)}")
        
        update_training_status(True, 35, "Training Random Forest model...")
        
        # Train model
        print("🤖 Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            max_depth=10,
            n_jobs=-1  # Use all CPU cores
        )
        
        update_training_status(True, 40, "Fitting model to data...")
        model.fit(X_train, y_train_encoded)
        print("✅ Model training completed!")
        
        update_training_status(True, 70, "Model training completed, saving model...")
        
        # Create backup of old model if it exists
        if os.path.exists(model_path):
            backup_path = model_path.replace('.pkl', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
            shutil.copy2(model_path, backup_path)
            print(f"📦 Backed up old model to: {backup_path}")
        
        # Save new model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump((model, le), model_path)
        print(f"💾 New model saved to: {model_path}")
        
        update_training_status(True, 80, "Model saved, evaluating performance...")
        
        # Evaluate on test data if available
        accuracy = None
        if os.path.exists(test_dir):
            print("📊 Evaluating model on test data...")
            try:
                X_test, y_test = load_images_from_folder(test_dir)
                if len(X_test) > 0:
                    y_test_encoded = le.transform(y_test)
                    y_pred = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test_encoded, y_pred)
                    print(f"🎯 Test Accuracy: {accuracy:.4f}")
                    print("\n📈 Classification Report:")
                    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
                    
                    update_training_status(True, 90, f"Evaluation complete. Accuracy: {accuracy:.1%}")
                else:
                    print("⚠️ No test data found for evaluation")
                    update_training_status(True, 90, "No test data available for evaluation")
            except Exception as e:
                print(f"⚠️ Could not evaluate on test data: {str(e)}")
                update_training_status(True, 90, f"Evaluation failed: {str(e)}")
        else:
            update_training_status(True, 90, "No test directory found, skipping evaluation")
        
        # Training summary
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\n" + "="*50)
        print("🎉 RETRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Training samples: {len(X_train)}")
        print(f"Classes: {len(le.classes_)}")
        print(f"Class names: {', '.join(le.classes_)}")
        print(f"Model type: {type(model).__name__}")
        print(f"Features: {model.n_features_in_}")
        print(f"Training time: {training_time:.2f} seconds")
        print(f"Model saved: {model_path}")
        if accuracy:
            print(f"Test accuracy: {accuracy:.1%}")
        print("="*50)
        
        # Final status update
        success_msg = f"Retraining completed! {len(X_train)} samples, {len(le.classes_)} classes"
        if accuracy:
            success_msg += f", {accuracy:.1%} accuracy"
            
        update_training_status(False, 100, success_msg, start_time, end_time)
        
        return model, le, accuracy
        
    except Exception as e:
        end_time = time.time()
        error_msg = f"Retraining failed: {str(e)}"
        print(f"❌ {error_msg}")
        update_training_status(False, 0, error_msg, start_time, end_time)
        raise

def check_data_quality(train_dir='../data/train'):
    """Check the quality and distribution of training data"""
    print("🔍 Checking data quality...")
    
    if not os.path.exists(train_dir):
        print(f"❌ Training directory not found: {train_dir}")
        return False
    
    class_info = {}
    total_images = 0
    
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_info[class_name] = len(images)
            total_images += len(images)
    
    if total_images == 0:
        print("❌ No images found in training directory!")
        return False
    
    print(f"📊 Data Summary:")
    print(f"Total images: {total_images}")
    print(f"Number of classes: {len(class_info)}")
    print(f"Images per class:")
    
    min_images = min(class_info.values()) if class_info else 0
    max_images = max(class_info.values()) if class_info else 0
    
    for class_name, count in sorted(class_info.items()):
        status = "⚠️" if count < 10 else "✅"
        print(f"  {status} {class_name}: {count} images")
    
    # Data quality warnings
    if min_images < 5:
        print("⚠️ WARNING: Some classes have very few images (< 5). Consider collecting more data.")
    
    if max_images / min_images > 10 if min_images > 0 else False:
        print("⚠️ WARNING: Large imbalance between classes. Consider balancing the dataset.")
    
    if total_images < 50:
        print("⚠️ WARNING: Very small dataset. Model performance may be limited.")
    
    return True

def clean_old_backups(model_path='../models/fruit_classifier.pkl', keep_last=5):
    """Clean old backup files, keeping only the most recent ones"""
    try:
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path).replace('.pkl', '')
        
        # Find all backup files
        backup_files = []
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                if file.startswith(f"{model_name}_backup_") and file.endswith('.pkl'):
                    backup_path = os.path.join(model_dir, file)
                    backup_files.append((backup_path, os.path.getmtime(backup_path)))
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old backups
        if len(backup_files) > keep_last:
            for backup_path, _ in backup_files[keep_last:]:
                try:
                    os.remove(backup_path)
                    print(f"🗑️ Removed old backup: {os.path.basename(backup_path)}")
                except Exception as e:
                    print(f"⚠️ Could not remove backup {backup_path}: {e}")
                    
    except Exception as e:
        print(f"⚠️ Error cleaning backups: {e}")

def main():
    """Main function for command line usage"""
    print("🍎 Fruit Classifier - Model Retraining")
    print("="*40)
    
    # Initialize status
    update_training_status(True, 0, "Initializing retraining process...", time.time())
    
    # Parse command line arguments
    train_dir = sys.argv[1] if len(sys.argv) > 1 else '../data/train'
    test_dir = sys.argv[2] if len(sys.argv) > 2 else '../data/test'
    model_path = sys.argv[3] if len(sys.argv) > 3 else '../models/fruit_classifier.pkl'
    
    try:
        # Check data quality first
        update_training_status(True, 2, "Checking data quality...")
        if not check_data_quality(train_dir):
            error_msg = "Data quality check failed. Please check your training data."
            print(f"❌ {error_msg}")
            update_training_status(False, 0, error_msg)
            return False
        
        # Clean old backups
        clean_old_backups(model_path)
        
        # Proceed with retraining
        model, le, accuracy = retrain_model(train_dir, test_dir, model_path)
        
        print("\n🎊 Retraining process completed successfully!")
        return True
        
    except Exception as e:
        error_msg = f"Retraining failed: {str(e)}"
        print(f"❌ {error_msg}")
        update_training_status(False, 0, error_msg, end_time=time.time())
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
