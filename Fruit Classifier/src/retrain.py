#!/usr/bin/env python3
"""
Standalone retraining script for the fruit classifier
Can be called directly or triggered by the API
"""

import os
import sys
import joblib
import shutil
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from src.preprocessing import load_images_from_folder, encode_labels

def retrain_model(train_dir='data/train', test_dir='data/test', model_path='models/fruit_classifier.pkl'):
    """
    Retrain the fruit classifier model
    
    Args:
        train_dir: Directory containing training data
        test_dir: Directory containing test data (optional)
        model_path: Path to save the new model
    """
    
    print("🚀 Starting model retraining...")
    print(f"Training data directory: {train_dir}")
    print(f"Model will be saved to: {model_path}")
    
    # Check if training directory exists
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    
    # Load training data
    print("📁 Loading training data...")
    try:
        X_train, y_train = load_images_from_folder(train_dir)
        print(f"Loaded {len(X_train)} training images")
        print(f"Classes found: {set(y_train)}")
    except Exception as e:
        raise Exception(f"Failed to load training data: {str(e)}")
    
    if len(X_train) == 0:
        raise Exception("No training data found!")
    
    # Encode labels
    print("🏷️ Encoding labels...")
    y_train_encoded, le = encode_labels(y_train)
    print(f"Label encoder classes: {list(le.classes_)}")
    
    # Train model
    print("🤖 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train_encoded)
    print("✅ Model training completed!")
    
    # Create backup of old model if it exists
    if os.path.exists(model_path):
        backup_path = model_path.replace('.pkl', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
        shutil.copy2(model_path, backup_path)
        print(f"📦 Backed up old model to: {backup_path}")
    
    # Save new model
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump((model, le), model_path)
    print(f"💾 New model saved to: {model_path}")
    
    # Evaluate on test data if available
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
            else:
                print("⚠️ No test data found for evaluation")
        except Exception as e:
            print(f"⚠️ Could not evaluate on test data: {str(e)}")
    
    # Training summary
    print("\n" + "="*50)
    print("🎉 RETRAINING COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Training samples: {len(X_train)}")
    print(f"Classes: {len(le.classes_)}")
    print(f"Model type: {type(model).__name__}")
    print(f"Features: {model.n_features_in_}")
    print(f"Model saved: {model_path}")
    print("="*50)
    
    return model, le, accuracy if 'accuracy' in locals() else None

def check_data_quality(train_dir='data/train'):
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

def main():
    """Main function for command line usage"""
    print("🍎 Fruit Classifier - Model Retraining")
    print("="*40)
    
    # Parse command line arguments
    train_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/train'
    test_dir = sys.argv[2] if len(sys.argv) > 2 else 'data/test'
    model_path = sys.argv[3] if len(sys.argv) > 3 else 'models/fruit_classifier.pkl'
    
    try:
        # Check data quality first
        if not check_data_quality(train_dir):
            print("❌ Data quality check failed. Please check your training data.")
            return False
        
        # Proceed with retraining
        model, le, accuracy = retrain_model(train_dir, test_dir, model_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Retraining failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)