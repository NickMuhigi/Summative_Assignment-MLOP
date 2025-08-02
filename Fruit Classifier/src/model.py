import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from src.preprocessing import load_images_from_folder, encode_labels
import warnings
warnings.filterwarnings('ignore')

def train_model(train_dir, save_path='models/fruit_classifier.pkl', compare_models=True, analyze_features=True):
    
    print("🚀 STARTING ENHANCED MODEL TRAINING")
    print("=" * 50)
    
    X, y = load_images_from_folder(train_dir)
    y_encoded, le = encode_labels(y)
    
    print(f"📊 Dataset loaded:")
    print(f"  Training samples: {len(X)}")
    print(f"  Features per sample: {X.shape[1]}")
    print(f"  Classes: {list(le.classes_)}")
    
    if analyze_features:
        print(f"\n🔬 PERFORMING FEATURE ANALYSIS")
        feature_results = analyze_image_features(train_dir, le.classes_)
        visualize_features(feature_results)
        interpret_features(feature_results)
    
    if compare_models:
        print(f"\n🤖 COMPARING MULTIPLE MODELS")
        best_model, model_results = compare_algorithms(X, y_encoded, le.classes_)
        
        display_model_comparison(model_results)
        
        print(f"\n🏆 Best Model Selected: {type(best_model).__name__}")
    else:
        print(f"\n🌲 Training Random Forest Model")
        best_model = RandomForestClassifier(n_estimators=100, random_state=42)
        best_model.fit(X, y_encoded)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump((best_model, le), save_path)
    print(f"\n💾 Model saved to {save_path}")
    
    return best_model, le

def evaluate_model(test_dir, model_path='models/fruit_classifier.pkl', detailed=True):
    
    print("📊 STARTING MODEL EVALUATION")
    print("=" * 40)
    
    model, le = joblib.load(model_path)
    X_test, y_test = load_images_from_folder(test_dir)
    y_test_encoded = le.transform(y_test)
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    print(f"✅ Overall Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    
    if detailed:
        print(f"\n📋 DETAILED CLASSIFICATION REPORT:")
        print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
        
        create_detailed_evaluation(y_test_encoded, y_pred, le.classes_)
        
        if hasattr(model, 'predict_proba'):
            analyze_prediction_confidence(model, X_test, y_test_encoded, le.classes_)
    else:
        print("Classification Report:\n", classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    
    return accuracy, y_pred


def analyze_image_features(data_dir, classes, sample_size=50):
    from PIL import Image
    
    print("🔍 Analyzing 3 Key Features: Color, Texture, Shape")
    feature_data = []

    for class_name in classes:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        sampled_files = image_files[:min(sample_size, len(image_files))]
        
        for filename in sampled_files:
            img_path = os.path.join(class_path, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img_array = np.array(img)
                mean_red = np.mean(img_array[:,:,0])
                mean_green = np.mean(img_array[:,:,1]) 
                mean_blue = np.mean(img_array[:,:,2])
                overall_texture = np.std(img_array)
                height, width = img_array.shape[:2]
                aspect_ratio = width / height
                brightness = np.mean(img_array)
                
                feature_data.append({
                    'class': class_name,
                    'mean_red': mean_red,
                    'mean_green': mean_green,
                    'mean_blue': mean_blue,
                    'texture': overall_texture,
                    'aspect_ratio': aspect_ratio,
                    'brightness': brightness
                })
                
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")
    
    return pd.DataFrame(feature_data)

def visualize_features(feature_df):
    if feature_df.empty:
        print("⚠️ No feature data to visualize")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('🔍 Three Key Features Analysis', fontsize=16, fontweight='bold')
    color_means = feature_df.groupby('class')[['mean_red', 'mean_green', 'mean_blue']].mean()
    color_means.plot(kind='bar', ax=axes[0,0], color=['red', 'green', 'blue'], alpha=0.7)
    axes[0,0].set_title('🎨 Average RGB Values by Class')
    axes[0,0].set_ylabel('RGB Value')
    axes[0,0].tick_params(axis='x', rotation=45)
    feature_df.boxplot(column='texture', by='class', ax=axes[0,1])
    axes[0,1].set_title('🔍 Texture Distribution by Class')
    axes[0,1].set_ylabel('Texture (Standard Deviation)')
    
    # 3. Shape Analysis - Aspect Ratio
    feature_df.boxplot(column='aspect_ratio', by='class', ax=axes[1,0])
    axes[1,0].set_title('📐 Aspect Ratio by Class')
    axes[1,0].set_ylabel('Width/Height Ratio')
    
    # 4. Brightness vs Texture scatter
    for class_name in feature_df['class'].unique():
        class_data = feature_df[feature_df['class'] == class_name]
        axes[1,1].scatter(class_data['brightness'], class_data['texture'], 
                         label=class_name, alpha=0.6)
    axes[1,1].set_xlabel('Brightness')
    axes[1,1].set_ylabel('Texture')
    axes[1,1].set_title('💡 Brightness vs Texture')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()

def interpret_features(feature_df):
    """Interpret the meaning of the 3 key features"""
    if feature_df.empty:
        return
        
    print("\n📖 FEATURE INTERPRETATION STORY")
    print("=" * 60)
    
    # Color Analysis
    print("🎨 COLOR SIGNATURES:")
    color_analysis = feature_df.groupby('class')[['mean_red', 'mean_green', 'mean_blue']].mean()
    for fruit in color_analysis.index:
        red, green, blue = color_analysis.loc[fruit]
        dominant_color = ['Red', 'Green', 'Blue'][np.argmax([red, green, blue])]
        print(f"  🍎 {fruit.title()}: {dominant_color}-dominant (R:{red:.1f}, G:{green:.1f}, B:{blue:.1f})")
    
    # Texture Analysis  
    print(f"\n🔍 TEXTURE PATTERNS:")
    texture_analysis = feature_df.groupby('class')['texture'].mean()
    for fruit in texture_analysis.index:
        texture_val = texture_analysis[fruit]
        texture_desc = "Highly textured" if texture_val > 40 else "Moderately textured" if texture_val > 25 else "Smooth"
        print(f"  🔍 {fruit.title()}: {texture_desc} surface ({texture_val:.1f})")
    
    # Shape Analysis
    print(f"\n📐 SHAPE CHARACTERISTICS:")
    shape_analysis = feature_df.groupby('class')['aspect_ratio'].mean()
    for fruit in shape_analysis.index:
        ratio = shape_analysis[fruit]
        shape_desc = "Wider than tall" if ratio > 1.1 else "Taller than wide" if ratio < 0.9 else "Square-like"
        print(f"  📏 {fruit.title()}: {shape_desc} proportions ({ratio:.2f})")


def compare_algorithms(X, y, class_names):
    """Compare different ML algorithms and return the best one"""
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42, probability=True),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"  🔄 Training {name}...")
        
        # Train and evaluate
        model.fit(X, y)
        train_pred = model.predict(X)
        train_acc = accuracy_score(y, train_pred)
        
        # Cross-validation for more reliable estimate
        cv_scores = cross_val_score(model, X, y, cv=5)
        
        results[name] = {
            'model': model,
            'train_accuracy': train_acc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"    📊 Train Accuracy: {train_acc:.4f}")
        print(f"    📊 CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Find best model based on cross-validation score
    best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
    best_model = results[best_model_name]['model']
    
    return best_model, results

def display_model_comparison(model_results):
    """Display model comparison results"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    model_names = list(model_results.keys())
    train_accs = [model_results[name]['train_accuracy'] for name in model_names]
    cv_means = [model_results[name]['cv_mean'] for name in model_names]
    cv_stds = [model_results[name]['cv_std'] for name in model_names]
    
    # Training accuracy comparison
    ax1.bar(model_names, train_accs, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
    ax1.set_title('🎯 Training Accuracy Comparison')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim(0, 1)
    
    # Cross-validation comparison
    ax2.bar(model_names, cv_means, yerr=cv_stds, capsize=5, alpha=0.7, 
            color=['skyblue', 'lightcoral', 'lightgreen'])
    ax2.set_title('📈 Cross-Validation Performance')
    ax2.set_ylabel('CV Score')
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.show()

def create_detailed_evaluation(y_true, y_pred, class_names):
    """Create detailed evaluation with confusion matrix"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('🔥 Confusion Matrix')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    # 2. Per-class accuracy
    class_accuracies = []
    for i, class_name in enumerate(class_names):
        class_correct = cm[i, i]
        class_total = np.sum(cm[i, :])
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        class_accuracies.append(class_accuracy)
    
    bars = axes[1].bar(class_names, class_accuracies, 
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    axes[1].set_title('📊 Per-Class Accuracy')
    axes[1].set_xlabel('Fruit Classes')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, class_accuracies):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print per-class performance
    print(f"\n📊 PER-CLASS PERFORMANCE:")
    for i, class_name in enumerate(class_names):
        print(f"  {class_name.title()}: {class_accuracies[i]:.2%}")

def analyze_prediction_confidence(model, X_test, y_true, class_names):
    """Analyze prediction confidence if model supports it"""
    
    if not hasattr(model, 'predict_proba'):
        return
    
    # Get prediction probabilities
    probabilities = model.predict_proba(X_test)
    max_probs = np.max(probabilities, axis=1)
    predictions = model.predict(X_test)
    
    # Analyze confidence by correctness
    correct_mask = predictions == y_true
    correct_confidences = max_probs[correct_mask]
    incorrect_confidences = max_probs[~correct_mask]
    
    print(f"\n🎯 PREDICTION CONFIDENCE ANALYSIS:")
    print(f"  Average confidence on correct predictions: {np.mean(correct_confidences):.2%}")
    print(f"  Average confidence on incorrect predictions: {np.mean(incorrect_confidences):.2%}")
    
    # Plot confidence distribution
    plt.figure(figsize=(10, 6))
    plt.hist(correct_confidences, alpha=0.7, label='Correct Predictions', bins=20, color='green')
    plt.hist(incorrect_confidences, alpha=0.7, label='Incorrect Predictions', bins=20, color='red')
    plt.xlabel('Prediction Confidence')
    plt.ylabel('Frequency')
    plt.title('🎯 Prediction Confidence Distribution')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Enhanced training with all features
    print("🚀 Starting Enhanced Model Training...")
    model, le = train_model(
        train_dir='../data/train',
        save_path='../models/fruit_classifier.pkl',
        compare_models=True,    # Set to False for faster training
        analyze_features=True   # Set to False to skip feature analysis
    )
    
    print("\n" + "="*60)
    
    # Enhanced evaluation
    print("📊 Starting Enhanced Model Evaluation...")
    accuracy, predictions = evaluate_model(
        test_dir='../data/test',
        model_path='../models/fruit_classifier.pkl',
        detailed=True  # Set to False for simple evaluation
    )
    
    print(f"\n🎉 TRAINING AND EVALUATION COMPLETE!")
    print(f"📊 Final Accuracy: {accuracy:.2%}")
    print(f"💾 Model saved to: models/fruit_classifier.pkl")
    print(f"🚀 Ready for deployment!")