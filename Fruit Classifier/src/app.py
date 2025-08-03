import streamlit as st
import requests
import os
import shutil
from PIL import Image
import tempfile
import zipfile
import time
import json

API_URL = "http://127.0.0.1:8000"

def check_api_status():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def save_uploaded_files_for_retraining(uploaded_files, labels):
    """Save uploaded files to training directory structure"""
    train_dir = "../data/train"  # Fixed path - removed ../
    
    saved_files = []
    saved_to_dirs = {}  # Track which directories we saved to
    
    for file, label in zip(uploaded_files, labels):
        if not label or not label.strip():
            print(f"Skipping file {file.name} - no label provided")
            continue
            
        # Clean the label - make it lowercase and replace spaces with underscores
        clean_label = label.strip().lower().replace(' ', '_').replace('-', '_')
        
        # Create directory for this class
        class_dir = os.path.join(train_dir, clean_label)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create filename with timestamp to avoid conflicts
        timestamp = str(int(time.time() * 1000))  # Use milliseconds for uniqueness
        file_extension = file.name.split('.')[-1] if '.' in file.name else 'jpg'
        filename = f"{clean_label}_{timestamp}.{file_extension}"
        filepath = os.path.join(class_dir, filename)
        
        # Save file
        try:
            with open(filepath, "wb") as f:
                f.write(file.getvalue())
            saved_files.append(filepath)
            
            # Track which directory we saved to
            if clean_label not in saved_to_dirs:
                saved_to_dirs[clean_label] = 0
            saved_to_dirs[clean_label] += 1
            
            print(f"✅ Saved {file.name} -> {filepath}")
            
        except Exception as e:
            print(f"❌ Failed to save {file.name}: {e}")
    
    return saved_files, saved_to_dirs

def get_training_status():
    """Get current training status from a local file"""
    status_file = "training_status.json"
    default_status = {
        "is_training": False,
        "progress": 0,
        "message": "Ready",
        "start_time": None,
        "end_time": None
    }
    
    try:
        if os.path.exists(status_file):
            with open(status_file, 'r') as f:
                return json.load(f)
    except:
        pass
    
    return default_status

def update_training_status(status_dict):
    """Update training status to local file"""
    status_file = "training_status.json"
    try:
        with open(status_file, 'w') as f:
            json.dump(status_dict, f)
    except:
        pass

st.title("Fruit Classifier")

# Check API status
api_online = check_api_status()
if api_online:
    st.success("🟢 API is online and ready!")
else:
    st.warning("🟡 API is offline - using local mode (retraining will work locally)")
    st.info("💡 To use API features, start the FastAPI server with: `uvicorn api:app --reload`")

st.markdown("---")

# Navigation tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📤 Upload & Retrain", "📊 Status"])

with tab1:
    st.header("Make a Prediction")
    st.write("Upload an image of a fruit to get a prediction!")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            if st.button("🔍 Predict", type="primary") and api_online:
                with st.spinner("Making prediction..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        response = requests.post(f"{API_URL}/predict", files=files)
                        
                        if response.status_code == 200:
                            result = response.json()
                            prediction = result.get('prediction', 'Unknown')
                            confidence = result.get('confidence', 0.0)
                            
                            st.success(f"🎉 **Prediction: {prediction.title()}**")
                            st.info(f"Confidence: {confidence:.1%}")
                            
                            if confidence < 0.7:
                                st.warning("⚠️ Low confidence. Consider uploading clearer images for retraining.")
                        else:
                            st.error(f"❌ Prediction failed: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

with tab2:
    st.header("📤 Upload Data & Retrain Model")
    st.write("Upload multiple images to improve the model performance.")
    
    # Check if currently training
    current_status = get_training_status()
    if current_status.get("is_training", False):
        st.warning("🔄 Model is currently being retrained. Please wait...")
        st.progress(current_status.get("progress", 0) / 100)
        st.info(f"Status: {current_status.get('message', 'Training...')}")
        
        if st.button("🔄 Refresh Status"):
            st.rerun()
    else:
        # Method selection
        method = st.radio("Upload Method:", ["Individual Images", "Zip File"])
        
        if method == "Individual Images":
            uploaded_files = st.file_uploader(
                "Choose multiple images...", 
                type=["jpg", "jpeg", "png"], 
                accept_multiple_files=True
            )
            
            if uploaded_files:
                st.success(f"📁 {len(uploaded_files)} files uploaded!")
                
                # Show preview
                if len(uploaded_files) <= 4:
                    cols = st.columns(len(uploaded_files))
                    for i, file in enumerate(uploaded_files):
                        with cols[i]:
                            image = Image.open(file)
                            st.image(image, caption=file.name, use_container_width=True)
                else:
                    cols = st.columns(4)
                    for i in range(4):
                        with cols[i]:
                            image = Image.open(uploaded_files[i])
                            st.image(image, caption=uploaded_files[i].name, use_container_width=True)
                    st.info(f"+ {len(uploaded_files) - 4} more images...")
                
                # Label assignment
                st.subheader("🏷️ Assign Labels")
                st.write("Assign a fruit class to each uploaded image:")
                
                # Get existing classes from directory structure
                existing_classes = []
                train_dir = "../data/train"
                if os.path.exists(train_dir):
                    existing_classes = [d for d in os.listdir(train_dir) 
                                     if os.path.isdir(os.path.join(train_dir, d))]
                
                if not existing_classes:
                    existing_classes = ["apple", "banana", "orange",]
                
                labels = []
                new_classes = {}
                
                for i, file in enumerate(uploaded_files):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.write(f"**{file.name}**")
                    with col2:
                        label_option = st.selectbox(
                            "Class:",
                            existing_classes + ["➕ New Class"],
                            key=f"label_{i}"
                        )
                        
                        if label_option == "➕ New Class":
                            new_class = st.text_input(
                                f"Enter new class name:", 
                                key=f"new_class_{i}",
                                placeholder="e.g., mango, pineapple"
                            )
                            if new_class.strip():
                                clean_new_class = new_class.strip().lower().replace(' ', '_')
                                labels.append(clean_new_class)
                                new_classes[i] = clean_new_class
                            else:
                                labels.append("")
                        else:
                            labels.append(label_option)
                
                st.markdown("---")
                
                # Show summary
                if labels:
                    label_counts = {}
                    for label in labels:
                        if label.strip():
                            label_counts[label] = label_counts.get(label, 0) + 1
                    
                    st.subheader("📊 Upload Summary")
                    for label, count in label_counts.items():
                        st.write(f"• **{label.title()}**: {count} images")
                
                # Retraining options
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save to Training Data", type="secondary"):
                        if all(label.strip() for label in labels):
                            try:
                                saved_files, saved_to_dirs = save_uploaded_files_for_retraining(uploaded_files, labels)
                                st.success(f"✅ Saved {len(saved_files)} files to training data!")
                                st.info("Files saved locally. You can now retrain the model.")
                                
                                # Show where files were saved
                                for class_name, count in saved_to_dirs.items():
                                    st.write(f"• {count} images saved to: `data/train/{class_name}/`")
                                        
                            except Exception as e:
                                st.error(f"❌ Error saving files: {str(e)}")
                        else:
                            st.error("❌ Please assign labels to all images before saving.")
                
                with col2:
                    if st.button("🚀 Save & Retrain Model", type="primary"):
                        if all(label.strip() for label in labels):
                            try:
                                # Update status to training
                                update_training_status({
                                    "is_training": True,
                                    "progress": 0,
                                    "message": "Preparing to train...",
                                    "start_time": time.time()
                                })
                                
                                # Save files first
                                saved_files, saved_to_dirs = save_uploaded_files_for_retraining(uploaded_files, labels)
                                
                                update_training_status({
                                    "is_training": True,
                                    "progress": 10,
                                    "message": f"Saved {len(saved_files)} files, starting training...",
                                    "start_time": time.time()
                                })
                                
                                st.success(f"✅ Saved {len(saved_files)} files!")
                                
                                # Show where files were saved
                                for class_name, count in saved_to_dirs.items():
                                    st.write(f"• {count} images saved to: `data/train/{class_name}/`")
                                
                                st.info("🚀 Starting retraining process...")
                                
                                # Start local retraining (skip API since it's not working)
                                import subprocess
                                try:
                                    # Use absolute path to ensure we're in the right directory
                                    current_dir = os.getcwd()
                                    retrain_script = os.path.join(current_dir, "retrain.py")
                                    
                                    if os.path.exists(retrain_script):
                                        subprocess.Popen([
                                            "python", retrain_script
                                        ], cwd=current_dir)
                                        st.success("✅ Local retraining started!")
                                    else:
                                        st.error("❌ retrain.py script not found!")
                                        update_training_status({
                                            "is_training": False,
                                            "progress": 0,
                                            "message": "Error: retrain.py not found"
                                        })
                                        
                                except Exception as e:
                                    st.error(f"❌ Failed to start retraining: {str(e)}")
                                    update_training_status({
                                        "is_training": False,
                                        "progress": 0,
                                        "message": f"Failed: {str(e)}"
                                    })
                                
                                # Auto-refresh to show progress
                                time.sleep(2)
                                st.rerun()
                                
                            except Exception as e:
                                st.error(f"❌ Error during retraining setup: {str(e)}")
                                update_training_status({
                                    "is_training": False,
                                    "progress": 0,
                                    "message": f"Error: {str(e)}"
                                })
                        else:
                            st.error("❌ Please assign labels to all images before retraining.")
        
        else:  # Zip File method
            st.subheader("📦 Upload Zip File")
            st.write("Upload a zip file containing folders with fruit images (folder name = class name)")
            
            uploaded_zip = st.file_uploader("Choose a zip file...", type=['zip'])
            
            if uploaded_zip:
                try:
                    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
                        file_list = zip_ref.namelist()
                        image_files = [f for f in file_list if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        
                        st.write(f"📦 Found {len(image_files)} images in the archive")
                        
                        # Show structure
                        folders = set()
                        for file in image_files:
                            if '/' in file:
                                folder = file.split('/')[0]
                                folders.add(folder)
                        
                        st.write(f"**Detected classes:** {', '.join(sorted(folders))}")
                        
                        if st.button("📂 Extract and Save for Training"):
                            # Extract to training directory
                            train_dir = "../data/train"
                            
                            with tempfile.TemporaryDirectory() as temp_dir:
                                zip_ref.extractall(temp_dir)
                                
                                extracted_count = 0
                                # Copy to training directory
                                for folder in folders:
                                    src_folder = os.path.join(temp_dir, folder)
                                    dst_folder = os.path.join(train_dir, folder.lower().replace(' ', '_'))
                                    
                                    if os.path.exists(src_folder):
                                        os.makedirs(dst_folder, exist_ok=True)
                                        for file in os.listdir(src_folder):
                                            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                                shutil.copy2(
                                                    os.path.join(src_folder, file),
                                                    os.path.join(dst_folder, file)
                                                )
                                                extracted_count += 1
                            
                            st.success(f"✅ Extracted {extracted_count} images to training directory!")
                            
                            if st.button("🚀 Start Retraining") and not current_status.get("is_training", False):
                                update_training_status({
                                    "is_training": True,
                                    "progress": 0,
                                    "message": "Starting retraining with extracted data...",
                                    "start_time": time.time()
                                })
                                st.info("🔄 Starting retraining with extracted data...")
                                st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error processing zip file: {str(e)}")

with tab3:
    st.header("📊 Model Status")
    
    # Get current training status
    current_status = get_training_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🤖 Model Information")
        
        if api_online:
            try:
                response = requests.get(f"{API_URL}/status")
                if response.status_code == 200:
                    status_data = response.json()
                    st.write(f"**Model Loaded:** {'✅ Yes' if status_data.get('model_loaded') else '❌ No'}")
                    st.write(f"**Classes:** {', '.join(status_data.get('model_classes', []))}")
                    st.write(f"**Total Predictions:** {status_data.get('total_predictions', 0)}")
                    
                    uptime = status_data.get('uptime_seconds', 0)
                    hours = uptime // 3600
                    minutes = (uptime % 3600) // 60
                    st.write(f"**Uptime:** {hours}h {minutes}m")
                else:
                    st.error("❌ Could not get API status")
            except Exception as e:
                st.error(f"❌ Error getting API status: {str(e)}")
        else:
            st.warning("🔴 API offline - showing local info")
            
            # Show local model info
            model_path = "../models/fruit_classifier.pkl"
            if os.path.exists(model_path):
                st.write("**Model File:** ✅ Found locally")
                mod_time = os.path.getmtime(model_path)
                st.write(f"**Last Modified:** {time.ctime(mod_time)}")
            else:
                st.write("**Model File:** ❌ Not found")
    
    with col2:
        st.subheader("🔄 Training Status")
        
        is_training = current_status.get('is_training', False)
        progress = current_status.get('progress', 0)
        message = current_status.get('message', 'Ready')
        
        if is_training:
            st.warning("🔄 Training in Progress")
            st.progress(progress / 100)
            st.write(f"**Status:** {message}")
            
            # Auto-refresh during training
            if st.button("🔄 Refresh"):
                st.rerun()
                
            # Auto-refresh every 5 seconds during training
            time.sleep(5)
            st.rerun()
        else:
            st.success("✅ Ready for Training")
            st.write(f"**Last Status:** {message}")
            
            if current_status.get('end_time'):
                end_time = time.ctime(current_status['end_time'])
                st.write(f"**Last Completed:** {end_time}")
    
    # Training data info
    st.subheader("📁 Training Data")
    train_dir = "../data/train"
    
    if os.path.exists(train_dir):
        class_info = {}
        total_images = 0
        
        for class_name in os.listdir(train_dir):
            class_path = os.path.join(train_dir, class_name)
            if os.path.isdir(class_path):
                image_count = len([f for f in os.listdir(class_path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_info[class_name] = image_count
                total_images += image_count
        
        st.write(f"**Total Images:** {total_images}")
        
        if class_info:
            st.write("**Images per Class:**")
            for class_name, count in sorted(class_info.items()):
                status_icon = "⚠️" if count < 10 else "✅"
                st.write(f"{status_icon} **{class_name.title()}**: {count} images")
        else:
            st.info("No training data found")
    else:
        st.warning("⚠️ No training data directory found")
        if st.button("📁 Create Training Directory"):
            os.makedirs(train_dir, exist_ok=True)
            st.success("✅ Training directory created!")
            st.rerun()

# Auto-refresh when training
if get_training_status().get("is_training", False):
    time.sleep(3)
    st.rerun() 
