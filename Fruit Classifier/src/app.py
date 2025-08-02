import streamlit as st
import requests
import os
import shutil
from PIL import Image
import tempfile
import zipfile
import time

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
    train_dir = "../data/train"
    
    # Create directories if they don't exist
    for label in set(labels):
        os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    
    saved_files = []
    for file, label in zip(uploaded_files, labels):
        # Create filename with timestamp to avoid conflicts
        timestamp = str(int(time.time()))
        filename = f"{label}_{timestamp}_{file.name}"
        filepath = os.path.join(train_dir, label, filename)
        
        # Save file
        with open(filepath, "wb") as f:
            f.write(file.getvalue())
        saved_files.append(filepath)
    
    return saved_files

st.title("🍎 Fruit Classifier Dashboard")

# Check API status
api_online = check_api_status()
if api_online:
    st.success("🟢 API is online and ready!")
else:
    st.error("🔴 API is offline. Please start the FastAPI server with: `uvicorn api:app --reload`")

st.markdown("---")

# Navigation tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📤 Upload & Retrain", "📊 Status"])

with tab1:
    st.header("🔮 Make a Prediction")
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
            
            # Get existing classes from API
            existing_classes = ["apple", "banana", "orange", "grape", "strawberry"]  # Default classes
            try:
                if api_online:
                    response = requests.get(f"{API_URL}/status")
                    if response.status_code == 200:
                        api_status = response.json()
                        existing_classes = api_status.get('model_classes', existing_classes)
            except:
                pass
            
            labels = []
            for i, file in enumerate(uploaded_files):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{file.name}**")
                with col2:
                    label = st.selectbox(
                        "Class:",
                        existing_classes + ["New Class"],
                        key=f"label_{i}"
                    )
                    if label == "New Class":
                        label = st.text_input(f"Enter new class:", key=f"new_class_{i}")
                    labels.append(label)
            
            st.markdown("---")
            
            # Retraining options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("💾 Save to Training Data", type="secondary"):
                    try:
                        saved_files = save_uploaded_files_for_retraining(uploaded_files, labels)
                        st.success(f"✅ Saved {len(saved_files)} files to training data!")
                        st.info("Files saved locally. You can now retrain the model.")
                    except Exception as e:
                        st.error(f"❌ Error saving files: {str(e)}")
            
            with col2:
                if st.button("🚀 Retrain Model", type="primary") and api_online:
                    if all(label.strip() for label in labels):  # Check all labels are filled
                        with st.spinner("🔄 Retraining model... This may take a few minutes."):
                            try:
                                # First save files locally
                                saved_files = save_uploaded_files_for_retraining(uploaded_files, labels)
                                
                                # Trigger retraining via API
                                files_for_api = []
                                for file, label in zip(uploaded_files, labels):
                                    # Rename file to include label for API
                                    new_filename = f"{label}_{file.name}"
                                    files_for_api.append(("files", (new_filename, file.getvalue(), file.type)))
                                
                                response = requests.post(f"{API_URL}/retrain", files=files_for_api)
                                
                                if response.status_code == 200:
                                    st.success("✅ Retraining started successfully!")
                                    st.info("Model is being retrained in the background. Check the Status tab for progress.")
                                    
                                    # Show training status
                                    progress_placeholder = st.empty()
                                    status_placeholder = st.empty()
                                    
                                    # Monitor training progress
                                    for _ in range(30):  # Check for up to 30 seconds
                                        try:
                                            status_response = requests.get(f"{API_URL}/training-status")
                                            if status_response.status_code == 200:
                                                training_status = status_response.json()
                                                
                                                progress = training_status.get('progress', 0)
                                                message = training_status.get('message', '')
                                                is_training = training_status.get('is_training', False)
                                                
                                                progress_placeholder.progress(progress / 100)
                                                status_placeholder.text(f"Status: {message}")
                                                
                                                if not is_training and progress == 100:
                                                    st.success("🎉 Model retraining completed!")
                                                    break
                                                elif not is_training:
                                                    st.error("❌ Training stopped unexpectedly")
                                                    break
                                        except:
                                            pass
                                        
                                        time.sleep(2)
                                else:
                                    st.error(f"❌ Retraining failed: {response.text}")
                            except Exception as e:
                                st.error(f"❌ Error during retraining: {str(e)}")
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
                    
                    st.write(f"**Detected classes:** {', '.join(folders)}")
                    
                    if st.button("📂 Extract and Save for Training"):
                        # Extract to training directory
                        train_dir = "data/train"
                        
                        with tempfile.TemporaryDirectory() as temp_dir:
                            zip_ref.extractall(temp_dir)
                            
                            # Copy to training directory
                            for folder in folders:
                                src_folder = os.path.join(temp_dir, folder)
                                dst_folder = os.path.join(train_dir, folder)
                                
                                if os.path.exists(src_folder):
                                    os.makedirs(dst_folder, exist_ok=True)
                                    for file in os.listdir(src_folder):
                                        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                                            shutil.copy2(
                                                os.path.join(src_folder, file),
                                                os.path.join(dst_folder, file)
                                            )
                        
                        st.success(f"✅ Extracted and saved images to training directory!")
                        
                        if st.button("🚀 Start Retraining") and api_online:
                            st.info("🔄 Starting retraining with extracted data...")
                            # You can add API call here to trigger retraining
            
            except Exception as e:
                st.error(f"❌ Error processing zip file: {str(e)}")

with tab3:
    st.header("📊 Model Status")
    
    if api_online:
        try:
            # Get API status
            response = requests.get(f"{API_URL}/status")
            if response.status_code == 200:
                status_data = response.json()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🤖 Model Information")
                    st.write(f"**Model Loaded:** {'✅ Yes' if status_data.get('model_loaded') else '❌ No'}")
                    st.write(f"**Classes:** {', '.join(status_data.get('model_classes', []))}")
                    st.write(f"**Total Predictions:** {status_data.get('total_predictions', 0)}")
                    
                    uptime = status_data.get('uptime_seconds', 0)
                    hours = uptime // 3600
                    minutes = (uptime % 3600) // 60
                    st.write(f"**Uptime:** {hours}h {minutes}m")
                
                with col2:
                    st.subheader("🔄 Training Status")
                    
                    # Get training status
                    training_response = requests.get(f"{API_URL}/training-status")
                    if training_response.status_code == 200:
                        training_status = training_response.json()
                        
                        is_training = training_status.get('is_training', False)
                        progress = training_status.get('progress', 0)
                        message = training_status.get('message', 'Ready')
                        
                        if is_training:
                            st.warning(f"🔄 Training in Progress")
                            st.progress(progress / 100)
                            st.write(f"**Status:** {message}")
                        else:
                            st.success("✅ Ready for Training")
                            st.write(f"**Last Status:** {message}")
                    else:
                        st.error("❌ Could not get training status")
            else:
                st.error("❌ Could not get API status")
        except Exception as e:
            st.error(f"❌ Error getting status: {str(e)}")
    else:
        st.error("🔴 API is offline")
    
    # Training data info
    st.subheader("📁 Training Data")
    train_dir = "data/train"
    
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
        st.write("**Images per Class:**")
        for class_name, count in class_info.items():
            st.write(f"- {class_name}: {count} images")
    else:
        st.warning("⚠️ No training data directory found")
        if st.button("📁 Create Training Directory"):
            os.makedirs(train_dir, exist_ok=True)
            st.success("✅ Training directory created!")

