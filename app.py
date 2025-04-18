# # app.p


# streamlit run app_pose.py

import streamlit as st
import os
import torch
from torchvision import transforms
from PIL import Image
import boto3

# AWS S3 Setup
bucket_name = "mlops-44448888"
local_model_path = "human_pose_classification"
s3_model_prefix = "ml-models/human_pose_classification/"

s3 = boto3.client('s3')

def download_model(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for file_obj in result['Contents']:
                s3_key = file_obj['Key']
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file)

# Title
st.title("Human Pose Classification Deployment 🚀")

# Model Download Button
button_download = st.button("Download Model")
if button_download:
    with st.spinner("Downloading model from S3... Please wait!"):
        download_model(local_model_path, s3_model_prefix)
    st.success("Model downloaded successfully!")

# Image Upload
uploaded_file = st.file_uploader("Upload an Image for Pose Classification", type=["jpg", "png", "jpeg"])

# Prediction Button
predict_button = st.button("Predict Pose")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Model
@st.cache_resource
def load_model():
    model = torch.load(os.path.join(local_model_path, 'model.pth'), map_location=device)
    model.eval()
    return model

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Predict
if predict_button and uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    with st.spinner("Predicting..."):
        model = load_model()
        output = model(input_tensor)
        predicted_class = output.argmax(dim=1).item()
        
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write(f"### Predicted Pose Class: {predicted_class}")

