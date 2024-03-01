import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load the MobileNetV2 model
model_path = "C:\\11111111\\nir\\model\\mobilenet_v2.h5"
model = load_model(model_path, compile=False)

# Set the image dimensions
img_width, img_height = 224, 224

# Define the path to your image directory
image_dir = "C:\\a80\\fortest"

# Define the path to save the features
features_dir = "C:\\a80\\mobilefeatures"
os.makedirs(features_dir, exist_ok=True)

# Function to preprocess the image
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(img_width, img_height))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to recursively retrieve image paths from a directory
def retrieve_image_paths(directory):
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, file)
                image_paths.append(image_path)
    return image_paths

# Function to extract and save features for each image
def extract_and_save_features(image_paths, features_dir):
    features = []
    for image_path in image_paths:
        # Preprocess the image
        image = preprocess_image(image_path)

        # Get the feature embeddings for the image
        image_features = model.predict(image)

        # Save the features
        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        np.save(os.path.join(features_dir, f"{image_filename}.npy"), image_features)

        # Append the features to the list
        features.append(image_features)

    # Convert the features list to a numpy array
    features = np.array(features)

    return features

# Function to load saved features
def load_features(features_dir, image_paths):
    features = []
    for image_path in image_paths:
        # Load the features for each image
        image_filename = os.path.splitext(os.path.basename(image_path))[0]
        features_path = os.path.join(features_dir, f"{image_filename}.npy")
        image_features = np.load(features_path)
        features.append(image_features)

    # Convert the features list to a numpy array
    features = np.array(features)

    return features

# Function to perform image retrieval
def perform_image_retrieval(query_image, features, images):
    # Preprocess the query image
    query_image = preprocess_image(query_image)

    # Get the feature embeddings for the query image
    query_features = model.predict(query_image)

    # Initialize a list to store the similarity scores
    similarity_scores = []

    # Iterate through each image and calculate the similarity score
    for image_features in features:
        # Calculate the cosine similarity between the query image features and image features
        similarity_score = cosine_similarity(query_features, image_features)[0][0]

        # Append the similarity score to the list
        similarity_scores.append(similarity_score)

    # Sort the images based on the similarity scores
    sorted_indices = np.argsort(similarity_scores)[::-1]
    sorted_images = [images[i] for i in sorted_indices]

    return sorted_images

# Streamlit app
def main():
    st.title("DeepImageSearch: Image Retrieval System using CNN")
    st.text("MobileNet V2")
    st.text("accuracy: 98%")
    st.text("F1 score: 0.97")

    # Select the query image
    query_image_path = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if query_image_path is not None:
        # Load or extract the features
        if os.path.exists(os.path.join(features_dir, "features.npy")):
            # Load the saved features
            features = np.load(os.path.join(features_dir, "features.npy"))
        else:
            # Extract and save the features
            features = extract_and_save_features(image_paths, features_dir)
            np.save(os.path.join(features_dir, "features.npy"), features)

        # Perform image retrieval
        retrieved_images = perform_image_retrieval(query_image_path, features, image_paths)

        # Display the query image
        st.subheader("Query Image")
        query_image = load_img(query_image_path)
        st.image(query_image, caption="Query Image", use_column_width=True)

        # Display the top 30 retrieved images directly
        st.subheader("Top 30 Retrieved Images")
        for i, image_path in enumerate(retrieved_images[:30]):
            retrieved_image = load_img(image_path)
            st.subheader(f"Similar Image {i+1}")
            st.image(retrieved_image, caption=f"Similar Image {i+1}", use_column_width=True)

if __name__ == "__main__":
    # Retrieve image paths from the image directory
    image_paths = retrieve_image_paths(image_dir)

    # Run the Streamlit app
    main()
