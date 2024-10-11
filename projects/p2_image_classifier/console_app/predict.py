import argparse
import tensorflow as tf
import tensorflow_hub as hub

import json
import numpy as np
from PIL import Image
from utility import process_image, predict


# Define the main function
def main():
    
    parser = argparse.ArgumentParser(description='Predict the class of an image using a trained model.')    
    parser.add_argument('image_path', type=str, help='Path to the image file.')
    parser.add_argument('model_path', type=str, help='Path to the trained model file.')

    # Optional arguments
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names.')
    
    args = parser.parse_args()

    print(f"Loading image: {args.image_path}")
    print(f"Loading model: {args.model_path}")
        
    model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    print("Model loaded successfully.")
    
    # Process the image and predict
    image_path = args.image_path
    top_k = args.top_k
    
    # Get top K predictions
    probs, classes = predict(image_path, model, top_k)
    
    # If a category names file is provided, map class labels to names
    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        class_labels = [class_names[str(cls)] for cls in classes]
    else:
        class_labels = classes
        
    print(f"Top {top_k} Predictions for image '{image_path}':")
    for i in range(top_k):
        print(f"{i + 1}: {class_labels[i]} with probability {probs[i]:.4f}")

if __name__ == "__main__":
    main()