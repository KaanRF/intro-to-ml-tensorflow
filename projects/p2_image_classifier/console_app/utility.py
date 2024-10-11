import numpy as np
import tensorflow as tf
from PIL import Image

# Process the image to the correct format for the model
def process_image(image_path):
    img = Image.open(image_path)
    
    # Resize the image and normalize it
    img = img.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize to 0-1 range
    
    return img

# Function to make predictions
def predict(image_path, model, top_k):
    # Preprocess the image
    img = process_image(image_path)
    
    # Add batch dimension for the model input
    img_batch = np.expand_dims(img, axis=0)
    
    # Make predictions
    preds = model.predict(img_batch)[0]
    
    # Get the top K predictions
    top_k_probs, top_k_classes = tf.math.top_k(preds, k=top_k)
    
    # Convert to numpy arrays
    top_k_probs = top_k_probs.numpy()
    top_k_classes = top_k_classes.numpy()
    
    return top_k_probs, top_k_classes