import os
import base64
from google.cloud import vision
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_text_from_image(image_data):
    """
    Extract text from an image using Google Cloud Vision API
    
    Args:
        image_data (str): Base64 encoded image data
        
    Returns:
        str: Extracted text from the image
    """
    try:
        # Check if GOOGLE_APPLICATION_CREDENTIALS is set
        if not os.getenv('GOOGLE_APPLICATION_CREDENTIALS'):
            print("Warning: GOOGLE_APPLICATION_CREDENTIALS not found in environment variables")
            return "Error: Cloud Vision API credentials not configured"
        
        # Initialize Vision client
        client = vision.ImageAnnotatorClient()
        
        # Decode base64 image data
        if image_data.startswith('data:image'):
            # Handle data URI format
            image_data = image_data.split(',')[1]
        
        image_content = base64.b64decode(image_data)
        image = vision.Image(content=image_content)
        
        # Perform text detection
        response = client.text_detection(image=image)
        texts = response.text_annotations
        
        # Extract full text from the response
        if texts:
            return texts[0].description
        else:
            return ""
            
    except Exception as e:
        print(f"Error extracting text from image: {e}")
        return f"Error processing image: {str(e)}"