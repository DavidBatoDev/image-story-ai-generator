from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
import os

# Load environment variables
load_dotenv(find_dotenv())

# Access token from .env
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Define the function
def img2text(url):
    try:
        # Initialize pipeline
        image_to_text = pipeline(
            'image-to-text',
            model='Salesforce/blip-image-captioning-base',
        )

        # Process the image
        text = image_to_text(url)
        print("Generated Text:", text)
        return text
    except Exception as e:
        print("Error during image-to-text processing:", str(e))
        return None

# Main block
if __name__ == "__main__":
    img2text('test-images/mock_interview.jpg')
