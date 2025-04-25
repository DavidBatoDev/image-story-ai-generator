# AI Story Generator

AI Story Generator is a Streamlit-based web application that converts images into captivating stories. Using advanced AI models, the app extracts captions from images, generates stories based on the captions, and converts the stories into speech.

## Features

- **Image Captioning**: Extract meaningful captions from uploaded images using `Salesforce/blip-image-captioning-base`.
- **Story Generation**: Generate creative stories based on image captions using `EleutherAI/gpt-neo-1.3B`.
- **Text-to-Speech**: Convert generated stories into speech using `gTTS`.

## Prerequisites

1. Python 3.8 or later
2. Install the required Python libraries by running:
   ```bash
   pip install -r requirements.txt
Download and install the models locally:

Image Captioning Model: Salesforce/blip-image-captioning-base
Story Generator Model: EleutherAI/gpt-neo-1.3B
Use the transformers library to download these models locally.

```python
from transformers import AutoModel, AutoTokenizer

# Download Image Captioning Model
AutoModel.from_pretrained("Salesforce/blip-image-captioning-base")
AutoTokenizer.from_pretrained("Salesforce/blip-image-captioning-base")

# Download Story Generator Model
AutoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
Set up your .env file for environment variables (if required).
```

### Installation
1. Clone the repository:
```bash
git clone https://github.com/you/DavidBatoDev/image-story-ai-generator.git
cd ai-story-generator
```
2. Set up a virtual environment (recommended):
```
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
streamlit run app.py
```

2. Open the application in your browser:
```arduino
http://localhost:8501
```
Upload an image, and let the AI generate a story and its audio version for you!

Project Structure
```bash
IMAGE-CAPTIONING-API/
│
├── myenv/                   # Virtual environment folder
├── test-images/             # Test images for the application
├── .env                     # Environment variable file
├── .gitignore               # Git ignore file
├── app.py                   # Main Streamlit application
├── output_audio_store/      # Directory to store audio outputs
├── requirements.txt         # Python dependencies
```
Dependencies
streamlit: For building the web application
transformers: For using pre-trained models from Hugging Face
gTTS: For text-to-speech conversion
python-dotenv: For managing environment variables
Install all dependencies with:

```bash
pip install -r requirements.txt
```

Acknowledgments
- Hugging Face Transformers
- Salesforce/blip-image-captioning-base
- EleutherAI/gpt-neo-1.3B
- Google Text-to-Speech (gTTS)
- Streamlit

