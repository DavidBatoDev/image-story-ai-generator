import streamlit as st
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import os

# Load environment variables
load_dotenv(find_dotenv())

# Image-to-Text Function
def img2text(url):
    try:
        # Initialize the image-to-text pipeline
        image_to_text = pipeline(
            'image-to-text',
            model='Salesforce/blip-image-captioning-base',
        )

        # Generate text from image
        text = image_to_text(url)[0]['generated_text']
        return text
    except Exception as e:
        st.error(f"Error during image-to-text processing: {str(e)}")
        return None

# Load Story Generator (GPT-Neo-1.3B)
def load_story_generator():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model.config.pad_token_id = model.config.eos_token_id  # Set pad_token_id
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

# Generate Story from Prompt
def generate_story(prompt, generator):
    try:
        # Generate story from the prompt
        story = generator(prompt, max_length=200, num_return_sequences=1, truncation=True)
        return story[0]["generated_text"]
    except Exception as e:
        st.error(f"Error during story generation: {str(e)}")
        return None

# Text-to-Speech using gTTS
def text_to_speech(text, description=None):
    try:
        # Create TTS from text
        tts = gTTS(text, lang='en')
        output_file = f"output_audio_{description or 'story'}.mp3"

        # Save the audio file
        tts.save(output_file)
        return output_file
    except Exception as e:
        st.error(f"Error during text-to-speech conversion: {str(e)}")
        return None

# Streamlit App
st.set_page_config(page_title="AI Story Generator", page_icon="📖", layout="wide")
st.title("AI Story Generator")
st.markdown("Generate creative stories from images using advanced AI technology.")

# Sidebar
st.sidebar.header("Upload Image")
image_file = st.sidebar.file_uploader("Upload an image file (JPEG/PNG):", type=["jpg", "png"])

if image_file is not None:
    st.image(image_file, caption="Uploaded Image", use_container_width=True)
    temp_file_path = f"temp_{image_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(image_file.getbuffer())

    # Step 1: Convert Image to Text
    with st.spinner("Extracting text from the image..."):
        image_caption = img2text(temp_file_path)

    if image_caption:
        st.subheader("Generated Caption:")
        st.success(image_caption)

        # Step 2: Generate Story
        with st.spinner("Generating story based on the caption..."):
            story_generator = load_story_generator()
            story = generate_story(image_caption, story_generator)

        if story:
            st.subheader("Generated Story:")
            st.text_area("Story", story, height=200)

            # Step 3: Convert Story to Speech
            with st.spinner("Converting story to audio..."):
                audio_file = text_to_speech(story, description="generated_story")

            if audio_file:
                st.subheader("Audio Output:")
                st.audio(audio_file, format="audio/mp3")
                st.success(f"Audio saved as: {audio_file}")

    # Clean up temporary files
    os.remove(temp_file_path)
else:
    st.info("Please upload an image to get started.")
