from dotenv import find_dotenv, load_dotenv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from gtts import gTTS
import torch
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
        print("Generated Text:", text)
        return text
    except Exception as e:
        print("Error during image-to-text processing:", str(e))
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
        print("Generated Story:", story[0]["generated_text"])
        return story[0]["generated_text"]
    except Exception as e:
        print("Error during story generation:", str(e))
        return None

# Text-to-Speech using gTTS
def text_to_speech(text, description=None):
    try:
        # Create TTS from text
        tts = gTTS(text, lang='en')
        output_file = f"output_audio_{description or 'story'}.mp3"

        # Save the audio file
        tts.save(output_file)
        print(f"Audio file saved as: {output_file}")
        return output_file
    except Exception as e:
        print("Error during text-to-speech conversion:", str(e))
        return None

# Main Functionality
if __name__ == "__main__":
    # Step 1: Convert Image to Text
    image_caption = img2text('test-images/mock_interview.jpg')

    if image_caption:
        # Step 2: Generate Story from Image Caption
        print("\nGenerating story based on the image caption...")
        story_generator = load_story_generator()
        story = generate_story(image_caption, story_generator)

        if story:
            print("\nFinal Story Output:\n", story)

            # Step 3: Convert Generated Story to Speech
            print("\nConverting story to speech...")
            audio_file = text_to_speech(story)

            if audio_file:
                print("\nText-to-Speech conversion complete. Audio saved at:", audio_file)
