import streamlit as st
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import torch

# Load environment variables from .env file
load_dotenv()

# Define Streamlit app
st.title("Youtube Summarizer Demo")
st.write("This demo uses a chat model to answer questions about a YouTube video.")

# Display video selection and question inputs
YT_video = st.text_input("YouTube Video ID (e.g., 1egAKCKPKCk?t=2743)")
query = st.text_input("Enter your question")

# Process user inputs
if st.button("Get Answer"):
    if not YT_video:
        st.warning("Please enter a YouTube video ID.")
    elif not query:
        st.warning("Please enter a question.")
    else:
        # Create a YouTube Data API client
        api_key = os.environ["YOUTUBE_API_KEY"]
        youtube = build("youtube", "v3", developerKey=api_key)

        # Retrieve video caption tracks
        caption_response = youtube.captions().list(
            part="id",
            videoId=YT_video
        ).execute()

                
        # Check if "items" key is present in caption_response
        if "items" in caption_response:
            caption_tracks = caption_response["items"]
            
            # Retrieve captions in different languages
            captions = []
            for track in caption_tracks:
                caption = youtube.captions().download(
                    id=track["id"],
                    tfmt="srt"
                ).execute()
                captions.append(caption)

            # Combine captions into a single text
            captions_text = "\n".join(captions)

            # Tokenize the captions and question
            tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
            encoded_inputs = tokenizer.encode_plus(
                captions_text,
                query,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )

            # Load the question-answering model
            model = AutoModelForQuestionAnswering.from_pretrained("bert-base-multilingual-cased")

            # Perform question-answering
            with torch.no_grad():
                outputs = model(**encoded_inputs)
                start_logits = outputs.start_logits
                end_logits = outputs.end_logits
                start_index = torch.argmax(start_logits)
                end_index = torch.argmax(end_logits)
                answer = tokenizer.convert_tokens_to_string(
                    tokenizer.convert_ids_to_tokens(
                        encoded_inputs["input_ids"][0][start_index : end_index + 1]
                    )
                )

            # Display the answer
            st.write("Answer:", answer)
        else:
            st.warning("No captions found for the provided YouTube video ID.")