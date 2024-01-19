from huggingface_hub import InferenceClient
import streamlit as st

show_app = st.container()


def text_to_image(text):
    client = InferenceClient()
    image = client.text_to_image(text)
    return image


def main(prompt):
    show_app.write("**You:** " + prompt)
    image = text_to_image(prompt)
    show_app.image(image,use_column_width=True)


prompt = st.chat_input("Send your prompt")
if prompt:
    main(prompt)