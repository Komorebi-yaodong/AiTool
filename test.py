from huggingface_hub import InferenceClient
import streamlit as st

show_app = st.container()


def free_text_to_image(text):
    client = InferenceClient(model="https://api-inference.huggingface.co/models/dataautogpt3/OpenDalleV1.1")
    image = client.text_to_image(text)
    return image


def main(prompt):
    show_app.write("**You:** " + prompt)
    image = free_text_to_image(prompt)
    show_app.image(image,use_column_width=True)


prompt = st.chat_input("Send your prompt")
if prompt:
    main(prompt)