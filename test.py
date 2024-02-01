import streamlit as st
from io import BytesIO
import requests
import base64
import time
import base64


show_app = st.container()


def query_vispunk(prompt):
    def request_generate(prompt):
        url = "https://motion-api.vispunk.com/v1/generate/generate_image"
        headers = {"Content-Type": "application/json"}
        data = {"prompt": prompt}
        try: 
            response = requests.post(url, headers=headers, json=data)
            return True,response.json()["task_id"]
        except Exception as e:
            st.error(f"Error: {e}")
            return False,None


    def request_image(task_id):
        url = "https://motion-api.vispunk.com/v1/generate/check_image_task"
        headers = {"Content-Type": "application/json"}
        data = {"task_id": task_id}
        try: 
            response = requests.post(url, headers=headers, json=data)
            return True,response.json()["images"][0]
        except Exception as e:
            return False,e
        
    flag_generate,task_id = request_generate(prompt)
    if flag_generate:
        while True:
            flag_wait,image_src = request_image(task_id)
            if not flag_wait:
                time.sleep(1)
            else:
                image_data = base64.b64decode(image_src)
                image = BytesIO(image_data)
                return True,image
    else:
        return False,task_id
    

def main(prompt):
    show_app.write("**You:** " + prompt)
    flag,image = query_vispunk(prompt)
    show_app.image(image,caption=prompt,use_column_width=True)


prompt = st.chat_input("Send your prompt")
if prompt:
    main(prompt)