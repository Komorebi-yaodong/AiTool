from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_mic_recorder import mic_recorder
from streamlit.components.v1 import html,iframe
import google.generativeai as genai
import speech_recognition as sr
from PyDeepLX import PyDeepLX
from docx import Document
from openai import OpenAI
import streamlit as st
from gtts import gTTS
from PIL import Image
import requests
import hashlib
import base64
import langid
import PyPDF2
import io

if "openai_model_list" not in st.session_state:
    
    # author parameter
    st.session_state.author_key = ""
    st.session_state.gpt_choice = True
    st.session_state.gpt_choice_name = "Gemini"

    # chat parameter
    st.session_state.mode_list = ["**🤖Chat**","**🔤Deeplx**","**🎨Txt2Img**"]
    st.session_state.mode = "**🤖Chat**"
    st.session_state.sys_prompt = ""
    st.session_state.chat_speech = True
    st.session_state.speech_input = False
    st.session_state.speech_input_lists = ["中文-zh","English-en","日本語-ja","Русский язык-ru","Deutsch-de","Français-fr","중국어-ko"]
    st.session_state.speech_language = st.session_state.speech_input_lists[0]
    st.session_state.audio_prompt = None
    st.session_state.chat_short_file = None

    st.session_state.openai_model_list = [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-instruct",
        "gpt-4",
        "gpt-4-32k",
        "gpt-4-1106-preview",
    ]
    st.session_state.openai_model = st.session_state.openai_model_list[0]
    st.session_state.openai_session = []
    st.session_state.openai_history = []

    st.session_state.google_model_list = ["gemini-pro","gemini-pro-vision"]
    st.session_state.google_model = st.session_state.google_model_list[0]
    st.session_state.google_session = []
    st.session_state.google_histgory = []
    st.session_state.google_attachment = None

    # translate parameter
    st.session_state.translate_session = []
    st.session_state.lang_lists = ["auto","中文-zh","English-en","日本語-ja","Русский язык-ru","Deutsch-de","Français-fr","중국어-ko"]
    st.session_state.target_lang = st.session_state.lang_lists[0]
    st.session_state.translate_speech = True
    st.session_state.translate_api_list = [
        "https://api.deeplx.org/translate",
        "https://deeplx.aivvm.com/",
        "PyDeeplx"]
    st.session_state.translate_api = st.session_state.translate_api_list[0]

    # draw parameter
    st.session_state.draw_model = "初始-StableDiffusion-2-1"
    st.session_state.draw_model_list = {
        "现实-AbsoluteReality_v1.8.1":"https://api-inference.huggingface.co/models/digiplay/AbsoluteReality_v1.8.1",
        "现实-Absolute-Reality-1.81":"https://api-inference.huggingface.co/models/Lykon/absolute-reality-1.81",
        "动漫-AingDiffusion9.2":"https://api-inference.huggingface.co/models/digiplay/AingDiffusion9.2",
        "现实动漫-BluePencilRealistic_v01":"https://api-inference.huggingface.co/models/digiplay/bluePencilRealistic_v01",
        "动漫写实-Counterfeit-v2.5":"https://api-inference.huggingface.co/models/gsdf/Counterfeit-V2.5",
        "动漫写实-Counterfeit-v25-2.5d-tweak":"https://api-inference.huggingface.co/models/digiplay/counterfeitV2525d_tweak",
        "动漫可爱-Cuteyukimix":"https://api-inference.huggingface.co/models/stablediffusionapi/cuteyukimix",
        "动漫可爱-Cuteyukimixadorable":"https://api-inference.huggingface.co/models/stablediffusionapi/cuteyukimixadorable",
        "现实动漫-Dreamshaper-7":"https://api-inference.huggingface.co/models/Lykon/dreamshaper-7",
        "现实动漫-Dreamshaper_LCM_v7":"https://api-inference.huggingface.co/models/SimianLuo/LCM_Dreamshaper_v7",
        "动漫3D-DucHaitenDreamWorld":"https://api-inference.huggingface.co/models/DucHaiten/DucHaitenDreamWorld",
        "现实-EpiCRealism":"https://api-inference.huggingface.co/models/emilianJR/epiCRealism",
        "现实照片-EpiCPhotoGasm":"https://api-inference.huggingface.co/models/Yntec/epiCPhotoGasm",
        "动漫丰富-Ether-Blu-Mix-b5":"https://api-inference.huggingface.co/models/tensor-diffusion/Ether-Blu-Mix-V5",
        "动漫-Flat-2d-Animerge":"https://api-inference.huggingface.co/models/jinaai/flat-2d-animerge",
        "动漫风景-Genshin-Landscape-Diffusion":"https://api-inference.huggingface.co/models/Apocalypse-19/Genshin-Landscape-Diffusion",
        "现实照片-Juggernaut-XL-v7":"https://api-inference.huggingface.co/models/stablediffusionapi/juggernaut-xl-v7",
        "现实风景-Landscape_PhotoReal_v1":"https://api-inference.huggingface.co/models/digiplay/Landscape_PhotoReal_v1",
        "艺术水墨-MoXin":"https://api-inference.huggingface.co/models/zhyemmmm/MoXin",
        "现实写实-OnlyRealistic":"https://api-inference.huggingface.co/models/stablediffusionapi/onlyrealistic",
        "现实-Realistic-Vision-v51":"https://api-inference.huggingface.co/models/stablediffusionapi/realistic-vision-v51",
        "初始-StableDiffusion-2-1":"https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1",
        "初始-StableDiffusion-XL-0.9":"https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-0.9",
        "动漫-TMND-Mix":"https://api-inference.huggingface.co/models/stablediffusionapi/tmnd-mix",
        "艺术-Zavychromaxl-v3":"https://api-inference.huggingface.co/models/stablediffusionapi/zavychromaxlv3",
        "Dalle-v1.1":"https://api-inference.huggingface.co/models/dataautogpt3/OpenDalleV1.1",
        "Dalle-3-xl":"https://api-inference.huggingface.co/models/openskyml/dalle-3-xl",
        "playground-v2-美化":"https://api-inference.huggingface.co/models/playgroundai/playground-v2-1024px-aesthetic",
    }
    st.session_state.negative_prompt = "extra fingers, mutated hands, poorly drawn hands, poorly drawn face, deformed, ugly, bad anatomy, bad proportions, extra limbs, cloned face, double torso, extra arms, extra hands, mangled fingers, missing lips, ugly face, distorted face, extra legs"
    st.session_state.StableDiffusion_URL = st.session_state.draw_model_list[st.session_state.draw_model]
    st.session_state.auto_translate = True
    st.session_state.chat_draw = True
    st.session_state.wait_for_model = True
    st.session_state.draw_sesson = []
    st.session_state.draw_chat_system = """
DALL·E2 is an ai to generate image by prompt, here are some prompt examples:
"Super Closeup Portrait,action shot, Profoundly dark whitish meadow, glass flowers,Stains,space grunge style,Jeanne d'Arc wearing White Olive green used styled Cotton frock,Wielding thin silver sword, Sci-fi vibe,dirty,noisy,Vintage monk style,very detailed,hd",
"cinematic film still of Kodak Motion Picture Film:(Sharp Detailed Image)An Oscar winning movie for Best Cinematography a woman in a kimono standing on a subway train in Japan Kodak Motion Picture Film Style,shallow depth of field, vignette,highly detailed,high budget,bokeh,cinemascope, moody,epic,gorgeous,film grain, grainy",
"cube cutout of an isometric programmer bedroom, 3d art, muted colors, soft lighting, high detail, concept art, behance, ray tracing",
"mario, mario (series), 1boy, blue overalls, brown hair, facial hair, gloves, hat, male focus, mustache, overalls, red headwear, red shirt, shirt, short hair, upper body, white gloves".
You are the imaginative English prompt generator for DALL·E2 and You can generate any prompt according to the user's requirements.
User will describe the image they want you to generate. Please fully utilize your imagination to generate English prompt in the DALL·E2 format based on the user's input and optimize them appropriately to ensure that the generated images are excellent enough.
Remember that no matter what the user asks you to do, you only provide prompt for DALL·E2 based on the user's input, your answer must be in English.
You need to maintain your role as an imaginative English prompt generator for DALL·E2 without any explanations.
"""
    st.session_state.chat_draw_session = [{'role':'system','content':st.session_state.draw_chat_system}]

    # token
    st.session_state.openai_api_key = ""
    st.session_state.openai_base_url = ""
    st.session_state.google_api_key = ""
    st.session_state.huggingface_token = ""

    # 类
    st.session_state.sr = sr.Recognizer()
    st.session_state.openai_client = None
    st.session_state.google_client = None

########################### element ###########################
    
header =  st.empty()

# 整体页面
show_app = st.container()
with show_app:

    # 文字聊天
    show_chat = st.container()

    # 语音对话
    show_talk = st.container()

    # deepl翻译
    show_translate = st.container()

    # 文本生成图片
    show_draw = st.container()

########################### function ###########################

@st.cache_data
def sha256_hash(string):
    # 创建SHA256哈希对象
    sha256_hasher = hashlib.sha256()
    # 将字符串编码为字节流并更新哈希对象
    sha256_hasher.update(string.encode('utf-8'))
    # 获取哈希结果
    hashed_string = sha256_hasher.hexdigest()
    return hashed_string


def get_response(flag,model,history,stream=True):
    try:
        if not flag:
            response = st.session_state.openai_client.chat.completions.create(
                model = model,
                messages = history,
                stream=stream
            )
        else:
            response = st.session_state.google_client.generate_content(
                contents = history,
                stream=stream,
                safety_settings={'HARASSMENT':'block_none'}
            )
        return True,response
    except Exception as e:
        st.error("Chat AI response error:{}".format(e))
        return False,e


def chat_ai(message,model,history,session,flag=st.session_state.gpt_choice,attachment=st.session_state.google_attachment):
    if len(history) != 0 and len(session) != 0:
        if history[-1]["role"] == "user":
            history.pop()
        if session[-1]["role"] == "user":
            session.pop()

    # openai
    if not flag:
        history.append({"role":"user","content":message})
        session.append({"role":"user","content":message})
        response_check,response = get_response(flag,model,history)
        if response_check:
            show_chat_page(flag,session)
            reply={"role":"assistant","content":""}
            with show_chat:
                with st.chat_message(reply["role"]):
                    line = st.empty()
                    for chunk in response:
                        message = chunk.choices[0].delta.content
                        if message is not None:
                            reply["content"] += message
                            line.empty()
                            line.write(reply["content"])
                history.append(reply)
                session.append(reply)
                if st.session_state.chat_speech == True:
                    if reply["content"] != "":
                        mytts(reply["content"])
            
            
        else:
            history.pop()
            session.pop()
            st.error(response)
    # google
    else:
        if model == "gemini-pro-vision":
            if attachment is not None:
                history=[{"role":"user","parts":[message,]+attachment},]
                session=[{"role":"user","parts":[message,]+attachment},]
                attachment = None
            else:
                st.error("Please attach a Image")
                return False
        else:
            history.append({"role":"user","parts":[message,]})
            session.append({"role":"user","parts":[message,]})
        response_check,response = get_response(flag,model,history)
        if response_check:
            show_chat_page(flag,session)
            reply = {"role":"model","parts":["",]}
            with show_chat:
                with st.chat_message(reply["role"]):
                    line = st.empty()
                    for chunk in response:
                        try:
                            message = chunk.text
                            reply["parts"][0] += message
                            line.empty()
                            line.write(reply["parts"][0])
                        except Exception as e:
                            print(f'{type(e).__name__}: {e}')
                history.append(reply)
                session.append(reply)
                if st.session_state.chat_speech == True:
                    if reply["parts"][0]!="":
                        mytts(reply["parts"][0])
            
            
        else:
            if model != "gemini-pro-vision":
                history.pop()
                session.pop()
            st.error(response)


def mytts(text):
    def autoplay_audio(audio_data:io.BytesIO):
        data = audio_data.getvalue()
        b64 = base64.b64encode(data).decode()
        md = f"""
                <audio controls autoplay="true" id="myAudio" style="width: 100%;">
                    <source src="data:audio/ogg;base64,{b64}" type="audio/ogg">
                </audio>
                <script>
                    var audio = document.getElementById("myAudio");
                    audio.playbackRate = 1.5; 
                </script>
                """
        html(md)
    
    text = text.replace("```"," ").replace("`"," ").replace("***"," ").replace("**"," ").replace("$$"," ").replace("###"," ").replace("##"," ").replace("#"," ").replace("---"," ")
    lang,conf = langid.classify(text)
    tts = gTTS(text=text,lang=lang)
    speach_BytesIO = io.BytesIO()
    tts.write_to_fp(speach_BytesIO)
    autoplay_audio(speach_BytesIO)
    st.write(lang,conf)


@st.cache_data
def audio2text(audio_prompt,language):
    audio_data = sr.AudioData(audio_prompt['bytes'],audio_prompt['sample_rate'],audio_prompt['sample_width'])
    output = st.session_state.sr.recognize_google(audio_data,language=language)
    return output


def show_chat_page(flag,session):
    if not flag:
        with show_chat:
            for section in session:
                with st.chat_message(section["role"]):
                    st.write(section["content"])
    else:
        with show_chat:
            for section in session:
                with st.chat_message(section["role"]):
                    for piece in section["parts"]:
                        if isinstance(piece,str):
                            st.write(piece)
                        elif isinstance(piece,Image.Image):
                            st.image(piece,use_column_width=True)


@st.cache_data
def get_file_reader(file,name,type):

    def get_text(file,type):
    
        def extract_text_from_docx(file):
            doc = Document(file)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        def extract_text_from_pdf(file):
            pdf = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(pdf.pages)):
                page = pdf.pages[page_num]
                text += page.extract_text()
                
            return text
        
        # 文件类型判断
        if type == 'pdf':
            text = extract_text_from_pdf(file)
        elif type == 'docx':
            text = extract_text_from_docx(file)
        elif type == 'txt' or type == 'md' or type == 'py' or type == 'c' or type == 'cpp' or type == 'js':
            text = file.getvalue().decode("utf-8")
        else:
            st.error("The file type is not supported.(only pdf, docx, txt, md supported)")
            return []
        
        return text
    
    
    def get_splitted_text(text):
        r_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=0
        )
        return r_splitter.split_text(text)
    
    assistant_reply = "Acknowledged"
    start_content = "You are a file reading bot. Next, the user will send a file. After reading, you should fully understand the content of the file and be able to analyze, interpret, and respond to questions related to the file in both Chinese and Markdown formats. Please only answer questions based on the content of the document. If the question is not mentioned in the document, please reply directly to the article without referring to other materials. Answer step-by-step."
    end_content = "File sent. Next, please reply in Chinese and format your response using markdown based on the content.'"
    st.session_state.openai_history = [{'role':'system','content':start_content}]
    st.session_state.google_histgory = [{'role':'user','parts':[start_content,]},{'role':'model','parts':[assistant_reply,]}]

    # 文本提取并拆分
    text = get_text(file,type)
    text_list = get_splitted_text(text)
    pages = len(text_list)
    start_message = f"The file name is {name}, and I will now send you the content of the file in {len(text_list)} sections. Please ensure that you are ready to receive the instructions for sending the file. Once you receive the instructions, please be prepared to answer my question."
    st.session_state.openai_history+=[{'role':'user','content':start_message},{'role':'assistant','content':assistant_reply}]
    st.session_state.google_histgory+=[{'role':'user','parts':[start_message,]},{'role':'model','parts':[assistant_reply,]}]

    # 分段输入
    for i in range(pages):
        st.session_state.openai_history+=[{'role':'user','content':text_list[i]},{'role':'assistant','content':assistant_reply}]
        st.session_state.google_histgory+=[{'role':'user','parts':[text_list[i],]},{'role':'model','parts':[assistant_reply,]}]

    # 结束文本输入
    st.session_state.openai_history+=[{'role':'user','content':end_content},{'role':'assistant','content':"I have finished reading the file content, you can ask me anything."}]
    st.session_state.google_histgory+=[{'role':'user','parts':[end_content,]},{'role':'model','parts':["I have finished reading the file content, you can ask me anything.",]}]



def deeplx_translate(text,source_lang,target_lang,api):
    if api == st.session_state.translate_api_list[0]:
        if source_lang is None:
            source_lang="auto"
        headers = {"Content-Type": "application/json"}
        body = {
            "text":text,
            "target_lang":target_lang,
            "source_lang":source_lang
        }
        try:
            response = requests.post(api, json=body, headers=headers)
            return True,response.json()["data"]
        except Exception as e:
            st.error("Deeplx response error: {}".format(e))
            return False,e
    elif api == st.session_state.translate_api_list[1]:
        if source_lang is None:
            source_lang,conf = langid.classify(text)
        headers = {"Content-Type": "application/json"}
        body = {
            "text":text,
            "target_lang":target_lang,
            "source_lang":source_lang
        }
        try:
            response = requests.post(api, json=body, headers=headers)
            return True,response.json()["response"]["translated_text"]
        except Exception as e:
            st.error("Deeplx response error: {}".format(e))
            return False,e
    elif api == st.session_state.translate_api_list[-1]:
        try:
            response = PyDeepLX.translate(text,'auto',target_lang)
            return True,response
        except Exception as e:
            st.error("Deeplx response error: {}".format(e))
            return False,e


def translate(text,target_lang,api=st.session_state.translate_api):
    st.session_state.translate_session.append({"role":"user","content":text})
    show_translate_page()
    if target_lang == "to":
        lang,conf = langid.classify(text)
        if lang == "zh":
            flag,reply = deeplx_translate(text,lang,"en",api)
        else:
            lang_list = [i[-2:] for i in st.session_state.lang_lists]
            lang_list.remove("to")
            if lang not in lang_list:
                flag,reply = deeplx_translate(text,"en","zh",api)
            else:
                flag,reply = deeplx_translate(text,lang,"zh",api)
    else:
        flag,reply = deeplx_translate(text,None,target_lang,api)
    if flag:
        st.session_state.translate_session.append({"role":"assistant","content":reply})
        
        with show_translate.chat_message("assistant"):
            st.write(reply)
        if st.session_state.translate_speech == True:
            if reply != "":
                mytts(reply)
    else:
        st.error(reply)


def show_translate_page():
    for section in st.session_state.translate_session:
        with show_translate.chat_message(section['role']):
            st.write(section['content'])


def text2img(prompt,token=st.session_state.huggingface_token,StableDiffusion_URL=st.session_state.StableDiffusion_URL):
    def query(payload):
        try:
            response = requests.post(StableDiffusion_URL, headers=StableDiffusion_headers, json=payload)
            if response.status_code == 200:
                return True, response.content
            else:
                return False,response.content
        except requests.exceptions.RequestException as e:
            return False,e
        
    StableDiffusion_headers = {"Authorization":"Bearer "+token}
    
    if st.session_state.chat_draw:
        if len(st.session_state.chat_draw_session) != 0:
            if st.session_state.chat_draw_session[-1]["role"] == "user":
                st.session_state.chat_draw_session.pop()
        st.session_state.chat_draw_session.append({"role":"user","content":prompt})
        response_check,response = get_response(False,st.session_state.openai_model,st.session_state.chat_draw_session,stream=False)
        if response_check:
            prompt = response.choices[0].message.content
            st.session_state.chat_draw_session.append({"role":"assistant","content":prompt})
            if st.session_state.auto_translate:
                lang,conf = langid.classify(prompt)
                if lang != "en":
                    flag,prompt = deeplx_translate(prompt,lang,"en",st.session_state.translate_api)
                    if not flag:
                        return False
            flag,response = query({
                "inputs":prompt,
                "negative_prompt":st.session_state.negative_prompt,
                "wait_for_model":st.session_state.wait_for_model,
            })
            image = response
            show_draw_page()
            st.session_state.draw_sesson.append({"prompt":prompt,"image":image,"flag":flag})
            with show_draw.chat_message("assistant"):
                if flag:
                    st.image(image,prompt,use_column_width=True)
                else:
                    st.write(prompt,"\n",image)
    else:
        if st.session_state.auto_translate:
            lang,conf = langid.classify(prompt)
            if lang != "en":
                flag,prompt = deeplx_translate(prompt,lang,"en",st.session_state.translate_api)
                if not flag:
                    return False
        flag,response = query({
            "inputs":prompt,
            "negative_prompt":st.session_state.negative_prompt,
            "wait_for_model":st.session_state.wait_for_model,
        })
        image = response
        show_draw_page()
        st.session_state.draw_sesson.append({"prompt":prompt,"image":image,"flag":flag})
        with show_draw.chat_message("assistant"):
            if flag:
                st.image(image,prompt,use_column_width=True)
            else:
                st.write(prompt,"\n",image)


def show_draw_page():
    for section in st.session_state.draw_sesson:
        with show_draw.chat_message("assistant"):
            if section["flag"]:
                st.image(section["image"],section["prompt"],use_column_width=True)
            else:
                st.write(section["prompt"],"\n",section["image"])
########################### mount ###########################

def new_chat():

    # openai
    if st.session_state.sys_prompt == "":
        st.session_state.openai_history = []
    else:
        st.session_state.openai_history = [{"role":"system","content":st.session_state.sys_prompt},]
    st.session_state.openai_session = []

    # google
    st.session_state.google_histgory = []
    st.session_state.google_session = []
    if st.session_state.google_api_key:
        genai.configure(api_key=st.session_state.google_api_key)
        st.session_state.google_client = genai.GenerativeModel(st.session_state.google_model)

    # translate
    st.session_state.translate_session = []
    
    # draw
    
    st.session_state.draw_sesson = []
    st.session_state.chat_draw_session = [{'role':'system','content':st.session_state.draw_chat_system}]

def author_channel():
    author_key_hash = sha256_hash(st.session_state.author_key.strip())
    if author_key_hash in st.secrets.pwsds:
        # openai
        st.session_state.openai_api_key = st.secrets.openai_api_keys[st.secrets.pwsds[author_key_hash]]
        st.session_state.openai_base_url = st.secrets.openai_base_urls[st.secrets.pwsds[author_key_hash]]
        st.session_state.openai_client = OpenAI(
            api_key=st.session_state.openai_api_key,
            base_url=st.session_state.openai_base_url
        )
        # google
        st.session_state.google_api_key = st.secrets.google_api_keys[st.secrets.pwsds[author_key_hash]]
        genai.configure(api_key=st.session_state.google_api_key)
        st.session_state.google_client = genai.GenerativeModel(st.session_state.google_model)
        # huggingface
        st.session_state.huggingface_token = st.secrets.huggingface_tokens[st.secrets.pwsds[author_key_hash]]
        

def gpt_choice():
    st.session_state.gpt_choice = not st.session_state.gpt_choice
    if st.session_state.gpt_choice:
        st.session_state.gpt_choice_name = "Gemini"
    else:
        st.session_state.gpt_choice_name = "ChatGPT"


def upload_google_attachment():
    st.session_state.google_attachment = st.session_state.google_attachment
    if st.session_state.google_attachment is not None:
        attachment = []
        for upload_img in st.session_state.google_attachment:
            attachment.append(Image.open(upload_img))
        st.session_state.google_attachment = attachment
        

def get_file_chat():

    def collect_file(file_upload):
        file_name = ".".join(file_upload.name.split('.')[0:-1])
        file_type = file_upload.name.split('.')[-1]

        return file_name,file_type
    
    st.session_state.chat_short_file = st.session_state.chat_short_file
    if st.session_state.chat_short_file:
        file_name,file_type = collect_file(st.session_state.chat_short_file)
        get_file_reader(st.session_state.chat_short_file,file_name,file_type)

def change_paramater():
    st.session_state.openai_api_key = st.session_state.openai_api_key
    st.session_state.openai_base_url = st.session_state.openai_base_url
    st.session_state.sys_prompt = st.session_state.sys_prompt
    st.session_state.google_api_key = st.session_state.google_api_key
    st.session_state.chat_speech = st.session_state.chat_speech
    st.session_state.google_api_key = st.session_state.google_api_key
    st.session_state.speech_input = st.session_state.speech_input
    st.session_state.speech_language = st.session_state.speech_language
    st.session_state.draw_model = st.session_state.draw_model
    st.session_state.StableDiffusion_URL = st.session_state.draw_model_list[st.session_state.draw_model]
    st.session_state.huggingface_token = st.session_state.huggingface_token
    st.session_state.negative_prompt = st.session_state.negative_prompt
    st.session_state.mode = st.session_state.mode
    st.session_state.translate_api = st.session_state.translate_api
    st.session_state.target_lang = st.session_state.target_lang
    st.session_state.translate_speech = st.session_state.translate_speech
    st.session_state.auto_translate = st.session_state.auto_translate
    st.session_state.chat_draw = st.session_state.chat_draw
    st.session_state.wait_for_model = st.session_state.wait_for_model

def get_save():
    change_paramater()
    # openai
    if st.session_state.openai_api_key and st.session_state.openai_base_url:
        st.session_state.openai_client = OpenAI(
            api_key=st.session_state.openai_api_key,
            base_url=st.session_state.openai_base_url
        )
    if len(st.session_state.openai_history) == 0:
        if st.session_state.sys_prompt != "":
            st.session_state.openai_history = [{"role":"system","content":st.session_state.sys_prompt},]
    # google
    if st.session_state.google_api_key:
        genai.configure(api_key=st.session_state.google_api_key)
        st.session_state.google_client = genai.GenerativeModel(st.session_state.google_model)

    # show
    if st.session_state.mode == "**🤖Chat**":
        if not st.session_state.gpt_choice:
            show_chat_page(False,st.session_state.openai_session)
        else:
            show_chat_page(True,st.session_state.google_session)
    elif st.session_state.mode == "**🔤Deeplx**":
        show_translate_page()
    elif st.session_state.mode == "**🎨Txt2Img**":
        show_draw_page()
    


########################### sidebar ###########################

with st.sidebar:
    
    # 新的开始
    with st.container():
        st.button("🆕 New Chat",use_container_width=True,key="New Chat")
        if st.session_state.get("New Chat"):
            new_chat()

    # 作者通道
    with st.container():
        st.session_state.author_key = st.text_input("author channel",type='password',value=st.session_state.author_key,key="author channel")
        if st.session_state.get("author channel"):
            author_channel()

    # 聊天设置
    with st.container():
        with st.expander("**Chat Settings**"):
            col1,col2 = st.columns(2)
            with col1:
                st.session_state.gpt_choice = st.toggle(st.session_state.gpt_choice_name,value=st.session_state.gpt_choice,on_change=gpt_choice)
            with col2:
                st.session_state.chat_speech = st.toggle("speech",st.session_state.chat_speech,on_change=change_paramater)
            if not st.session_state.gpt_choice:
                st.session_state.openai_model = st.selectbox("Chat Models",sorted(st.session_state.openai_model_list),on_change=new_chat)
                st.session_state.openai_api_key = st.text_input("api key",value=st.session_state.openai_api_key,type='password')
                st.session_state.openai_base_url = st.text_input("api base",value=st.session_state.openai_base_url)
                st.session_state.sys_prompt = st.text_input("sys prompt",value=st.session_state.sys_prompt,on_change=change_paramater)
                st.session_state.chat_short_file = st.file_uploader("Chat short file",label_visibility="collapsed")
                st.button("ChatFile",use_container_width=True,key="ChatFile")
                if st.session_state.get("ChatFile"):
                    get_file_chat()
            else:
                st.session_state.google_model = st.selectbox("Chat Models",sorted(st.session_state.google_model_list),on_change=new_chat)
                st.session_state.google_api_key = st.text_input("api key",value=st.session_state.google_api_key,type='password',on_change=change_paramater)
                if st.session_state.google_model == "gemini-pro-vision":
                    st.session_state.google_attachment = st.file_uploader("Image for gemini-pro-vision",type=['jpg','png','jpeg'],accept_multiple_files=True,label_visibility="collapsed")
                    st.button("Send Image",key="google attachment",use_container_width=True)
                    if st.session_state.get("google attachment"):
                        upload_google_attachment()
                else:
                    st.session_state.chat_short_file = st.file_uploader("Chat short file",label_visibility="collapsed")
                    st.button("ChatFile",use_container_width=True,key="ChatFile")
                    if st.session_state.get("ChatFile"):
                        get_file_chat()
            st.session_state.speech_input = st.toggle("talk mode",st.session_state.speech_input,on_change=change_paramater)
                
    # 翻译设置
    with st.container():
        with st.expander("**Translate Settings**"):
            st.session_state.translate_api = st.selectbox("Translate API",st.session_state.translate_api_list,on_change=change_paramater)
            st.session_state.target_lang = st.selectbox("Target Language",st.session_state.lang_lists,on_change=change_paramater)
            st.session_state.translate_speech = st.toggle('translate speech', st.session_state.translate_speech,on_change=change_paramater)
    
    # 绘画设置
    with st.container():
        with st.expander("**Draw Settings**"):
            st.session_state.draw_model = st.selectbox('Draw Models', sorted(st.session_state.draw_model_list.keys(),key=lambda x:x.split("-")[0]),on_change=change_paramater)
            st.session_state.huggingface_token = st.text_input('Huggingface Token',type='password',value=st.session_state.huggingface_token,on_change=change_paramater)
            st.session_state.negative_prompt = st.text_input('Negative Prompt',value=st.session_state.negative_prompt,on_change=change_paramater)
            # col1,col2,col3 = st.columns(3)
            # with col1:
            st.session_state.chat_draw = st.toggle('Chat', st.session_state.chat_draw,on_change=change_paramater)
            # with col2:
            st.session_state.auto_translate = st.toggle('Translate', st.session_state.auto_translate,on_change=change_paramater)
            # with col3:
            st.session_state.wait_for_model = st.toggle('Wait', st.session_state.wait_for_model,on_change=change_paramater)

    # 保存
    st.button("Save",use_container_width=True,key="Save")
    if st.session_state.get("Save"):
        get_save()


    # 模式
    with st.container():
        with st.container():
            st.session_state.mode = st.radio("Choose Mode",st.session_state.mode_list,on_change=change_paramater)


########################### 聊天展示区 ###########################

if st.session_state.mode == "**🤖Chat**":
    if not st.session_state.gpt_choice:
        header.write("<h2> 🤖 "+st.session_state.openai_model+"</h2>",unsafe_allow_html=True)
    else:
        header.write("<h2> 🤖 "+st.session_state.google_model+"</h2>",unsafe_allow_html=True)
    if not st.session_state.speech_input:
        user_prompt = st.chat_input("Send a message")
        if user_prompt:
            if not st.session_state.gpt_choice:
                chat_ai(user_prompt,st.session_state.openai_model,st.session_state.openai_history,st.session_state.openai_session)
            else:
                chat_ai(user_prompt,st.session_state.google_model,st.session_state.google_histgory,st.session_state.google_session)
    else:
        with st.container():
            st.session_state.speech_language = st.selectbox("🎙️language",st.session_state.speech_input_lists,on_change=change_paramater)
            st.session_state.audio_prompt = mic_recorder(
                start_prompt="🎙️开始说话",
                stop_prompt="🛑结束说话", 
                just_once=True,
                use_container_width=True,
                callback=None,
                args=(),
                kwargs={},
                key=None
            )
        if st.session_state.audio_prompt:
            user_prompt = audio2text(st.session_state.audio_prompt,st.session_state.speech_language[-2:])
            if not st.session_state.gpt_choice:
                chat_ai(user_prompt,st.session_state.openai_model,st.session_state.openai_history,st.session_state.openai_session)
            else:
                chat_ai(user_prompt,st.session_state.google_model,st.session_state.google_histgory,st.session_state.google_session)


elif st.session_state.mode == "**🔤Deeplx**":
    header.write("<h2> 🔤 Deeplx-"+st.session_state.target_lang+"</h2>",unsafe_allow_html=True)
    txt_prompt = st.chat_input("Input your content to be translated",max_chars=5000)
    if txt_prompt:
        translate(txt_prompt,st.session_state.target_lang[-2:])

elif st.session_state.mode == "**🎨Txt2Img**":
    header.write("<h2> 🎨 "+st.session_state.draw_model+"</h2>",unsafe_allow_html=True)
    draw_prompt = st.chat_input("Send your prompt")
    if draw_prompt:
        text2img(draw_prompt)

change_paramater()