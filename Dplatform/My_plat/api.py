import requests
import torch
import json
import openai
from PIL import Image  # Import the required libraries
import base64
from io import BytesIO
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from diffusers import StableDiffusionPipeline
# My OpenAI API key
openai.api_key = "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"
# My hugging_face API key
headers = {"Authorization": "Bearer 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"}


# # Use the text generation function to generate a response
# def ChatGPT(payload) :
#     response = openai.Completion.create(
#     engine="text-davinci-002",
#     prompt=json.dumps(payload),
#     max_tokens=4000,
#     temperature=0.2,)
#     return (response["choices"][0]["text"]) 


# Use the text generation function to generate a response
def GPT35(payload) :
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content":payload }])
        return (response["choices"][0]["content"]) 


def ASR_WHISPER(payload) :
        file = open(payload, "rb")
        response = openai.Audio.transcribe("whisper-1", file)
        return (response["text"]) 

API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h-lv60-self"


def transcription_model(payload):
    with open(payload, "rb") as f:
        data = f.read()
        # data.decode('utf-16')
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))


API_URL = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-fr"
def french_translator(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	

 


API_URL_translator = "https://api-inference.huggingface.co/models/t5-small"
def Translator(payload):
	response = requests.post(API_URL_translator, headers=headers, json=payload)
	return response.json()



API_URL_en2ar  = "https://api.openai.com/v1/engines/davinci/jobs"
def Translaten2ar(payload):
    # payload = {
    #     "model": "text-davinci-002",
    #     "prompt": f"translate this to arabic: {text}",
    #     "max_tokens": 1024,
    #     "temperature": 0.5
    # }
	response = requests.post(API_URL_en2ar, headers=headers, json=payload)
	return response.json()




API_URL_Summarizer = "https://api-inference.huggingface.co/models/sshleifer/distilbart-xsum-12-3"
def Summarizer(payload):
	# response = requests.post(API_URL_Summarizer, headers=headers, json=payload)
    data = json.dumps(payload)
    response = requests.request("POST", API_URL_Summarizer, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))
	# return response.json()
	



API_URL_text_Gen = "https://api-inference.huggingface.co/models/gpt2"
def text_generation(payload):
	response = requests.post(API_URL_text_Gen, headers=headers, json=payload)
	return response.json()

	

	


API_URL_QA = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
def QA(payload):
	response = requests.post(API_URL_QA, headers=headers, json=payload)
	return response.json()
	

def Image_generation(payload):
# Load the pre-trained model
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    data = json.dumps(payload)
    # Generate images using the model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    output = model.generate(data, max_length=1024, top_k=50, top_p=0.9, temperature=0.7,pad_token_id=50256)
    # Tokenize the output using the GPT2Tokenizer
   
    tokens = tokenizer.decode(output, skip_special_tokens=True)
    # Convert the tokens back into the original image format
    original_image = decode_tokens(tokens)
    return original_image

