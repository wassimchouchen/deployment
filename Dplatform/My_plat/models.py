import requests
from diffusers import StableDiffusionPipeline
import torch
import json
import openai
from PIL import Image




# My OpenAI API key
openai.api_key = "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"
# My hugging_face API key
headers = {"Authorization": "Bearer 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"}


# Use the text generation function to generate a response
def ChatGPT(payload) :
    response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=json.dumps(payload),
    max_tokens=4000,
    temperature=0.2,)
    return (response["choices"][0]["text"]) 


     


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
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16, revision="fp16")
    pipe = pipe.to("cuda")

    prompt = json.dumps(payload)
    image = pipe(prompt).images[0]  
    
    # image.save("name.png")
    return(image)

def Openai_Generator_image(payload):
    response = openai.Image.create(
    prompt=json.dumps(payload),
    n=1,
    size="256x256")
    image_url = response['data'][0]['url']
    return image_url



# def TranslatorGPT(payload):
#     # Use the translation function to generate a translated text
#     response = openai.Completion.create(
#         engine="text-davinci-002",
#         prompt=json.dumps(payload),
#         max_tokens=1024,
#         temperature=0.5,
#         n = 1,
#         stop = "##",
#         languages=["en", "fr"],
#         models=["translation"],)
#     generated_translation = response["choices"][0]["text"]
#     return generated_translation
# API_URL_conversational = "https://api-inference.huggingface.co/models/facebook/blenderbot-400M-distill"
# headers = {"Authorization": "Bearer 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"}

# def conversational(payload):
# 	response = requests.post(API_URL_conversational, headers=headers, json=payload)
#     result=response.json()
#     return (result) 