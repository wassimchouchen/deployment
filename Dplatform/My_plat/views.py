from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from My_plat import models
import json
from django.http import HttpResponse
from PIL import Image
from io import BytesIO
import base64
import soundfile as sf


@csrf_exempt 
def receive_audio(request):
    # Handle the POST request and access the audio data here
    if request.method == 'POST':
        audio_data = request.body
        print("received body")
        audio_data_decoded = base64.b64decode(audio_data)
        print("received decode")
        audio_data_info = sf.info(audio_data_decoded)
        sf.write("audio_file.wav", audio_data_decoded, audio_data_info.samplerate)
        print("audio_data_info")

        return HttpResponse("Audio received")




@csrf_exempt 
def FB_transcribe(request):
    """this function execute the hugging_face api to transcribe audio"""

    if request.method == 'POST':
        audio_data = request.body
        # audio_data_decoded = base64.b64decode(audio_data)
        audio_data.decode('utf-8')
        print("decoded")
        output = models.transcription_model(audio_data_decoded)
        return output


## test this api tomorrow 
@csrf_exempt 
def Diffuser_function(request):
    """this function execute the diffuser api to generate images"""

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        
        text = body.get('text', "")
        output = models.Image_generation(text)
        # img = open('output', 'rb')
        # image = FileResponse(img)
        image=Image.open(BytesIO(output))
        return image



@csrf_exempt 
def Openai_Generator_image_function(request):
    """this function execute the openai api to generate images"""

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        # print(body, type(body))
        # text = body["text"]
        text = body.get('text', "")
        output = models.Openai_Generator_image(text)
        # img = open('output', 'rb')
        # image = FileResponse(img)
        # image=Image.open(BytesIO(output))
        return output


## i need to code solution that make the model continue it's inference until the session ended, continue working on result, chatting 
@csrf_exempt
def conversational_function(request):
    """this function execute the conversational api"""

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        response = models.ChatGPT(text)
        return HttpResponse(response)



@csrf_exempt
def ChatGPT_function(request):
    """this function execute the conversational api but with chatGPT of OpenAI """

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        response = models.ChatGPT(text)
        return HttpResponse(response)



## i need to code solution that make the model continue it's inference until the session ended, continue working on result
@csrf_exempt
def text_generation_function(request):
    """ this function execute the text_generation api"""

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        generated_text = models.text_generation(text)
        return HttpResponse(generated_text)





@csrf_exempt
def summarizer_function(request):
    """ this function execute the summarization api"""

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        Summary = models.Summarizer(text)
        return HttpResponse(Summary)




####       Translation    ###############################################


@csrf_exempt
def Germany_Translator_function(request):
    """this function execute the translator api"""

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        translated = models.Translator(text)       
        return HttpResponse(translated[0]["translation_text"])
 

@csrf_exempt
def arabic_Translator_function(request):
    """this function execute the translator api"""

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        translated = models.TranslatorGPT(text)       
        return HttpResponse(translated)





@csrf_exempt
def french_Translator_function(request):
    """this function execute the translator api"""

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        translated = models.french_translator(text)       
        return HttpResponse(translated[0]["translation_text"])



















# class call_model(APIView):

#     def get(self,request):
#         if request.method == 'GET':
            
#             # sentence is the query we want to get the prediction for
#             params =  request.GET.get('sentence')
            
#             # predict method used to get the prediction
#             response = WebappConfig.predictor.predict(sentence)
            
#             # returning JSON response
#             return JsonResponse(response)





























# from django.shortcuts import render
# from rest_framework import viewsets
# from django.core import serializers
# from rest_framework.response import Response
# from rest_framework import status
# from .forms import ApprovalForm
# from django.http import HttpResponse
# from django.http import JsonResponse
# from django.contrib import messages
# from .models import approvals
# from . serializers import approvalsSerializers
# import pickle
# from keras import backend as K
# import joblib
# import numpy as np
# from sklearn import preprocessing
# import pandas as pd
# from collections import defaultdict, Counter

# def index(request):
#     context={'a':1}
#     return render(request, 'index.html' ,context)

    



# def nlp(request):
#      print (request)
#      print (request.POST.dict())
#      import torch
#      choice = request.POST["choice"]
#      text = request.POST.get("body1", "")
#     #size = Len(text)
#      sen_length =request.POST.get("body2")
#     if(choice == "gpt-2"):
#         print("gpt-2")
#         if text ==" "
#             context={'predictedLabel':'Please enter text to Process'}
#         else:
#             from summarizer import Summarizer, TransformerSummarizer
#             GPT2_model = TransformerSummarizer (transformer_type="GPT2", ,transformer_model_key=gpt2-medium)
#             gpt_summary =" "+ join(GPT2_model(text, inum_sentences=int(sen_length)))
#             print (gpt_summary)
                                                                
#             context={'predictedLabel':'GPT Summarized Text: \n'+gpt_summary}
#     elif choice == "bert":
#         print("bert ")
#         if text ==""
#             context={'predictedLabel':'Please enter text to Process'}
#         else:
#             from summarizer import Summarizer, TransformerSummarizer
#             bert_model = Summarizer ()
#             bert_summary = .join(bert_model( (text, num_sentences=int(sen_length))))
#             print(bert_summary)
#             context={'predictedLabel':'Bert Summarized Text: \n' + bert_summary}

#     # elif choice == "sentence_completion":
#     #      if text ==" "
#     #      context={'predictedLabel':'Please enter text to Process'}
#     #     else:
#     #         from transformers import AutoTokenizer, AutoModelForCausalLM
#     #         tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
#     #         model = AutoModelForCausalLM.from_pretrained("distilgpt2", output_hidden_states=True)
#     #         PATH = './models/simple_transformer.h5'
#     #         model = torch.load(PATH)
#     #         #text = "The chicken didn't cross the road because it was"
#     #          # Tokenize the input string
#     #         input = tokenizer.encode(text, return_tensors=*pt)
#     #          # Run the model
#     #         output = model.generate(input, max_length=int(sen_length), do_sample=True)
#     #          # Print the output
#     #         print(output)


