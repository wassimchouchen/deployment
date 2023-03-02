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
import tensorflow as tf
import openai
import transformers
import re
import requests
from django.contrib.auth.models import User
from rest_framework import serializers, viewsets, status
from rest_framework.decorators import action
from rest_framework.permissions import AllowAny
from rest_framework.authtoken.models import Token
from django.contrib.auth import get_user_model
from django.contrib.auth import login
import jwt
from django.contrib.auth import authenticate
openai.api_key = "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"
headers = {"Authorization": "Bearer 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"}



##authentification class for register and login(including creating user)

class UserSerializer(serializers.Serializer):
    username = serializers.CharField()
    email = serializers.EmailField()
    password = serializers.CharField(write_only=True)

    def validate_username(self, value):
        """
        Check that the username is unique.
        """
        if User.objects.filter(username=value).exists():
            raise serializers.ValidationError("A user with that username already exists.")
        return value

    def validate_email(self, value):
        """
        Check that the email is unique.
        """
        if User.objects.filter(email=value).exists():
            raise serializers.ValidationError("A user with that email already exists.")
        return value
    
    def create(self, validated_data):
        user = User.objects.create(
            username=validated_data['username'],
            email=validated_data['email']
        )
        user.set_password(validated_data['password'])
        user.save()
        return user
    
    def update(self, instance, validated_data):
        instance.username = validated_data.get('username', instance.username)
        instance.email = validated_data.get('email', instance.email)
        instance.save()
        return instance

@csrf_exempt  
def register(request):
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    serializer = UserSerializer(data=body)
    if serializer.is_valid():
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        password = body.get('password') 
        user = serializer.save()        
        user.set_password(password)  
        user = serializer.save()
        return JsonResponse(serializer.data, status=201)
    return JsonResponse(serializer.errors, status=400)


@csrf_exempt 
def loginn(request):
    body_unicode = request.body.decode('utf-8')
    body = json.loads(body_unicode)
    username = body.get('username')
    password = body.get('password')
    user = authenticate(request,username=username, password=password)
    # user = authenticate(request, username=username, password=password, backend='django.contrib.auth.backends.ModelBackend')
    if user:
        print(user)
        if user.is_active:
            login(request, user)
            secret = 'secret'
            token = jwt.encode({'username': user.username},secret , algorithm='HS256')
            return JsonResponse({"token": token})
            # return JsonResponse({"token": token.key})
        else:
            return JsonResponse({"error": "User is not active."}, status=400)
    else:
        print(user)
        return JsonResponse({"error": "Invalid credentials."}, status=400)

#######################################################################################################################

def generate_3d(request):
    # ...
    if request.method == 'POST':
        # ...
        points = response['choices'][0]['text']

        # parse the points string and convert to a list of 3D coordinates
        points_list = []
        for point in points.split(','):
            x, y, z = point.split(' ')
            points_list.append([float(x), float(y), float(z)])

        # save the points to a JSON file
        with open('points.json', 'w') as f:
            json.dump(points_list, f)

        # return the JSON file as a response
        with open('points.json', 'r') as f:
            points_json = f.read()
        return HttpResponse(points_json, content_type='application/json')


# set the API URL

# set the API headers (if necessary)
API_KEY = "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"
@csrf_exempt 
def generate_3d_points(request):
    openai.api_key = API_KEY
    model_engine = "openai/point-e"
    if request.method == 'POST':

        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        completions = openai.Completion.create(
            engine=model_engine,
            prompt=text,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )
        message = completions.choices[0].text
        generated_points = generate_points(prompt)
        print(generated_points)
        response_data = {'generated_points': generated_points}
        return  JsonResponse(response_data)


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








# Set the API endpoint URL
API_URL = "https://api.openai.com/v1/images/generations"
API_KEY = "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"
@csrf_exempt 
def generate_image(request):
    # Make sure the request is a POST request
    if request.method == "POST":
    # Get the input data from the POST request body
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
    # Set the API request headers
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
     }
    # Set the API request data
        data = {
            "model": "image-alpha-001",
            "prompt": text,
            "num_images": 1,
            "size": "256x256",
            "response_format": "image",
    }
    # Send the API request
        response = requests.post(API_URL, headers=headers, json=data)
    # Get the image URL from the response
        # response_data = response.json()
        # image_url = response_data["data"][0]["url"]
        # print(image_url)
    # Return the image URL as a response
        
        image_data = response.content
        # Create an HttpResponse object with the image data
        response = HttpResponse(image_data, content_type='image/jpeg')
        return response


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
        print(generated_text)
        text = generated_text[0]["generated_text"]
        print(text)
        return HttpResponse(text)





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
      if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        # translated = models.Translaten2ar(text)       
        # return HttpResponse(translated)
        API_URL_en2ar  = "https://api.openai.com/v1/engines/davinci/jobs"


        payload = {
            "model": "text-davinci-002",
            "prompt": f"translate this to arabic: {text}",
            "max_tokens": 1024,
            "temperature": 0.5
        }

        response = requests.post(API_URL_en2ar, headers=headers, json=payload)
        response_json = response.json()
        print(response_json)
        translated_text = response_json(text)
        translated = translated_text       
        return HttpResponse(translated[0]["translation_text"])
    # """this function execute the translator api from english to arabic"""

    # if request.method == 'POST':
    #     body_unicode = request.body.decode('utf-8')
    #     body = json.loads(body_unicode)
    #     text = body.get('text', "")
    #     translated = models.Translaten2ar(text)       
    #     return HttpResponse(translated)





@csrf_exempt
def french_Translator_function(request):
    """this function execute the translator api"""

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        translated = models.french_translator(text) 
        return HttpResponse(translated[0]["translation_text"])

















# @csrf_exempt 
# def Openai_Generator_image_function(request):
#     """this function execute the openai api to generate images"""

#     if request.method == 'POST':
#         body_unicode = request.body.decode('utf-8')
#         body = json.loads(body_unicode)
#         # print(body, type(body))
#         # text = body["text"]
#         text = body.get('text', "")
#         output = models.Openai_Generator_image(text)
#         # img = open('output', 'rb')
#         # image = FileResponse(img)
#         # image=Image.open(BytesIO(output))
#         return output


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


