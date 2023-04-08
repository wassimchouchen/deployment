from django.views.decorators.csrf import csrf_exempt
from My_plat import models
from django.http import HttpResponse, JsonResponse
import json, requests
from My_plat import apiii

headers = {"Authorization": "Bearer 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"}

@csrf_exempt
def transcribe_view(request):
    if request.method == 'POST':
        audio_file = request.FILES.get('audio')
        if audio_file is None or not audio_file.name.endswith('.wav'):
            return JsonResponse({'error': 'Invalid audio file'})

        with open('record.wav', 'wb') as f:
            for chunk in audio_file.chunks():
                f.write(chunk)
        result = apiii.ASR_WHISPER('record.wav')
        return JsonResponse({'result': result})
    
    return JsonResponse({'error': 'Invalid request method'})


API_URL = "https://api.openai.com/v1/images/generations"
API_KEY = "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"
@csrf_exempt 
def generate_image(request):
    if request.method == "POST":
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
     }
        data = {
            "model": "image-alpha-001",
            "prompt": text,
            "num_images": 1,
            "size": "256x256",
            "response_format": "image",
    }
        response = requests.post(API_URL, headers=headers, json=data)
    # Get the image URL from the response
        # response_data = response.json()
        # image_url = response_data["data"][0]["url"]
        # print(image_url)
    # Return the image URL as a response
        
        image_data = response.content
        # image_data = image_url

        # Create an HttpResponse object with the image data
        response = HttpResponse(image_data, content_type='image/jpeg')
        return response

















previous_response = ""
@csrf_exempt
def text_generation_function(request):
    """ 
    this function execute the text_generation api, it's a master piece :* 
    """

    global previous_response
    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        print(previous_response)
        if previous_response=="":
             response = apiii.GPT35(text)
        else:
             response = apiii.GPT35(previous_response)
    
        previous_response = response[0]["generated_text"]
        
        
        return JsonResponse({"generated_text": response[0]["generated_text"]})
    
@csrf_exempt
def conversational_function(request):
    """
    this function execute the conversational api
    """

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        response = apiii.conversationa(text)
        return HttpResponse(response["generated_text"])


@csrf_exempt
def GPT_turbo(request):
    """
    this function execute the gpt35 api
    """

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        response = apiii.GPT35(text)
        return HttpResponse(response)
    

@csrf_exempt
def ChatGPT_function(request):
    """this function execute the conversational api but with chatGPT of OpenAI """

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        
        response = apiii.ChatGPT(text)
        return HttpResponse(response)
    
@csrf_exempt
def summarizer_function(request):
    """ this function execute the summarization api"""

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        Summary = apiii.Summarizer(text)
        return HttpResponse(Summary)

@csrf_exempt
def Translation(request):
    """this function execute the translator api"""

    if request.method == 'POST':
        body_unicode = request.body.decode('utf-8')
        body = json.loads(body_unicode)
        text = body.get('text', "")
        to_lang = body.get('to_language', "")
        if to_lang == "Germany":
            translated = apiii.Translator(text)    
        elif to_lang == "arabic":
            translated = apiii.en2ar(text)    
        elif to_lang=="french":
            translated = apiii.french_translator(text)
        elif to_lang=="english":
            translated=apiii.ar2en(text)
        else:
            return HttpResponse("please select a language")
        
        return HttpResponse(translated)
    

      





""" this is a gpt3.5 response about how he could help us:  interesting response

    As an AI language model, I can assist you in several areas, such as:
1. Writing: I can help in drafting emails, creating content for social media, writing essays or reports, etc.
2. Research: I can assist in finding information on a particular topic and help in organizing it.
3. Math and Science: I can solve mathematical problems, equations, and help in understanding scientific theories andconcepts.
4. Translation: I can translate text from one language to another.
5. Time Management: I can help you manage your schedule, set reminders, and plan your tasks.
6. Personalized assistance: I can provide suggestions based on your interests, preferences, and behavior patterns.
Please let me know how I can best assist you!




"""


