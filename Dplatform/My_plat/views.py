from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import HttpResponse
import openai
from .apiii import * 

openai.api_key = "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"
headers = {"Authorization": "Bearer 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"}


def chatbot(request):
    if request.method == 'POST':
        payload = request.POST.get('payload')
        chat_log = request.POST.get('chat_log')
        if not chat_log:
            # Start a new conversation
            chat_log = payload + '\n'
        else:
            # Continue an existing conversation
            chat_log += 'user: ' + payload + '\n'
        # Generate a response using GPT-3
        # response = openai.Completion.create(
        #     engine='davinci',
        #     prompt=chat_log,
        #     max_tokens=1024,
        #     n=1,
        #     stop=None,
        #     temperature=0.7,
        # )
        response = models.ChatGPT(chat_log)
        # Append the response to the chat log
        chat_log += 'bot: ' + response + '\n'

        # Return the chat log as a JSON response
        return JsonResponse({'chat_log': chat_log})


import openai
from django.http import JsonResponse
# from .models import Conversation



def chatbotv2(request):
    if request.method == 'POST':
        payload = request.POST.get('payload')
        conversation_id = request.POST.get('conversation_id')

        if not conversation_id:
            # Start a new conversation
            conversation = Conversation.objects.create(chat_log='')
        else:
            # Retrieve the conversation from the database
            conversation = Conversation.objects.get(id=conversation_id)

        # Append the user's message to the chat log
        conversation.chat_log += 'user: ' + payload + '\n'

        # Generate a response using GPT-3 and the conversation history/context
        response = openai.Completion.create(
            engine='davinci',
            prompt=conversation.chat_log,
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.7,
        )

        # Append the response to the chat log
        conversation.chat_log += 'bot: ' + response.choices[0].text + '\n'

        # Save the conversation to the database
        conversation.save()

        # Return the conversation ID and chat log as a JSON response
        return JsonResponse({'conversation_id': conversation.id, 'chat_log': conversation.chat_log})








###############################################################################################################
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
        # generated_points = generate_points(prompt)
        # print(generated_points)
        response_data = {'generated_points': message}
        return  JsonResponse(response_data)
    










# import concurrent.futures

# model = None

# def load_model(model_size):
#     global model
#     if model is None:
#         model = whisper.load_model(model_size)
#     return model

# def transcribe(path, model_size):
#     if path[-3:] != 'wav':
#         subprocess.call(['ffmpeg', '-i', path, 'audio.wav', '-y'])
#         path = 'audio.wav'

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(load_model, model_size)
#         model = future.result()

#     result = model.transcribe(path)
#     return result


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



