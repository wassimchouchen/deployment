"""Dplatform URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
# from My_plat.views import AuthViewSet
from My_plat import views

# from django.urls import include
# from rest_framework import routers

# router = routers.DefaultRouter()
# router.register(r'auth', AuthViewSet, basename='auth')



urlpatterns = [
    path('admin/', admin.site.urls),
    path('auth/login', views.loginn),
    path('auth/register', views.register),
    path("talk_to_GPT", views.ChatGPT_function),           
    path('Germany_translator', views.Germany_Translator_function),     
    path("french_translator", views.french_Translator_function),
    path('summarization', views.summarizer_function),
    path('text_generation', views.text_generation_function),
    path('Text_to_Image', views.generate_image),
    path('chat', views.conversational_function),


    path("arabic_translator", views.arabic_Translator_function),
    path('3D_points', views.generate_3d_points),


    path("transcription",views.receive_audio),
    path("transcriptionFB",views.FB_transcribe),
]