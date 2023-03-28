from django.contrib.auth import login
import jwt
from django.contrib.auth import authenticate
import requests
from django.contrib.auth.models import User
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
import json
from django.http import HttpResponse
import requests
from rest_framework import serializers
from rest_framework.authtoken.models import Token
from django.contrib.auth import get_user_model
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
