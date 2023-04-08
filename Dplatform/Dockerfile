FROM python:3.9

ENV PYTHONUNBUFFERED 1

WORKDIR /deployment

COPY ./requirement.txt /deployment/django-platform/requirement.txt 
RUN pip install -r /deployment/django-platform/requirement.txt

COPY . /deployment

CMD ["python", "manage.py", "runserver"]


# FROM python:3
# FROM ubuntu:18.04
# ENV PYTHONUNBUFFERED 1

# WORKDIR /deployment

# ADD  . /deployment

# COPY ./requirement.txt  /deployment/django-platform/requirement.txt 

# RUN pip install h5py==3.1.0
# RUN pip install -r requirement.txt


# CMD [ "python", "manage.py", "runserver" ]
# COPY . /deployment