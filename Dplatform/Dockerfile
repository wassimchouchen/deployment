FROM python:3.9

ENV PYTHONUNBUFFERED 1

WORKDIR /deployment

COPY ./requirements.txt /deployment/django-platform/requirements.txt 
RUN pip install -r /deployment/django-platform/requirements.txt

COPY . /deployment

CMD ["python", "manage.py", "runserver"]


