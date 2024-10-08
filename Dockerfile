FROM python:3.10

ENV DockerHOME=/home/app/webapp  
  
RUN mkdir -p $DockerHOME  

WORKDIR $DockerHOME  
  
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1  
 
RUN pip install --upgrade pip  
 
COPY . $DockerHOME  
 
RUN pip install -r requirements.txt  
  
EXPOSE 8000 
 
CMD gunicorn --workers=4 text_summarizer.wsgi:application --bind 0.0.0.0:8000