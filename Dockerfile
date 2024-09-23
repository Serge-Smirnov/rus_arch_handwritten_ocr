FROM python:3.8.18

ENV APP_NAME hcr-engine
ENV TZ Europe/Moscow

# выставляем необходимый часовой пояс в системном окружении контейнера
RUN mv /etc/localtime /etc/localtime.bak \
	&& ln -s /usr/share/zoneinfo/${TZ} /etc/localtime

WORKDIR /app/

COPY /app/requirements.txt /app/requirements.txt \
     /app/*.py /app/ \
     /app/segmentation_model/*.py /app/segmentation_model/ \
     /app/translation_model/*.py /app/translation_model/ \
     /app/.env /app/

RUN /usr/local/bin/python -m pip install --upgrade pip &&\
    apt update &&\
    # для работы cv2 на debian 12 
    apt install libegl1 -y &&\
    apt install -y libgl1-mesa-glx -y &&\
    apt-get install wget -y &&\
    apt-get install unzip -y &&\
    python3 -m venv /app/venv &&\
    /app/venv/bin/pip3 install -r /app/requirements.txt

RUN /app/venv/bin/python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'



EXPOSE 8087

CMD ["/bin/bash", "-c", "while true; do foo; sleep 2; done & wait"]