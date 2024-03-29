FROM python:3.8.12-slim

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY package_folder package_folder

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y && rm -rf /var/lib/apt/lists/*

# RUN CONTAINER LOCALLY
#CMD uvicorn package_folder.api_file:app --host 0.0.0.0

# RUN CONTAINER DEPLOYED
ENTRYPOINT ["entrypoint.sh"]
CMD ["test"]
