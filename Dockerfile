FROM tensorflow/tensorflow:2.10.0

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
COPY package_folder package_folder

# RUN CONTAINER LOCALLY
#CMD uvicorn package_folder.api_file:app --host 0.0.0.0

# RUN CONTAINER DEPLOYED
ENTRYPOINT ["entrypoint.sh"]
CMD ["test"]