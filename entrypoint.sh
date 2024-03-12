#!/bin/bash
set -eux

if [[ $1 = 'train' ]]
then
  python package_folder/train_model.py $2
fi

if [[ $1 = 'launch_api' ]]
then
  uvicorn package_folder.api_file:app --host 0.0.0.0 --port $PORT
fi

if [[ $1 = 'test' ]]
then
   echo 'hello world'
fi

exec "$@"