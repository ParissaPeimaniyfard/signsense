#!/bin/bash
set -eux

if [[ $1 = 'train' ]]
then
  exec python package_folder/train_model.py $2
fi

if [[ $1 = 'launch_api' ]]
then
  exec uvicorn package_folder.api_file:app --host 0.0.0.0 --port $PORT
fi

if [[ $1 = 'test' ]]
then
   exec echo 'hello world'
fi

exec "$@"