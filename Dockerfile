# set base image (host OS)
FROM python:3.7

# set the working directory in the container
WORKDIR /usr/src/app

# install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src/ .

# command to run on container start
CMD [ "python", "./main.py" ]
