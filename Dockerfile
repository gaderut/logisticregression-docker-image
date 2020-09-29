# set base image (host OS)
FROM python:3.7

# set the working directory in the container
RUN mkdir /lgr
WORKDIR /lgr

# install dependencies
COPY requirements.txt /lgr
RUN pip install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY . /lgr

# command to run on container start
CMD [ "python", "main.py" ]

