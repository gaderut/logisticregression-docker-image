# set base image (host OS)
FROM python:3.7-slim as base
                                       
# Pull small size base image
FROM base as builder
                                                     
# Create libraries install layer
RUN mkdir /install
WORKDIR /install

# install dependencies
COPY requirements.txt /requirements.txt                 
RUN pip install --prefix=/install -r /requirements.txt

# start from base layer
FROM base

# Copy installed libraries                                                                   
COPY --from=builder /install /usr/local

# Copy actual application
COPY . /app                                                                     
WORKDIR /app

# command to run on container start
CMD [ "python3", "main.py" ]                                      
