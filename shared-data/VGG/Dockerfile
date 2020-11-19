FROM ubuntu:18.04
WORKDIR /usr/bin

COPY preprocess1.py ./
RUN chmod 777 ./preprocess1.py

COPY Augmentation.py ./
RUN chmod 777 ./Augmentation.py

COPY Data_Split.py ./
RUN chmod 777 ./Data_Split.py

COPY VGG_model.py ./
RUN chmod 777 ./VGG_model.py

COPY hpo_checkpointing.py ./
RUN chmod 777 ./hpo_checkpointing.py

COPY Test.py ./
RUN chmod 777 ./Test.py


RUN apt-get update && apt install python3-pip wget curl -y
RUN python3 -m pip install --upgrade pip setuptools
RUN pip3 install \
        --upgrade pip \
        keras==2.1.5 \
        tensorflow==1.13.1 \
        tensorflow-gpu==1.15 \
        numpy \
        pandas \
        pillow \
        sklearn \
        optuna \
        scikit-image  \
        optkeras \
        h5py==2.10.0 
