FROM opensciencegrid/tensorflow-gpu:2.3-cuda-10.1
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


RUN apt-get update && apt install wget curl -y
RUN pip3 install \
    --upgrade pip \
    keras==2.4.3 \
    optuna \
    scikit-image  \
    optkeras \
    h5py==2.10.0 
