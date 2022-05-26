
FROM continuumio/anaconda3:latest

RUN conda install pytorch torchvision --channel pytorch
RUN mkdir -p /conda
WORKDIR /conda


# docker build --rm=true -t condatorch .