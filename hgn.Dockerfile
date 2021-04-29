FROM python:3.8

COPY configs ./hgn/configs
COPY csr_mhqa ./hgn/csr_mhqa
COPY eval ./hgn/eval
COPY models ./hgn/models
COPY scripts ./hgn/scripts
COPY utils ./hgn/utils

COPY envs.py ./hgn/envs.py
COPY model_envs.py ./hgn/model_envs.py
COPY predict.py ./hgn/predict.py
COPY train.py ./hgn/train.py
COPY run.sh ./hgn/run.sh

COPY requirements.txt ./hgn/requirements.txt
RUN pip install --no-cache-dir -r ./hgn/requirements.txt

RUN python -m spacy download en_core_web_lg

