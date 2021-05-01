FROM nvcr.io/nvidia/pytorch:21.04-py3
CMD nvidia-smi

WORKDIR /workspace

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

#RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
#WORKDIR ./apex
#ENV TORCH_CUDA_ARCH_LIST="compute capability"
#RUN pip install -v --disable-pip-version-check --no-cache-dir --use-feature=in-tree-build ./
#
#WORKDIR /workspace

RUN python -m spacy download en_core_web_lg

RUN apt-get install -y vim