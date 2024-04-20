# https://github.com/allenai/docker-images
# https://github.com/allenai/docker-images/pkgs/container/cuda/24038895?tag=11.2-ubuntu20.04-v0.0.15
FROM ghcr.io/allenai/cuda:11.2-ubuntu20.04-v0.0.15

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install --no-install-recommends --assume-yes \
      protobuf-compiler

# Install transformers
RUN conda install pytorch=1.12.0 cudatoolkit=11.3 -c pytorch # needed for cuda11.3
# This is 4.26.0.dev0 (needed the latest version for flan related fix)
RUN pip install git+https://github.com/huggingface/transformers.git@8637316e5e94ba0a2493e5df7846f2f23f46eaef
RUN pip install accelerate==0.15.0
RUN pip install -i https://test.pypi.org/simple/ bitsandbytes-cuda115
RUN pip install sentencepiece
RUN pip install protobuf==3.20.1 # needed to avoid error.

# Skipping deepspeed for now as it's not getting install correctly.
# RUN DS_BUILD_OPS=1 pip install git+https://github.com/microsoft/DeepSpeed.git@d9b788d773ce97281ee63064cc99993cb82397e2

RUN pip install fastapi
RUN pip install "uvicorn[standard]"

COPY serve_models /run/serve_models/
COPY constants.py /run/constants.py

# To run the server directly:
ENTRYPOINT ["uvicorn", "serve_models.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/run/"]

# To run bash:
# ENTRYPOINT ["bash", "-l"]
