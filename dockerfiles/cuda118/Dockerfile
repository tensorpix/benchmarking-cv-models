FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

RUN apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
	python3 \
	curl && \
	apt clean && \
	rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
	python3 /tmp/get-pip.py

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

COPY ./src /workdir/src
WORKDIR /workdir

ENTRYPOINT [ "python3", "-m", "src.train" ]
