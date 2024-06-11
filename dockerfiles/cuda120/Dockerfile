FROM ubuntu:22.04

RUN apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
	python3 \
	python3-pip \
	curl && \
	apt clean && \
	rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

COPY ./src /workdir/src
WORKDIR /workdir

ENTRYPOINT [ "python3", "-m" ]
