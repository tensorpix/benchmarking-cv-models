FROM nvcr.io/nvidia/pytorch:23.10-py3

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

COPY src/ /workdir/src
WORKDIR /workdir/src

ENTRYPOINT [ "python3", "-m", "src.train" ]
