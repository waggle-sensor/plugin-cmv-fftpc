FROM waggle/plugin-base:1.1.1-base

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt
COPY test-data/sage-sgp.mov /app/test-data/

COPY app.py inf.py /app/

WORKDIR /app
ENTRYPOINT ["python3", "/app/app.py"]
