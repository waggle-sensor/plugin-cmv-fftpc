FROM waggle/plugin-base:1.1.1-base

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt
COPY app.py inf.py /app/
COPY test-data/sgptsimovieS01.a1.20160726.000000.mpg /app/test-data/

WORKDIR /app
ENTRYPOINT ["python3", "/app/app.py"]
