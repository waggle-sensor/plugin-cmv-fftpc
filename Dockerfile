FROM waggle/plugin-base:1.1.1-base

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt
RUN pip3 install --no-cache-dir --upgrade git+https://github.com/waggle-sensor/pywaggle # buildkit
COPY app.py cmv.py inf.py /app/
COPY test-data/sgptsimovieS01.a1.20160726.000000.mpg /app/test-data/

#ARG SAGE_STORE_URL="https://osn.sagecontinuum.org"
#ARG BUCKET_ID_MODEL="cafb2b6a-8e1d-47c0-841f-3cad27737698"

#ENV SAGE_STORE_URL=${SAGE_STORE_URL} \
#    BUCKET_ID_MODEL=${BUCKET_ID_MODEL}

#RUN sage-cli.py storage files download ${BUCKET_ID_MODEL} model640.pt --target /app/model640.pt \
#  && sage-cli.py storage files download ${BUCKET_ID_MODEL} yolov4.cfg --target /app/yolov4.cfg \
#  && sage-cli.py storage files download ${BUCKET_ID_MODEL} yolov4.weights --target /app/yolov4.weights

RUN chmod +x /app/app.py

WORKDIR /app
ENTRYPOINT ["python3", "/app/app.py"]
#ENTRYPOINT ["/bin/bash"]
