FROM ironbcc/demo1

ADD . /root/demo1

WORKDIR /root/demo1
RUN pip install -r requirements.txt
CMD ["python", "app.py"]