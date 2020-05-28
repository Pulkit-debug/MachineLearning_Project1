FROM python

RUN pip install -U pip

RUN pip install tensorflow
RUN pip install numpy matplotlib pandas seaborn keras
RUN pip install sklearn

RUN mkdir mlops

COPY . /mlops

ENTRYPOINT [ "python" ]

CMD [ "/mlops/programFile.py" ]
