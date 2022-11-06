FROM python:3.8 
WORKDIR /late-residual

COPY requirements.txt requirements.txt
COPY Code Code
COPY experiment_vars.yml experiment_vars.yml

RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

CMD ["python3", "Code/run_experiment_script.py"]
