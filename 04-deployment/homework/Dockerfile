FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv 

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

# copy in CLI script
WORKDIR /app
COPY predict.py .

# default entrypoint: run May 2023
ENTRYPOINT ["python", "predict.py", "--year", "2023", "--month", "5"]