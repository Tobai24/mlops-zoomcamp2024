FROM python:3.12.3

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "predict.py", "project", "./"]

EXPOSE 9694
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9694", "predict:app" ]
