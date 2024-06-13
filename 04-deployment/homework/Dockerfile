FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set working directory
WORKDIR /app

RUN pip install pandas==2.2.2
RUN pip install scikit-learn==1.5.0
RUN pip install pyarrow==15.0.2
RUN pip install cloudpickle==3.0.0


COPY [ "script.py", "./"]

# Run the script
CMD ["python", "script.py"]
