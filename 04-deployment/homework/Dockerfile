FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Set working directory
WORKDIR /app

RUN pip install pandas
RUN pip install scikit-learn


COPY [ "mean.py", "./"]

# Run the script
CMD ["python", "mean.py"]
