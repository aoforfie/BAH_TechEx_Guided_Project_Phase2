FROM python:3.10-slim-buster
WORKDIR /app
COPY . /app
RUN pip install --no-cahe-dir -r requirements.txt
EXPOSE 8000
ENV NAME GuidedProject2
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
