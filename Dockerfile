
FROM python:3.9-slim


RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*


ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64/

COPY . /app
WORKDIR /app


COPY requirements.txt .


RUN pip install --no-cache-dir -r requirements.txt


ARG SPARK_VERSION=3.4.0
ARG HADOOP_VERSION=3

RUN curl -O https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
    && tar xzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz -C /usr/local/ \
    && rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

ENV SPARK_HOME=/usr/local/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}
ENV PATH="$SPARK_HOME/bin:$PATH"
ENV PYTHONPATH="$SPARK_HOME/python:$PYTHONPATH"
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

COPY main.py .


CMD ["python", "main.py"]
