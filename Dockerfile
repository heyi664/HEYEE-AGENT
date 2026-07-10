FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc g++ make curl ca-certificates tar \
       libstdc++6 libcurl4 openssl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fSL https://archive.apache.org/dist/rocketmq/rocketmq-client-cpp/2.1.0/rocketmq-client-cpp-2.1.0-bin-release.tar.gz -o /tmp/rocketmq-client-cpp.tar.gz \
    && mkdir -p /tmp/rocketmq-client-cpp /tmp/rocketmq-client-cpp-inner \
    && tar -xzf /tmp/rocketmq-client-cpp.tar.gz -C /tmp/rocketmq-client-cpp --strip-components=1 \
    && tar -xzf /tmp/rocketmq-client-cpp/centos7/rocketmq-client-cpp-2.1.0-bin-release.tar.gz -C /tmp/rocketmq-client-cpp-inner --strip-components=1 \
    && find /tmp/rocketmq-client-cpp-inner -name 'librocketmq.so*' -exec cp -a {} /usr/local/lib/ \; \
    && ldconfig \
    && rm -rf /tmp/rocketmq-client-cpp /tmp/rocketmq-client-cpp-inner /tmp/rocketmq-client-cpp.tar.gz

COPY pyproject.toml README.md ./
COPY agent_service ./agent_service
COPY frontend ./frontend

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

RUN python -c "from rocketmq.client import PushConsumer, TransactionMQProducer; print('rocketmq client ok')"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "agent_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
