# HYEEE Agent Service

HYEEE Agent Service 是一个 Python/FastAPI 服务，负责：

- 聊天接口与工具调用编排
- 知识库创建与文档上传
- 文档分块任务启动
- RocketMQ 分块消息消费
- 文档读取、Tika 文本提取、分块、Embedding、pgvector 持久化
- 静态前端页面服务

项目目前支持本地 mock 调试，也支持在 Linux/WSL/Docker 环境中接入真实 RocketMQ 消费。

## Quick Start

创建虚拟环境并安装依赖：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

复制环境变量：

```powershell
Copy-Item .env.example .env
```

启动服务：

```powershell
.\.venv\Scripts\python.exe -m agent_service.main
```

打开页面：

```text
http://127.0.0.1:8000/ui/chat.html
http://127.0.0.1:8000/ui/knowledge-base.html
http://127.0.0.1:8000/ui/knowledge-upload.html
```

健康检查：

```http
GET http://127.0.0.1:8000/health
```

运行测试：

```powershell
.\.venv\Scripts\python.exe -m ruff check agent_service tests
.\.venv\Scripts\python.exe -m pytest
```

## Environment

常用配置在 `.env.example` 中。

### Agent Model

本地默认 mock：

```env
AGENT_MOCK_MODE=true
```

使用真实 OpenAI-compatible Chat 模型时：

```env
AGENT_MOCK_MODE=false
AI_PROVIDER=openai
AI_BASE_URL=https://api.openai.com/v1
AI_MODEL=gpt-4o-mini
```

推荐把 key 放在本机环境变量，不写入 `.env`：

```powershell
setx AI_API_KEY "your_api_key"
```

### MCP Client

MCP is optional. The service starts and KB/general chat works normally with the default
`MCP_ENABLED=false`; no business MCP Server is required for local development.

After the business project exposes a Streamable HTTP MCP endpoint, configure the client here:

```env
MCP_ENABLED=true
MCP_SERVER_URL=http://business-service-host:8081/mcp
# MCP_SERVER_TOKEN=optional-bearer-token
MCP_TIMEOUT_SECONDS=10
MCP_TOOL_PREFIX=
MCP_FAIL_FAST=false
MCP_CONTEXT_MAX_CHARS=6000
```

At startup the client completes `initialize` and `tools/list`, then registers the discovered
tools. A connection/discovery failure is logged and does not prevent the Agent service from
starting when `MCP_FAIL_FAST=false`; MCP intent execution is then rendered as unavailable rather
than fabricated as real-time data.

### Distributed Stream Cancellation

Single-instance development needs no additional configuration. To coordinate stop requests across
multiple Agent instances, point every instance to the same Redis deployment:

```env
STREAM_CANCEL_REDIS_URL=redis://:password@redis-host:6379/0
STREAM_CANCEL_KEY_PREFIX=heyee:stream:cancel
STREAM_CANCEL_CHANNEL=heyee:stream:cancel
STREAM_CANCEL_TTL_SECONDS=1800
STREAM_CANCEL_MAX_TASKS=10000
```

The cancellation endpoint writes a TTL-protected Redis marker before publishing `taskId` through
Redis Pub/Sub. Every instance subscribes to the channel; the instance holding the LLM HTTP stream
finds its local cancellation handle and aborts that stream. Redis is not read per token: token
callbacks use a local cancellation flag, while Redis is checked only at task registration to
close the cancel-before-register race. A cancelled task keeps its local tombstone until the same
TTL expires, so delayed callbacks cannot resume delivery. If partial-answer persistence fails,
the error is logged but `cancel`/`done` still close the stream.

### Stream Admission Control

RAG requests are long-running streams, so a QPS limit alone cannot protect model concurrency.
The service therefore admits a bounded number of streams before `ChatService.prepare` begins:

```env
STREAM_QUEUE_ENABLED=true
# Leave empty for one-process local FIFO mode.
# Set the same Redis URL on every instance for cluster-wide fairness.
STREAM_QUEUE_REDIS_URL=redis://:password@redis-host:6379/0
STREAM_QUEUE_KEY_PREFIX=heyee:stream:queue
STREAM_QUEUE_MAX_CONCURRENT=3
STREAM_QUEUE_MAX_WAIT_SECONDS=20
STREAM_QUEUE_LEASE_SECONDS=600
STREAM_QUEUE_POLL_INTERVAL_MS=200
```

Without `STREAM_QUEUE_REDIS_URL`, development uses an in-process FIFO queue. With Redis, all
instances share a FIFO waiting ZSET, a monotonic sequence, per-request TTL entry markers, and an
expiring permit ZSET. A Lua claim script atomically removes stale queue entries, verifies that a
request is inside the first `max_concurrent` live positions, and allocates a lease. Redis Pub/Sub
wakes waiting instances after a release, while polling remains as a fallback. Permit leases are
renewed during a stream and recover automatically after a crashed worker.

If Redis is explicitly configured but unavailable, admission returns `queue_unavailable` rather
than silently falling back to per-instance limits. This prevents a multi-instance deployment from
over-admitting model streams. A queue timeout returns `queue_timeout`; a queued task can still be
cancelled immediately through the existing stream cancellation endpoint.

### Embedding

当前 Embedding 方案：硅基流动 OpenAI-compatible API + `BAAI/bge-m3`。

```env
EMBEDDING_PROVIDER=siliconflow
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIMENSION=1024
EMBEDDING_BATCH_SIZE=32
EMBEDDING_TIMEOUT_SECONDS=60
```

把硅基流动 key 放在本机环境变量：

```powershell
setx EMBEDDING_API_KEY "your_siliconflow_api_key"
```

### Tika

消费者通过 Tika Server 提取纯文本：

```env
TIKA_SERVER_URL=http://127.0.0.1:9998
TIKA_TIMEOUT_SECONDS=60
```

### RocketMQ

Windows Python 客户端不支持真实 RocketMQ 发送/消费，Windows 本机建议 mock 调试。
真实 RocketMQ consumer 建议跑在 Linux/WSL/Docker。

```env
ROCKETMQ_MOCK_MODE=true
ROCKETMQ_NAME_SERVER=127.0.0.1:9876
ROCKETMQ_PRODUCER_GROUP=heyee-agent-chunk-producer
ROCKETMQ_CONSUMER_GROUP=heyee-agent-chunk-consumer
ROCKETMQ_CHUNK_TOPIC=heyee-knowledge-document-chunk
ROCKETMQ_CHUNK_TAG=START_CHUNK
```

### Database And Object Storage

```env
DATABASE_URL=postgresql+psycopg://user:password@192.168.23.129:5432/ragent
RUSTFS_ENDPOINT=http://192.168.23.129:9000
RUSTFS_ACCESS_KEY=rustfsadmin
RUSTFS_SECRET_KEY=rustfsadmin
RUSTFS_REGION=us-east-1
```

## Main Flows

### Chat Flow

```text
frontend/chat.html
  -> POST /v1/agent/chat/stream (fetch + SSE)
  -> StreamChatService
  -> ChatService.prepare (memory / RAG / MCP / Prompt)
  -> StreamLLMService (model selection)
  -> OpenAI-compatible or Ollama upstream stream
  -> SSE meta/message/finish/done -> browser
```

`POST /v1/agent/chat` remains available for callers that require one complete JSON response.

### Streaming Chat Flow

The chat page uses `fetch()` to POST to `/v1/agent/chat/stream` and reads the response body as
SSE. This is intentionally not browser `EventSource`, because the request includes a JSON body.

```json
{
  "userId": 1,
  "conversationId": "optional-existing-conversation",
  "message": "查询本月销售数据",
  "deepThinking": false,
  "history": []
}
```

```text
browser fetch POST
  -> /v1/agent/chat/stream
  -> immediately receive meta(taskId, phase=queued)
  -> stream admission (local FIFO or Redis shared FIFO + expiring permit)
  -> meta(taskId, phase=preparing)
  -> ChatService.prepare
  -> parallel KB retrieval + MCP execution
  -> Prompt scene selection
  -> StreamLLMService
  -> upstream model SSE / JSON-line stream
  -> StreamChatService callback queue
  -> downstream SSE events
  -> persist assistant message and finish
```

The stream endpoint emits the following events:

| Event | Payload | Meaning |
| --- | --- | --- |
| `meta` | `taskId`, `phase=queued|preparing|answering`; `answering` also includes conversation and retrieval metadata | The client gets `taskId` before queue admission, allowing an early stop request. |
| `message` | `type=think|response`, `delta` | Incremental reasoning or answer text. |
| `finish` | `messageId`, `title`, `sources`, `toolCalls`, `ragIntent` | Assistant reply persisted successfully. |
| `error` | `message`, optional `detail` | Preparation or model streaming failed. |
| `cancel` | `taskId`, `messageId`, `title`, `partial` | The active stream was cancelled; non-empty generated text is persisted as an interrupted reply when storage is available. |
| `done` | `taskId` | Terminal event; clients should release the reader. |

`queue_timeout` and `queue_unavailable` are returned as `error.code` values before preparation;
no model request is started in either case.

Cancel a running stream with:

```http
POST /v1/agent/chat/stream/{taskId}/cancel
```

When the user clicks **停止**, the browser first calls the cancellation endpoint and keeps its
reader open for `cancel` and `done`; it falls back to aborting `fetch` only if that request fails.
The task manager keeps a local `taskId -> handle` registry for fast token-loop checks. When
`STREAM_CANCEL_REDIS_URL` is configured it also persists cancellation markers and subscribes to
Redis Pub/Sub, so stop requests can land on a different instance from the active LLM stream.

### Knowledge Upload Flow

```text
frontend/knowledge-upload.html
  -> POST /v1/knowledge-documents/upload
  -> KnowledgeDocumentService.upload_file/upload_url
  -> ObjectStorageService 上传 RustFS/S3
  -> t_knowledge_document 插入 PENDING 文档记录
```

### Start Chunk Flow

```text
frontend/knowledge-upload.html
  -> POST /v1/knowledge-documents/{docId}/chunks/start
  -> KnowledgeDocumentService.start_chunking
  -> RocketMQ 事务消息
  -> 本地事务回调内 CAS: PENDING -> RUNNING
  -> 消息提交给消费者
```

注意：CAS 更新必须在事务消息回调里。CAS 失败时消息不应投递。

### Chunk Consumer Flow

```text
RocketMQ/mock message
  -> KnowledgeChunkConsumer
  -> KnowledgeChunkPipeline
  -> 查 t_knowledge_document
  -> 写 t_knowledge_document_chunk_log RUNNING
  -> ObjectStorageService 读取 fileUrl
  -> TikaTextExtractor 提取纯文本
  -> ChunkingService 执行 fixed_size / structure_aware
  -> EmbeddingService 调 BAAI/bge-m3
  -> KnowledgeChunkRepository 原子写入 chunks + vectors + doc status
  -> 更新 chunk log SUCCESS/FAILED 和耗时
```

原子持久化包含五步，在同一个 PostgreSQL 事务里完成：

1. DELETE 旧 chunks
2. INSERT 新 chunks
3. DELETE 旧 vectors
4. INSERT 新 vectors
5. UPDATE 文档状态为 `SUCCESS` 并更新 `chunk_count`

## Mock Chunk Consumer

本地不接 RocketMQ 时，可以用 mock 接口调试完整 pipeline。
仅当 `AGENT_MOCK_MODE=true` 时注册：

```http
POST /v1/knowledge-documents/chunks/mock-consume
Content-Type: application/json
```

```json
{
  "docId": "your_doc_id",
  "messageId": "mock-message-id",
  "requestedBy": "local-dev"
}
```

前提：数据库里该文档状态应为 `RUNNING`，并且 `file_url` 能被当前服务读取。

## Real RocketMQ Consumer

在 Linux/WSL/Docker 中运行：

```powershell
.\.venv\Scripts\python.exe -m agent_service.consumers.run_knowledge_chunk_consumer
```

Linux 示例：

```bash
python -m agent_service.consumers.run_knowledge_chunk_consumer
```

消费者订阅：

```text
topic = ROCKETMQ_CHUNK_TOPIC
consumer group = ROCKETMQ_CONSUMER_GROUP
```

## Database SQL

`sql/20260628_add_chunk_log_message_id.sql`

- 给 `t_knowledge_document_chunk_log` 增加 `message_id varchar(64)`
- 增加 message_id 索引

`sql/20260628_create_knowledge_vector.sql`

- 创建 pgvector 扩展
- 创建 `t_knowledge_vector`
- 向量字段为 `vector(1024)`，匹配 `BAAI/bge-m3`
- 创建 doc/chunk/vector 索引

`sql/20260707_create_intent_node.sql`

- 创建 `t_intent_node` 意向树节点表
- 支持 `DOMAIN -> CATEGORY -> TOPIC` 三层结构
- 支持 `KB / SYSTEM / MCP` 节点类型
- 支持 KB 节点绑定 `collection_name/top_k`
- 支持 MCP 节点绑定 `mcp_tool_id`
- 支持 prompt 片段、节点样例、排序、启停和软删除字段

## Project Structure

```text
agent_service/       后端服务源码
frontend/            静态前端页面和本地 JS/CSS 资源
sql/                 数据库变更脚本
tests/               自动化测试
pyproject.toml       项目依赖、测试和 ruff 配置
.env.example         环境变量模板
README.md            项目说明
```

## Backend Files

### `agent_service/main.py`

FastAPI 应用入口。

- 创建 app
- 注册 CORS、中间件、异常处理
- 注册 API 路由
- 挂载 `/ui` 静态前端
- `python -m agent_service.main` 启动 HTTP 服务

### `agent_service/api/chat.py`

聊天 API。

- `POST /v1/agent/chat`：同步返回完整 JSON，兼容非流式调用方
- `POST /v1/agent/chat/stream`：SSE 流式输出 `meta/message/finish/done` 等事件
- `POST /v1/agent/chat/stream/{task_id}/cancel`：取消当前进程内的流式任务

### `agent_service/api/health.py`

健康检查 API。

- 暴露 `GET /health`
- 返回服务状态和版本信息

### `agent_service/api/knowledge.py`

知识库和文档 API。

- `GET /v1/knowledge-bases`：列出知识库
- `POST /v1/knowledge-bases`：创建知识库
- `POST /v1/knowledge-documents/upload`：上传本地文件或 URL 文档
- `POST /v1/knowledge-documents/{document_id}/chunks/start`：启动分块事务消息

### `agent_service/api/knowledge_chunk_dev.py`

开发用 mock 消费 API。

- `POST /v1/knowledge-documents/chunks/mock-consume`
- 只在 `AGENT_MOCK_MODE=true` 时由 `main.py` 注册
- 用于本机不接 RocketMQ 时调试消费者 pipeline

### `agent_service/core/config.py`

统一配置定义。

- 服务端口和 mock 开关
- Chat 模型配置
- Embedding 配置
- Tika Server 配置
- 数据库配置
- RustFS/S3 配置
- RocketMQ 配置
- 上传限流配置

### `agent_service/core/errors.py`

统一异常和异常处理器。

- `ModelUnavailableError`
- `MessageQueueUnavailableError`
- HTTPException 处理
- 未预期异常处理

### `agent_service/core/logging.py`

日志基础配置。

### `agent_service/db/session.py`

数据库 engine 工厂。

- 从 `DATABASE_URL` 创建 SQLAlchemy engine
- 使用 `pool_pre_ping=True`

## Services

### `agent_service/services/chat_service.py`

聊天业务服务。

- `prepare()` 统一执行会话记忆、RAG/MCP 检索、Prompt 组装等前置步骤
- 同步路径调用 `LLMService`，流式路径复用预处理结果
- 在完成时持久化 assistant 消息、更新会话标题和指标
- 用户停止且已有正文时，持久化部分回答并标记为 `interrupted`

### `agent_service/services/stream_chat_service.py`

流式聊天编排服务。

- 将预处理、模型增量回调、SSE 事件和最终持久化串成一条链路
- 输出 `meta`、`message`、`finish`、`error`、`cancel`、`done` 事件
- 在预处理前先注册并发送 `taskId`，覆盖“取消先到、任务后注册”的竞态
- 支持浏览器断连和显式取消；取消后保存非空部分回答并发送 `cancel(messageId)`
- 在 RAG 预处理前接入公平并发准入，排队期间也可取消

### `agent_service/services/stream_queue_limiter.py`

流式请求准入控制服务。

- 单实例使用本地 FIFO 队列，适合未配置 Redis 的开发环境
- 多实例使用 Redis FIFO ZSET、单调序号、entry TTL、可过期 permit 和 Pub/Sub 唤醒
- 通过 Lua 原子认领队首并发窗口，避免跨节点重复放行；许可租约自动续期和兜底回收

### `agent_service/services/stream_llm_service.py`

流式模型调用门面。

- 按模型选择器路由到 OpenAI-compatible 或 Ollama 客户端
- 将模型的思考片段和正文片段分别回调为 `think`、`response`
- 将流式成功/失败结果回写模型健康状态

### `agent_service/services/stream_task_manager.py`

流式任务管理器。

- 维护本地 `taskId -> cancellation handle` 映射，供 Token 回调低延迟检查
- 使用本地取消标志和句柄中断，避免将取消导致的 I/O 异常误报为模型错误
- 可选 Redis Key（TTL）+ Pub/Sub：先持久化取消标记、后广播，支持跨实例取消
- 注册时读取 Redis 标记、绑定句柄时再次检查本地状态，覆盖两个关键时序竞态

### `agent_service/services/llm_service.py`

大模型调用门面。

- mock 模式返回本地测试回复
- 真实模式调用 `FunctionCallService`

### `agent_service/services/function_call_service.py`

OpenAI-compatible tool/function calling 编排。

- 调用模型
- 解析 tool calls
- 执行工具
- 多轮工具循环

### `agent_service/services/prompt_service.py`

Prompt 模板服务。

- 按 `KB_ONLY`、`MCP_ONLY`、`MIXED` 和普通场景选择 System Prompt
- 将 KB 文档和 MCP 工具结果分别封装为 evidence，并与当前问题合并到最后一条 user 消息
- 支持意图节点 `promptTemplate` 覆盖和 `promptSnippet` 规则注入
- 为不同场景提供 temperature/top-p 参数建议

### `agent_service/services/knowledge_document_service.py`

知识库文档核心服务。

- 创建知识库
- 上传文件到对象存储
- 上传 URL 文件
- 校验 chunk 策略和配置
- 插入 `t_knowledge_document`
- 启动分块：发送 RocketMQ 事务消息，并在本地事务回调里 CAS 更新状态

### `agent_service/services/object_storage_service.py`

RustFS/S3 对象存储服务。

- 确保 bucket 存在
- 生成 presigned put URL
- 上传本地文件
- 上传异步流
- 根据 `s3://bucket/key` 下载文件 bytes

### `agent_service/services/rocketmq_transaction_producer.py`

RocketMQ 事务消息生产者。

- mock 模式：直接执行本地事务并校验消息体
- 真实模式：调用 `TransactionMQProducer`
- 本地事务失败时返回 rollback，避免消息投递
- Windows Python 客户端不可用时返回明确的消息队列异常

### `agent_service/services/document_object_reader.py`

文档对象读取服务。

- 接收 `fileUrl`
- 通过 `ObjectStorageService.download_file_url` 读取文件 bytes

### `agent_service/services/tika_text_extractor.py`

Tika Server 文本提取服务。

- 调用 `TIKA_SERVER_URL/tika`
- 上传文件 bytes
- 返回纯文本
- 空文本或 Tika 失败会抛错

### `agent_service/services/chunking_service.py`

分块服务。

- `fixed_size`：固定大小分块，支持 overlap、句末/换行边界回退、URL 断行和中文软换行修复
- `structure_aware`：Markdown 友好结构感知分块，识别标题、段落、代码块、图片/链接原子行，并按块边界打包

### `agent_service/services/embedding_service.py`

Embedding 服务。

- 调用 OpenAI-compatible `/embeddings`
- 默认模型 `BAAI/bge-m3`
- 校验向量维度为 `EMBEDDING_DIMENSION=1024`
- 按 `EMBEDDING_BATCH_SIZE` 批量请求

### `agent_service/services/knowledge_chunk_pipeline.py`

分块消费者核心 pipeline。

- 查文档
- 幂等处理 SUCCESS 文档
- 写 chunk log RUNNING
- 对对象读取、Tika、Embedding、持久化做重试
- 调用分块和 Embedding
- 原子持久化 chunks/vectors/doc 状态
- 更新 chunk log SUCCESS/FAILED
- 失败时更新文档状态 FAILED

## Consumers

### `agent_service/consumers/knowledge_chunk_consumer.py`

mock 消费器。

- 接收 dict 消息体
- 转换为 `KnowledgeChunkMessage`
- 调用 `KnowledgeChunkPipeline`

### `agent_service/consumers/rocketmq_knowledge_chunk_consumer.py`

真实 RocketMQ PushConsumer。

- 订阅 `ROCKETMQ_CHUNK_TOPIC`
- 消息体 JSON 反序列化
- 写入 message id
- 调用分块 pipeline
- 成功返回 `CONSUME_SUCCESS`
- 失败返回 `RECONSUME_LATER`

### `agent_service/consumers/run_knowledge_chunk_consumer.py`

消费者命令行入口。

- 初始化配置和日志
- 启动 `RocketMqKnowledgeChunkConsumer.start_forever()`

## Repositories

### `agent_service/repositories/knowledge_repository.py`

知识库和文档基础仓储。

- 查询知识库列表
- 按名称/collection 查询知识库
- 插入知识库
- 插入文档
- CAS 更新文档状态 `PENDING -> RUNNING`，并返回分块消息所需字段

### `agent_service/repositories/knowledge_chunk_repository.py`

分块消费者仓储。

- 查询待分块文档
- 插入 chunk log
- 更新 chunk log 成功/失败
- 更新文档 FAILED
- 原子替换 chunks 和 vectors，并更新文档 SUCCESS

## Schemas

### `agent_service/schemas/chat.py`

聊天 API 请求/响应模型。

### `agent_service/schemas/health.py`

健康检查响应模型。

### `agent_service/schemas/knowledge.py`

知识库和文档 API 模型。

- 知识库创建请求/响应
- 文档 URL 上传请求
- 文档上传结果
- 启动分块响应

### `agent_service/schemas/chunking.py`

分块消费者内部数据模型。

- `KnowledgeChunkMessage`
- `TextChunk`
- `VectorChunk`

## MCP And Tools

### `agent_service/mcp/contracts.py`

MCP 协议内部数据结构。

### `agent_service/mcp/http_client.py`

Streamable HTTP MCP 客户端。

- initialize
- list tools
- call tool
- close session

### `agent_service/mcp/adapter.py`

把远端 MCP tools 注册进内部工具注册表，并保留 JSON Schema 供参数提取使用。

### `agent_service/mcp/parameter_extractor.py`

基于 LLM 的 Schema 约束参数提取。

- 把用户问题与工具参数 Schema 转为参数提取 Prompt
- 清理 Markdown 围栏、只保留 Schema 白名单字段、做 JSON 类型转换和默认值回填
- 缺少必填参数时请求澄清，网络/解析异常时安全降级为默认参数

### `agent_service/mcp/execution.py`

MCP 意图执行编排。

- 对命中的 MCP 意图并发提取参数并调用对应注册工具
- 将成功、缺参、不可用和失败结果格式化为可审计的 MCP context
- 工具异常不会中断 KB 或普通聊天链路，也不会伪造实时数据

### `agent_service/tools/registry.py`

工具注册表。

- 注册工具
- 查询工具 schema
- 执行工具

### `agent_service/tools/builtin.py`

内置工具。

- 注册默认本地工具

## Middleware

### `agent_service/middleware/upload_rate_limit.py`

上传接口并发限流。

- 仅限制 `/v1/knowledge-documents/upload`
- 使用 Redis zset 信号量
- Redis 不可用时返回 503
- 无许可时返回 429

## RAG

### `agent_service/rag/schemas.py`

RAG 检索相关结构。

### `agent_service/rag/retriever.py`

RAG 检索入口占位/基础实现。

- `search()` 支持可选 `collection_name` 参数，为后续按知识库 Collection 定向检索预留入口
- `search_many()` 仍保持多子问题并行检索的基础行为

### RAG Query Rewrite

已完成：

- `RewriteResult` 表达原问题、改写问题和子问题列表
- LLM 查询改写和子问题拆分
- 改写失败时保留原问题兜底
- 历史消息选择支持轮次、总字符数和单条消息字符数预算控制
- JSON 返回清洗支持 Markdown code fence 容错

### RAG Intent Tree

当前已完成的意向树能力：

- 意向树数据模型：`IntentLevel`、`IntentKind`、`IntentNode`、`NodeScore`、`SubQuestionIntent`、`IntentGroup`、`GuidanceDecision`
- DB 仓储：从 `t_intent_node` 读取启用且未删除的平铺节点
- 平铺节点构树：按 `parent_code` 组装父子关系，填充 `full_path`，生成 `all_nodes/leaf_nodes/id_to_node`
- 孤儿节点兜底：父节点缺失时保留为根节点，避免节点在加载时丢失
- 可选缓存封装：`IntentTreeCache` 支持 Redis 不可用时降级
- Prompt 模板加载：`PromptTemplateLoader` 支持 `agent_service/prompts/*.st` 文件和 `{slot}` 替换
- LLM 意向分类：只序列化叶子节点，使用 key-value 文本而不是整树 JSON
- LLM 返回容错：支持 JSON 数组和 `{results: [...]}`，跳过未知 ID、缺 `id/score` 的结果，并按分数降序
- 子问题意向解析：每个子问题独立分类，单个子问题失败时降级为空列表
- 分数过滤和数量限制：支持 `rag_intent_min_score` 和 `rag_max_intent_count`
- 总量封顶：多子问题场景下使用“每个子问题保底 1 个 + 剩余额度按全局分数竞争”的策略
- 意向分组：支持 KB/MCP 分组和 SYSTEM-only 判断
- 歧义引导：单子问题、多 KB 品类且分数接近时，在检索前生成澄清选项
- 灰区 LLM 确认：分数比处于灰色区间时调用低温 LLM 做二次判断，失败时默认触发引导
- KB 定向检索封装：按 KB 意向节点的 `collection_name/top_k` 调用检索器

本项目已完成的 Chat/RAG 检索闭环：

- 主 Chat/RAG Pipeline：意向识别结果已串入 `ChatService`，改写后的问题和子问题驱动检索
- 真实 pgvector 定向检索：按意向节点的 `collection_name/top_k` 查询启用的知识库向量、文档和分块
- 多通道检索与后处理：支持意向定向、PostgreSQL 关键词、全局向量三通道；去重、Rerank、全局 `topK` 截断和阶段日志
- 最终上下文与来源：精排后的分块写入最终 Prompt，并以结构化 `sources` 返回和展示在聊天页

相比 `D:\Googledownload\ragent-main`，当前仍待完善的闭环：

- Redis/DB 加载闭环尚未完全接通：还没有统一的“Redis 优先、未命中 DB、写回缓存、CRUD 后清缓存”加载器
- MCP Client 链路已实现：可发现远端工具、按 JSON Schema 提取参数、调用工具并把结果写入 Prompt；但实际业务数据取决于另一个项目提供并配置 MCP Server，当前默认不配置 Server。
- SYSTEM 短路尚未接入：当前有 `is_system_only()` 判断，但还没有用 SYSTEM 节点模板直接返回
- 意向树管理端和 CRUD API 尚未完成：目前靠 SQL 初始化和仓储读取，尚无前端管理页、批量启停、缓存失效接口
- Elasticsearch/BM25 专用检索服务尚未接入：当前关键词通道使用 PostgreSQL 全文/词面检索作为无额外中间件的实现

P0：明天优先完善的 ragent-main 对齐项：

- P0-1：意向树加载闭环。补齐 Redis 优先读取、DB 未命中重建、写回 Redis、节点变更后清缓存；替换当前单纯进程内懒加载。
- P0-2：LLM 分类参数控制。扩展 `LLMService.complete()` 支持 options，让意向分类可指定 `temperature=0.1`、`top_p=0.3`、`thinking=false`，和普通聊天参数解耦。
- P0-3：分类 Prompt 细化。按 ragent-main 模板补充实体导向 vs 主题导向、默认只返回 1 个核心意向、多问题才多返回、空数组规则、评分标准和 MCP/SYSTEM 差异说明。
- P0-4：识别链路 trace。记录 rewrite、classify、resolve、guidance 各阶段耗时、失败原因、LLM 原始响应摘要、被跳过的未知 ID/缺字段结果，方便排查 bad case。
- P0-5：歧义引导补强。补 category/domain alias 匹配、用户显式品类命中判断、用户选择后的下一轮闭环，并把 guidance 状态写入会话上下文避免重复引导。
- P0-6：并发与降级控制。给多子问题意向分类增加并发上限、单任务超时、失败降级和日志，避免一次请求产生过多 LLM 并发。
- P0-7：端到端集成测试。补真实意向树样例数据 + mock LLM 的完整链路测试，覆盖改写 JSON、分类 JSON、歧义 JSON、空意向树、DB/缓存/LLM 失败兜底。

暂不纳入本轮 P0 的内容：

- SYSTEM 直接回复、Elasticsearch/BM25 专用服务，以及业务项目的 MCP Server 落地；KB pgvector 检索、Rerank、MCP Client、流式答案生成已实现。

相关文件：

- `agent_service/rag/intent_models.py`
- `agent_service/repositories/intent_node_repository.py`
- `agent_service/rag/intent_tree.py`
- `agent_service/rag/intent_tree_cache.py`
- `agent_service/rag/prompt_template_loader.py`
- `agent_service/rag/intent_classifier.py`
- `agent_service/rag/intent_resolver.py`
- `agent_service/rag/ambiguity_checker.py`
- `agent_service/rag/intent_guidance.py`
- `agent_service/rag/intent_directed_retriever.py`
- `agent_service/prompts/intent-classifier.st`
- `agent_service/prompts/guidance-ambiguity-check.st`
- `agent_service/prompts/guidance-prompt.st`

## Memory And Harness

### `agent_service/memory/history_builder.py`

聊天历史上下文构造。

### `agent_service/harness/runner.py`

本地运行/调试辅助入口。

## Frontend Files

### `frontend/chat.html`

聊天页面。

- 聊天输入
- 消息列表
- Markdown 渲染
- DOMPurify 清理 HTML
- 用 `fetch` 调用 `/v1/agent/chat/stream` 并解析 SSE 帧
- 增量渲染 `think` / `response` 内容，接收来源、工具调用和会话元数据
- 生成期间显示“停止”按钮，调用取消接口并中止浏览器请求

### `frontend/knowledge-base.html`

创建知识库页面。

- 填写知识库名称
- 填写 Embedding 模型，默认 `BAAI/bge-m3`
- 填写 collection 名称
- 调用 `/v1/knowledge-bases`

### `frontend/knowledge-upload.html`

文档上传页面。

- 选择知识库
- 上传本地文件或远程 URL
- 设置分块策略和配置
- 调用 `/v1/knowledge-documents/upload`
- 上传成功后显示 docId
- docId 存在后启用“开始分块”按钮
- 调用 `/v1/knowledge-documents/{docId}/chunks/start`

### `frontend/css/chat.css`

聊天页面样式。

### `frontend/css/knowledge-upload.css`

知识库创建和上传页面共用样式。

### `frontend/css/main.css`

全局基础样式。

### `frontend/css/element.css` and `frontend/css/fonts/*`

Element UI 本地样式和字体。

### `frontend/js/vue.js`

Vue 运行时。

### `frontend/js/axios.min.js`

HTTP 请求库。

### `frontend/js/element.js`

Element UI 组件库。

### `frontend/js/marked.umd.js`

Markdown 渲染库。

### `frontend/js/purify.min.js`

HTML 清理库，防止不安全内容进入页面。

## Tests

### `tests/conftest.py`

测试公共配置。

- 默认启用 mock 模型
- 关闭 chunk pipeline 重试，避免测试变慢
- 每次测试清理 settings cache

### `tests/test_chat_api.py`

聊天 API 测试。

### `tests/test_chunking_service.py`

分块策略测试。

- fixed_size 边界对齐
- 中文软换行修复
- structure_aware 保持代码块完整
- structure_aware 按块边界切分

### `tests/test_frontend.py`

前端静态页面和资源测试。

### `tests/test_function_call.py`

Function calling 编排测试。

### `tests/test_health.py`

健康检查测试。

### `tests/test_knowledge_upload.py`

知识库创建、上传页面、分块启动事务逻辑测试。

### `tests/test_knowledge_chunk_pipeline.py`

mock 消费者 pipeline 测试。

- 成功链路
- 失败时更新日志和文档状态

### `tests/test_mcp_adapter.py`

MCP tool adapter 测试。

### `tests/test_mcp_http_client.py`

MCP HTTP client 测试。

### `tests/test_prompt_service.py`

Prompt 服务测试。

### `tests/test_stream_chat_service.py`

流式聊天链路测试。

- 正常输出 `meta`、思考/正文增量、`finish` 和 `done`
- 取消后输出 `cancel/done`，并持久化非空部分回答为已中断消息
- 覆盖取消先到、任务后注册的 Redis 标记兜底竞态
- 模型异常时输出 `error/done` 并记录失败状态

### `tests/test_upload_rate_limit.py`

上传并发限流测试。

## Development Notes

### Windows 本机调试 RocketMQ

`rocketmq-client-python` 不支持 Windows 真连接 RocketMQ。

Windows 本机建议：

- `ROCKETMQ_MOCK_MODE=true`
- 使用 `/v1/knowledge-documents/chunks/mock-consume` 调试消费者 pipeline

真实 RocketMQ 发送/消费建议：

- WSL
- Linux
- Docker

### API Key

不要把真实 key 写进仓库。

PowerShell 当前终端：

```powershell
$env:EMBEDDING_API_KEY="your_siliconflow_api_key"
```

持久用户环境变量：

```powershell
[Environment]::SetEnvironmentVariable("EMBEDDING_API_KEY", "your_siliconflow_api_key", "User")
```

设置后重启终端或 IDE。
