# 🧠 Core Concepts

This document introduces the core concepts of the Wegent platform to help you understand each component and their relationships.

---

## ⚠️ Terminology Note: Team vs Bot

> **Important:** Please note the distinction between code-level terminology and user interface display names.

| Code/CRD Level | UI Display | Description |
|----------------|------------|-------------|
| **Team** | **Agent** | The user-facing AI agent that executes tasks |
| **Bot** | **Bot** | A building block component that makes up a Team |

**Simple Understanding:**
- **Bot** = A configured AI worker unit (includes prompt, runtime, model)
- **Team** = A "working team" composed of one or more Bots - this is what users interact with to execute tasks

---

## 📋 Concept Overview

Wegent is built on Kubernetes-style declarative API and CRD (Custom Resource Definition) design patterns, providing a standardized framework for creating and managing AI agent ecosystems.

### Core Resource Types

| Icon | Code Name | Description | Analogy |
|------|-----------|-------------|---------|
| 👻 | **Ghost** | The "soul" of an agent | Defines personality and capabilities |
| 🧠 | **Model** | AI model configuration | Brain configuration parameters |
| 🐚 | **Shell** | Runtime environment | Executable program container |
| 🤖 | **Bot** | Agent building block | Ghost + Shell + Model |
| 👥 | **Team** | User-facing agent | Combination of multiple Bots |
| 🤝 | **Collaboration** | Collaboration mode | Interaction pattern between Bots |
| 💼 | **Workspace** | Work environment | Isolated code workspace |
| 🎯 | **Task** | Task | Work unit assigned to a Team |
| 📚 | **KnowledgeBase** | Knowledge repository | RAG document storage |
| 🔍 | **Retriever** | Storage backend | Vector database configuration |
| ⚡ | **Skill** | On-demand capability | Dynamically loaded tools |

---

## 👻 Ghost - Soul of the Agent

Ghost represents the "soul" of an agent, defining its personality, capabilities, and behavior patterns.

### YAML Configuration Example

```yaml
apiVersion: agent.wecode.io/v1
kind: Ghost
metadata:
  name: developer-ghost
  namespace: default
spec:
  systemPrompt: "You are a professional software developer, skilled in using TypeScript and React to develop frontend applications."
  mcpServers:
    github:
      env:
        GITHUB_PERSONAL_ACCESS_TOKEN: ghp_xxxxx
      command: docker
      args:
        - run
        - -i
        - --rm
        - -e
        - GITHUB_PERSONAL_ACCESS_TOKEN
        - ghcr.io/github/github-mcp-server
status:
  state: "Available"
```

---

## 🧠 Model - AI Model Configuration

Model defines AI model configuration, including environment variables and model parameters.

### Model Types

Wegent supports multiple model types for different AI capabilities:

| Model Type | Description | Use Case |
|------------|-------------|----------|
| `llm` | Large Language Model (default) | Chat, code generation, reasoning |
| `embedding` | Embedding Model | Text vectorization for RAG |

### API Formats

| Format | Description | Recommended For |
|--------|-------------|-----------------|
| `chat/completions` | Traditional chat API (default) | General use |
| `responses` | New responses API | Agent scenarios (OpenAI only) |

### YAML Configuration Example

```yaml
apiVersion: agent.wecode.io/v1
kind: Model
metadata:
  name: claude-model
  namespace: default
spec:
  modelType: llm                        # Model type (default: llm)
  apiFormat: chat/completions           # API format (default: chat/completions)
  contextWindow: 200000                 # Max context window size
  maxOutputTokens: 8192                 # Max output tokens
  modelConfig:
    env:
      ANTHROPIC_MODEL: "openrouter,anthropic/claude-sonnet-4"
      ANTHROPIC_AUTH_TOKEN: "sk-xxxxxx"
      ANTHROPIC_BASE_URL: "http://xxxxx"
      ANTHROPIC_DEFAULT_HAIKU_MODEL: "openrouter,anthropic/claude-haiku-4.5"
status:
  state: "Available"
```

### Specialized Model Examples

```yaml
# Embedding Model Configuration
apiVersion: agent.wecode.io/v1
kind: Model
metadata:
  name: embedding-model
spec:
  modelType: embedding
  embeddingConfig:
    dimensions: 1536
    encoding_format: "float"
  modelConfig:
    env:
      OPENAI_API_KEY: "sk-xxx"
```

---

## 🐚 Shell - Runtime Environment

Shell is the container where agents run, specifying the runtime environment.

### Shell Types

| Type | Execution | Description |
|------|-----------|-------------|
| `ClaudeCode` | local_engine | Claude Code SDK, supports code execution |
| `Agno` | local_engine | Agno framework, team collaboration |
| `Chat` | local_engine | Lightweight chat Shell, LangGraph architecture |
| `Dify` | external_api | External Dify API proxy |

### YAML Configuration Example

```yaml
apiVersion: agent.wecode.io/v1
kind: Shell
metadata:
  name: claude-shell
  namespace: default
  labels:
    type: "local_engine"               # Execution type label
spec:
  runtime: "ClaudeCode"
  supportModel:
    - "openai"
    - "anthropic"
status:
  state: "Available"
```

---

## 🤖 Bot - Complete Agent Instance

Bot is a complete agent instance combining Ghost (soul), Shell (container), and Model (configuration).

### YAML Configuration Example

```yaml
apiVersion: agent.wecode.io/v1
kind: Bot
metadata:
  name: developer-bot
  namespace: default
spec:
  ghostRef:
    name: developer-ghost
    namespace: default
  shellRef:
    name: claude-shell
    namespace: default
  modelRef:
    name: claude-model
    namespace: default
status:
  state: "Available"
```

---

## 👥 Team - Collaborative Team

Team defines a collection of Bots working together with specific roles and collaboration patterns.

### YAML Configuration Example

```yaml
apiVersion: agent.wecode.io/v1
kind: Team
metadata:
  name: dev-team
  namespace: default
spec:
  members:
    - name: "developer"
      botRef:
        name: developer-bot
        namespace: default
      prompt: "You are the developer in the team, responsible for implementing features..."
      role: "leader"
    - name: "reviewer"
      botRef:
        name: reviewer-bot
        namespace: default
      prompt: "You are the code reviewer in the team, responsible for reviewing code quality..."
      role: "member"
  collaborationModel: "pipeline"
status:
  state: "Available"
```

---

## 🤝 Collaboration Models

Four collaboration patterns define how Bots interact within a Team:

### 1. **Pipeline**
Sequential execution where each Bot's output feeds into the next.
```
Developer Bot → Reviewer Bot → Tester Bot → Deployer Bot
```

### 2. **Route**
Leader assigns tasks to appropriate Bots based on content.
```
User Query → Leader Bot → {Frontend Bot | Backend Bot | DB Bot}
```

### 3. **Coordinate**
Leader coordinates parallel Bot execution and aggregates results.
```
Leader Bot → [Analyst Bot, Data Bot, Report Bot] → Leader Bot (aggregate)
```

### 4. **Collaborate**
All Bots share context and freely discuss.
```
[Bot A ↔ Bot B ↔ Bot C] (shared context)
```

---

## 💼 Workspace - Work Environment

Workspace defines the team's work environment, including repository and branch information.

### YAML Configuration Example

```yaml
apiVersion: agent.wecode.io/v1
kind: Workspace
metadata:
  name: project-workspace
  namespace: default
spec:
  repository:
    gitUrl: "https://github.com/user/repo.git"
    gitRepo: "user/repo"
    gitRepoId: 12345
    branchName: "main"
    gitDomain: "github.com"
status:
  state: "Available"
```

---

## 🎯 Task - Executable Work Unit

Task is an executable work unit assigned to a Team, associating Team and Workspace.

### YAML Configuration Example

```yaml
apiVersion: agent.wecode.io/v1
kind: Task
metadata:
  name: implement-feature
  namespace: default
spec:
  title: "Implement new feature"
  prompt: "Please implement a user authentication feature with JWT tokens"
  teamRef:
    name: dev-team
    namespace: default
  workspaceRef:
    name: project-workspace
    namespace: default
status:
  state: "Available"
  status: "PENDING"
  progress: 0
```

---

## 🔄 Concept Relationship Diagram

```mermaid
graph LR
    Ghost["👻 Ghost"] --> Bot["🤖 Bot"]
    Model["🧠 Model"] --> Bot
    Shell["🐚 Shell"] --> Bot
    Bot --> Team["👥 Team"]
    Collaboration["🤝 Collaboration"] --> Team
    Team --> TeamInstance["👥 Team Instance"]
    Workspace["💼 Workspace"] --> TeamInstance
    User["👤 User"] --> Task["🎯 Task"]
    Task --> TeamInstance
    TeamInstance --> Task
```

---

## 💡 Best Practices

### 1. Ghost Design
- ✅ Clearly define the agent's expertise
- ✅ Provide clear behavioral guidelines
- ✅ Configure necessary MCP tools

### 2. Bot Composition
- ✅ Create specialized Bots for different tasks
- ✅ Reuse Ghost and Model configurations
- ✅ Choose appropriate Shell types

### 3. Team Building
- ✅ Select suitable collaboration models
- ✅ Define clear member roles
- ✅ Provide clear task prompts for each member

---

## 🔗 Related Resources

- [YAML Specification](../reference/yaml-specification.md) - Complete YAML configuration format
- [Collaboration Models](./collaboration-models.md) - Detailed explanation of collaboration patterns
- [Creating Bots](../guides/user/creating-bots.md) - How to create and configure Bots
- [Creating Teams](../guides/user/creating-teams.md) - How to build collaborative teams
- [Skill System](./skill-system.md) - On-demand capabilities and tools

---

## 📚 KnowledgeBase - Document Storage for RAG

KnowledgeBase is a CRD that manages document collections for Retrieval-Augmented Generation (RAG).

### YAML Configuration Example

```yaml
apiVersion: agent.wecode.io/v1
kind: KnowledgeBase
metadata:
  name: my-knowledge-base
  namespace: default
spec:
  name: "Project Documentation"
  description: "Technical documentation for the project"
  document_count: 0                     # Cached document count
  retrievalConfig:
    retriever_name: my-retriever        # Retriever reference
    retriever_namespace: default
    embedding_config:
      model_name: text-embedding-3-small
      model_namespace: default
    retrieval_mode: hybrid              # vector | keyword | hybrid
    top_k: 5                            # Results to return (1-10)
    score_threshold: 0.7                # Score threshold (0.0-1.0)
    hybrid_weights:
      vector_weight: 0.7
      keyword_weight: 0.3
status:
  state: "Available"
```

### Key Features

- **Document Management**: Upload, index, and manage documents
- **Multiple Retrieval Modes**: Vector, keyword, or hybrid search
- **Embedding Integration**: Configure embedding models for vectorization
- **Configurable Weights**: Fine-tune hybrid search balance

---

## 🔍 Retriever - Vector Storage Backend

Retriever is a CRD that configures the vector storage backend for RAG functionality.

### Supported Storage Types

| Type | Description |
|------|-------------|
| `elasticsearch` | Elasticsearch with dense vectors |
| `qdrant` | Qdrant vector database |

### Index Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `fixed` | Single fixed index | Small datasets |
| `rolling` | Hash-based sharding | Large datasets |
| `per_dataset` | One index per knowledge base | Multi-tenant isolation |
| `per_user` | One index per user | User-level isolation |

### YAML Configuration Example

```yaml
apiVersion: agent.wecode.io/v1
kind: Retriever
metadata:
  name: my-retriever
  namespace: default
spec:
  storageConfig:
    type: elasticsearch                 # elasticsearch | qdrant
    url: "http://elasticsearch:9200"
    username: "elastic"
    password: "password"                # Optional
    apiKey: "api-key"                   # Optional
    indexStrategy:
      mode: per_user                    # Index strategy
      prefix: "wegent"                  # Index prefix
  retrievalMethods:
    vector:
      enabled: true
      defaultWeight: 0.7
    keyword:
      enabled: true
      defaultWeight: 0.3
    hybrid:
      enabled: true
  description: "Elasticsearch retriever for RAG"
status:
  state: "Available"
```

### Best Practices

- Use `per_user` mode for Elasticsearch deployments
- Enable hybrid search for better retrieval quality
- Set appropriate score thresholds to filter irrelevant results

---

<p align="center">Understanding these core concepts is fundamental to using Wegent! 🚀</p>
