# Relatório Técnico: Implementação de agentes e extração de dados com DSPy e Ollama

## 1. Visão Geral

O repositório contém uma prova de conceito desenvolvida em python, focada na construção de fluxos de inteligência artificial generativa para automação de atendimento ao cliente. O projeto utiliza o framework DSPy para orquestração de prompts e ferramentas, integrado a um modelo de linguagem local (Llama 3.1 8B) executado via Ollama. 

## 2. Componentes Desenvolvidos

### 2.1. Configuração do Modelo (config.py)
A base do projeto estabelece a conexão com o modelo de linguagem. Foi configurado um cliente apontando para uma instância local do Ollama executando o modelo `llama3.1:8b`. Essa abordagem garante que o processamento seja feito localmente, o que elimina custos recorrentes de APIs externas e mantém a estrita privacidade dos dados analisados.

### 2.2. Extração de Dados Estruturados (01_extraction.py)
Este módulo demonstra como transformar textos desestruturados, como e-mails de suporte, em dados estruturados e tipados.
- **Modelagem de Dados:** Utiliza a biblioteca Pydantic para definir a estrutura de saída exata (nome do cliente, problema relatado, escala de urgência de 1 a 10 e sentimento da mensagem).
- **Estratégias de Inferência:** O código compara a eficácia de duas assinaturas nativas do DSPy: o `dspy.Predict` (que gera a resposta diretamente) e o `dspy.ChainOfThought` (que exige que o modelo produza uma linha de raciocínio lógico antes de fornecer a classificação estruturada). O processamento é feito sobre um arquivo JSON com base em exemplos de e-mails reais de suporte.

### 2.3. Agente Autônomo com Tool Use (02_agent.py)
Este módulo implementa um agente conversacional desenhado para buscar informações ativamente e responder a dúvidas.
- **Ferramenta (Tool):** Foi desenvolvida a função `search_order`, que simula uma consulta a um banco de dados para recuperar o status de pedidos de clientes (entregue, cancelado, processando, etc.).
- **Orquestração ReAct:** Através da classe `dspy.ReAct`, o agente recebe perguntas em linguagem natural (ex: "Onde está o meu pedido PED_001 e PED_002?"). Ele é capaz de raciocinar sobre a pergunta, acionar a ferramenta de busca de forma autônoma passando os IDs corretos, analisar a resposta simulada do banco de dados e, por fim, compor uma resposta educada e final para o cliente.

## 3. Implementação em Software de Larga Escala

Para evoluir este laboratório para um ambiente de produção robusto e corporativo, as seguintes alterações de arquitetura e engenharia são recomendadas:

### 3.1. Infraestrutura e Inferência Distribuída
Em vez do Ollama local, uma arquitetura de escala exige que o modelo seja hospedado em clusters conteinerizados usando tecnologias focadas em alta vazão, como vLLM ou NVIDIA Triton Inference Server. Essas ferramentas permitem agrupamento contínuo de requisições (Continuous Batching), maximizando o uso de múltiplas GPUs. O cliente DSPy deverá apontar para o balanceador de carga destes clusters.

### 3.2. Ingestão Assíncrona via Mensageria
A rotina de extração de tickets de suporte não deve ser executada em lote local. O sistema deve operar de forma orientada a eventos: todo e-mail recebido é encaminhado para uma fila de mensagens (como Apache Kafka, RabbitMQ ou AWS SQS). Consumidores independentes retiram os e-mails da fila, executam a inferência via DSPy para extrair o nível de urgência e o sentimento, e disparam ações no sistema de CRM (como Zendesk ou Salesforce) para priorização automática.

### 3.3. Integração com Microsserviços Reais
A ferramenta de consulta de pedidos no agente deve substituir o dicionário estático por integrações reais via HTTP/REST ou gRPC com os sistemas transacionais da empresa (logística, faturamento, etc.). Nessa camada, é imprescindível aplicar padrões de resiliência, como Circuit Breakers e lógicas de tentativa (Retry), para assegurar que a inatividade momentânea de um sistema de rastreamento não cause falhas crônicas no agente conversacional.

### 3.4. Otimização Avançada com DSPy (Teleprompters)
Para operar em escala de forma previsível, a maior vantagem do DSPy deve ser ativada: os otimizadores automáticos. Em produção, constrói-se um conjunto de dados de validação. Ferramentas do DSPy, como o `BootstrapFewShotWithRandomSearch` ou `MIPRO`, são usadas durante o processo de build do software para "compilar" o agente. O framework ajustará automaticamente os prompts internos e injetará os melhores exemplos (few-shots) no contexto, garantindo alta precisão nas respostas de forma sistêmica, sem depender de "engenharia de prompt" manual.

### 3.5. Gestão de Estado e Observabilidade
O agente atual lida com instruções únicas (stateless). Para sustentar diálogos prolongados (como em um bot de WhatsApp), deve-se implementar um banco de dados em memória de baixa latência (ex: Redis) para manter o contexto conversacional da sessão. Além disso, a telemetria é fundamental: toda execução do DSPy deve enviar logs para plataformas de observabilidade de LLM (como Langfuse ou Datadog), permitindo rastrear custos de tokens, taxa de sucesso do uso de ferramentas, latência e permitindo auditorias rápidas no raciocínio do modelo.
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [Relatório Técnico: Implementação de agentes e extração de dados com DSPy e Ollama](#relatório-técnico-implementação-de-agentes-e-extração-de-dados-com-dspy-e-ollama)
  - [1. Visão Geral](#1-visão-geral)
  - [2. Componentes Desenvolvidos](#2-componentes-desenvolvidos)
    - [2.1. Configuração do Modelo (config.py)](#21-configuração-do-modelo-configpy)
    - [2.2. Extração de Dados Estruturados (01_extraction.py)](#22-extração-de-dados-estruturados-01_extractionpy)
    - [2.3. Agente Autônomo com Tool Use (02_agent.py)](#23-agente-autônomo-com-tool-use-02_agentpy)
  - [3. Implementação em Software de Larga Escala](#3-implementação-em-software-de-larga-escala)
    - [3.1. Infraestrutura e Inferência Distribuída](#31-infraestrutura-e-inferência-distribuída)
    - [3.2. Ingestão Assíncrona via Mensageria](#32-ingestão-assíncrona-via-mensageria)
    - [3.3. Integração com Microsserviços Reais](#33-integração-com-microsserviços-reais)
    - [3.4. Otimização Avançada com DSPy (Teleprompters)](#34-otimização-avançada-com-dspy-teleprompters)
    - [3.5. Gestão de Estado e Observabilidade](#35-gestão-de-estado-e-observabilidade)

<!-- /code_chunk_output -->

