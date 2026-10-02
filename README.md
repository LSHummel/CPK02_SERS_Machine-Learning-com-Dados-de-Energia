[Readme_cp_SERS.md](https://github.com/user-attachments/files/32940426/Readme_cp_SERS.md)
# Machine Learning com Dados de Energia

**CPK02 · SERS** — Avaliação: APIs de energia renovável e aprendizado de máquina

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LSHummel/CPK02_SERS_Machine-Learning-com-Dados-de-Energia/blob/main/aula_apis_energia_renovavel_ml.ipynb)
**AUTOR**
Matheus Pimenta Martini - RM:569400
Lucas Seiji - RM:569673
Leonardo Soares Rodrigues - RM:572986

## Objetivo

Consultar duas APIs públicas de dados abertos, organizar os dados em CSV e resolver duas tarefas de aprendizado supervisionado, treinando e comparando **três algoritmos em cada uma**.

| Tarefa | Tipo | Pergunta | Alvo (`y`) |
|---|---|---|---|
| 1 — ANEEL | Classificação | A partir da potência e da localização de um empreendimento, é possível dizer se a fonte é **Solar, Eólica ou Hidráulica**? | `fonte` |
| 2 — Open-Meteo | Regressão | Dadas as condições meteorológicas e a hora local em **Petrolina (PE)**, qual é a radiação solar horizontal média naquela hora? | `radiacao_w_m2` |

## Dados

### Fontes e período

| | Tarefa 1 — ANEEL | Tarefa 2 — Open-Meteo |
|---|---|---|
| Conjunto | SIGA – Sistema de Informações de Geração da ANEEL | Historical Weather API |
| Acesso | API CKAN/DataStore (`datastore_search`), `resource_id` `11ec447d-698d-4ab8-977f-b424d5deee6a` | `archive-api.open-meteo.com/v1/archive` |
| Recorte | Tipos `UFV`, `EOL`, `UHE`, `PCH` e `CGH`, até 1.200 registros por tipo | Lat. −9,39 · Lon. −40,50 · fuso `America/Recife` · horas das 7h às 17h |
| Período | Cadastro vigente na data da consulta (a base é atualizada continuamente) | **01/04/2025 a 30/06/2025**, dados horários |
| Autenticação | Nenhuma (consulta pública, sem token) | Nenhuma (consulta pública, sem token) |

Os dados do Open-Meteo são estimativas de modelos/reanálise, não leituras de um sensor específico.

### Arquivos gerados

**`aneel_classificacao_orange.csv`** — 3.876 empreendimentos (um por linha), sem valores ausentes.

| Coluna | Significado | Uso |
|---|---|---|
| `potencia_kw` | Potência outorgada em kW (capacidade autorizada, não energia gerada) | entrada (X) |
| `latitude` | Latitude aproximada, em graus decimais | entrada (X) |
| `longitude` | Longitude aproximada, em graus decimais | entrada (X) |
| `fonte` | `Solar` (UFV), `Eólica` (EOL) ou `Hidráulica` (UHE + PCH + CGH) | alvo (y) |

- Registros recebidos: UFV 1.200 · EOL 1.200 · UHE 221 · PCH 537 · CGH 718. UFV e EOL atingiram o limite da consulta, então são os primeiros 1.200 registros devolvidos pela API, não uma amostra aleatória; por isso a distribuição das classes **não** representa a matriz elétrica brasileira.
- Classes: Hidráulica 1.476 (38,1%) · Solar 1.200 (31,0%) · Eólica 1.200 (31,0%). O desequilíbrio é leve e foi tratado com divisão estratificada e métricas com média macro.
- `SigTipoGeracao`, nome, CEG e campos de combustível ficaram fora de X porque revelariam a classe.

**`meteo_regressao_orange.csv`** — 1.001 horas (91 dias × 11 horas), sem valores ausentes.

| Coluna | Significado | Uso |
|---|---|---|
| `data_hora` | Data e hora local | só ordenação e divisão temporal |
| `temperatura_c` | Temperatura do ar a 2 m (°C) | entrada (X) |
| `umidade_pct` | Umidade relativa a 2 m (%) | entrada (X) |
| `nuvens_pct` | Cobertura total de nuvens (%) | entrada (X) |
| `vento_kmh` | Velocidade do vento a 10 m (km/h) | entrada (X) |
| `hora` | Hora local (7 a 17) | entrada (X) |
| `radiacao_w_m2` | Radiação solar global horizontal média da hora anterior (W/m²) | alvo (y) |

## Metodologia

**Tarefa 1 — Classificação**
- Exploração: tipos das colunas, contagem por classe e valores ausentes.
- Divisão **estratificada** 80/20 com `random_state=42`: 3.100 empreendimentos para treino e 776 para teste, com a mesma proporção de classes nos dois conjuntos.
- Modelos (scikit-learn, parâmetros padrão), todos na mesma divisão: **KNN** (k = 5), **Regressão Logística** e **Random Forest**.
- Métricas: acurácia, precisão, recall e F1 com **média macro** (mesmo peso para cada classe) e matriz de confusão de cada modelo.

**Tarefa 2 — Regressão**
- Exploração: valores ausentes e matriz de correlação de Pearson.
- Divisão **temporal**, sem embaralhar: as primeiras 800 horas (01/04/2025 07h → 12/06/2025 14h) para treino e as 201 finais (12/06/2025 15h → 30/06/2025 17h) para teste.
- Modelos: **Regressão Linear**, **Árvore de Decisão** e **Random Forest** (`random_state=42` nos dois últimos).
- Métricas: MAE (W/m²), MSE ((W/m²)²) e R², além do gráfico de valores reais × previstos do Random Forest.

## Resultados

### Tarefa 1 — Classificação da fonte (776 empreendimentos de teste)

| Modelo | Acurácia | Precisão (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| KNN | 0,869 | 0,869 | 0,867 | 0,868 |
| Regressão Logística | 0,724 | 0,719 | 0,719 | 0,719 |
| **Random Forest** | **0,976** | **0,977** | **0,974** | **0,975** |

**Modelo escolhido: Random Forest** — o melhor em todas as métricas, com F1 acima de 0,97 nas três classes. A Regressão Logística ficou bem atrás porque traça fronteiras lineares entre as classes, enquanto a relação entre localização e fonte é regional e não linear.

Matriz de confusão do Random Forest (19 erros em 776):

| Real \ Previsto | Eólica | Hidráulica | Solar |
|---|---|---|---|
| **Eólica** | **235** | 4 | 1 |
| **Hidráulica** | 2 | **294** | 0 |
| **Solar** | 5 | 7 | **228** |

A classe mais difícil é **Solar** (recall 0,95): 7 usinas solares foram previstas como Hidráulica e 5 como Eólica. Hidráulica é a classe que mais recebe falsos positivos (precisão 0,964).

**Por que potência e localização não bastam numa aplicação real:** as coordenadas indicam a região, mas não o que de fato define cada fonte — rios e relevo, regime de ventos, irradiação local —, e empreendimentos de fontes diferentes podem ter potência e localização parecidas, como mostram os casos confundidos. Uma aplicação real precisaria de variáveis geográficas, hidrográficas e climáticas adicionais.

### Tarefa 2 — Radiação solar em Petrolina (201 horas de teste)

| Modelo | MAE (W/m²) | MSE ((W/m²)²) | R² |
|---|---|---|---|
| Regressão Linear | 145,20 | 30.034,20 | 0,360 |
| Árvore de Decisão | 88,79 | 15.291,16 | 0,674 |
| **Random Forest** | **66,80** | **7.307,42** | **0,844** |

**Melhor modelo: Random Forest** — erro absoluto médio de cerca de 67 W/m² e R² de 0,844 no período de teste, contra 0,360 da Regressão Linear.

**O peso da hora:** a radiação sobe pela manhã, atinge o pico perto do meio-dia e cai à tarde, numa curva em forma de sino. Por isso a correlação linear entre `hora` e radiação é baixa (0,12), embora a hora seja decisiva: a Regressão Linear só representa um efeito que cresce ou decresce com a hora, enquanto as árvores dividem o dia em faixas e capturam o ciclo solar. Temperatura (0,64) e umidade (−0,60) são as variáveis mais correlacionadas com a radiação e também acompanham esse ciclo diário (correlação de 0,70 e −0,73 com a hora).

**Radiação não é energia gerada:** o modelo estima radiação horizontal em W/m² — potência por área —, não a produção de um sistema fotovoltaico. A energia gerada (kWh) depende ainda da área e da eficiência dos módulos (tipicamente 15% a 22%), da inclinação e orientação dos painéis, das perdas por temperatura e das perdas em cabos e no inversor.

## Como executar

**Opção 1 — Google Colab (recomendado):** abra pelo botão *Open in Colab* acima e use *Ambiente de execução → Executar tudo*. As células de consulta baixam os dados das APIs e geram os dois CSVs no próprio ambiente.

**Opção 2 — Local (Jupyter):**

```bash
git clone https://github.com/LSHummel/CPK02_SERS_Machine-Learning-com-Dados-de-Energia.git
cd CPK02_SERS_Machine-Learning-com-Dados-de-Energia
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn notebook
jupyter notebook aula_apis_energia_renovavel_ml.ipynb
```

Execute as células na ordem (*Kernel → Restart & Run All*). É preciso acesso à internet, mas nenhuma chave de API.

> **Reprodutibilidade:** as células de consulta regravam os CSVs com os dados atuais das APIs. Como a base da ANEEL é atualizada continuamente, uma nova consulta pode trazer registros diferentes e alterar levemente os números da Tarefa 1; os CSVs versionados neste repositório correspondem aos resultados acima. O notebook foi executado no Google Colab (Python 3.13, scikit-learn 1.6).

## Estrutura do repositório

```
├── README.md
├── aula_apis_energia_renovavel_ml.ipynb   # consultas às APIs, análise, 6 modelos e conclusões
├── aneel_classificacao_orange.csv         # dados da Tarefa 1 (classificação)
└── meteo_regressao_orange.csv             # dados da Tarefa 2 (regressão)
```

## Fontes

- [ANEEL — SIGA (Sistema de Informações de Geração)](https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel)
- [ANEEL — recurso e campos usados](https://dadosabertos.aneel.gov.br/dataset/siga-sistema-de-informacoes-de-geracao-da-aneel/resource/11ec447d-698d-4ab8-977f-b424d5deee6a)
- [Open-Meteo — Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
