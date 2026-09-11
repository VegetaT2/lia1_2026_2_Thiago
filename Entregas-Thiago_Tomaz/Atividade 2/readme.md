<div align="center">

  <h1>🎬 IMDB Sentiment Analysis with Deep Learning</h1>
  <p><strong>Rede Neural Convolucional 1D para Mineração de Opinião e Análise de Sentimento</strong></p>

  <p>
    <a href="https://colab.research.google.com/github/VegetaT2/lia1_2026_2_Thiago/blob/main/Entregas-Thiago_Tomaz/Atividade%202/kerasimdb.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab">
    </a>
    <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white&style=flat-square" alt="TensorFlow">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white&style=flat-square" alt="Python">
    <img src="https://img.shields.io/badge/Keras-D00000?logo=keras&logoColor=white&style=flat-square" alt="Keras">
    <img src="https://img.shields.io/badge/License-MIT-blue?style=flat-square" alt="License">
  </p>

  <sub>Transformando resenhas cinematográficas em probabilidades com Processamento de Linguagem Natural.</sub>[cite: 2]

</div>

---

### 📋 Visão Geral

Este projeto implementa um pipeline completo focado na **classificação binária de sentimentos** (positivo ou negativo)[cite: 2] utilizando representações latentes contínuas (*Word Embeddings*)[cite: 2] e convoluções unidimensionais (**Conv1D**)[cite: 2].
---

### 📊 Conjunto de Dados

Treinado no **IMDB Large Movie Review Dataset**:

| Especificação | Detalhes |
| :--- | :--- |
| **Volume Total** | 50.000 críticas em língua inglesa |
| **Distribuição** | 25.000 treino / 25.000 teste (balanceado 50/50) |
| **Vocabulário** | 10.000 termos mais frequentes do corpus |
| **Comprimento de Entrada** | Fixado em 250 tokens com padding e truncamento |

---

### 🏗️ Arquitetura do Modelo

A rede foi projetada para capturar dependências locais e padrões textuais com alta eficiência computacional:

```text
=================================================================
 Layer (type)                 Output Shape              Param #   
=================================================================
 InputLayer                   (None, 250)               0         
 Embedding (64D)              (None, 250, 64)           640,000   
 Conv1D (Filtros: 64, k=5)    (None, 246, 64)           20,544    
 MaxPooling1D (pool_size=4)   (None, 61, 64)            0         
 Conv1D (Filtros: 64, k=5)    (None, 57, 64)            20,544    
 GlobalMaxPooling1D           (None, 64)                0         
 Dense (ReLU)                 (None, 64)                4,160     
 Dropout (0.5)                (None, 64)                0         
 Dense (Sigmoid)              (None, 1)                 65        
=================================================================
---

## 📈 Desempenho e Resultados

Resultados consolidados após 5 épocas de treinamento com algoritmo Adam e função de perda `binary_crossentropy`[cite: 2]:

* **Acurácia no Teste:** `85.06%`[cite: 2]
* **Loss no Teste:** `0.5631`[cite: 2]

### Matriz de Métricas Detalhada (Conjunto de Teste)

| Classe | Precision | Recall | F1-Score | Suporte |
| :--- | :---: | :---: | :---: | :---: |
| **Negativo (0)** | 0.82[cite: 2] | 0.89[cite: 2] | 0.86[cite: 2] | 12.500[cite: 2] |
| **Positivo (1)** | 0.88[cite: 2] | 0.81[cite: 2] | 0.84[cite: 2] | 12.500[cite: 2] |
| **Acurácia Média** | — | — | **0.85**[cite: 2] | **25.000**[cite: 2] |

---

## 🕷️ Testes com Críticas Reais: *Spider-Man: Brand New Day*

Inferência em lote com resenhas textuais não rotuladas sobre o filme[cite: 2]:

* **[1]** *"Spider-Man Brand New Day is exactly the fresh start Peter Parker needed. Tom Holland delivers a mature, grounded and deeply emotional performance, back in New York City with incredible street-level action."*[cite: 2]  
  ➡️ **Sentimento Previsto:** `POSITIVO` | **Grau de Confiança:** `74.54%`[cite: 2]

* **[2]** *"A massive disappointment. Brand New Day has a boring narrative, horrible pacing, and completely throws away Peter's development with forced drama and dull villains."*[cite: 2]  
  ➡️ **Sentimento Previsto:** `NEGATIVO` | **Grau de Confiança:** `100.00%`[cite: 2]

* **[3]** *"The swinging scenes look visually stunning and the raw physical combat in the streets was fantastic. It feels like classic comic book Spider-Man finally brought to life on the big screen."*[cite: 2]  
  ➡️ **Sentimento Previsto:** `POSITIVO` | **Grau de Confiança:** `99.95%`[cite: 2]

* **[4]** *"The plot was a convoluted mess. The emotional core fell totally flat and the dialogue felt uninspired and poorly written throughout the entire two hours."*[cite: 2]  
  ➡️ **Sentimento Previsto:** `NEGATIVO` | **Grau de Confiança:** `100.00%`[cite: 2]

* **[5]** *"An extraordinary return to form! The bittersweet tone of Peter dealing with his solitude while protecting the city creates one of the best superhero movies in years."*[cite: 2]  
  ➡️ **Sentimento Previsto:** `POSITIVO` | **Grau de Confiança:** `100.00%`[cite: 2]

* **[6]** *"Felt rushed and lifeless. Overhyped marketing for a generic movie that fails to deliver any real excitement or memorable moments."*[cite: 2]  
  ➡️ **Sentimento Previsto:** `NEGATIVO` | **Grau de Confiança:** `100.00%`[cite: 2]

---

## 🚀 Como Executar no Google Colab

Abra e execute o código interativo diretamente na nuvem:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VegetaT2/lia1_2026_2_Thiago/blob/main/kerasimdb.ipynb)

