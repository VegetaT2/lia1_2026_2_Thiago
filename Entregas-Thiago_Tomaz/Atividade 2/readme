<div align="center">

  <h1>🎬 IMDB Sentiment Analysis with Deep Learning</h1>
  <p><strong>Rede Neural Convolucional 1D para Mineração de Opinião e Análise de Sentimento</strong></p>

  <p>
    <a href="https://colab.research.google.com/github/SEU_USUARIO/SEU_REPOSITORIO/blob/main/NOME_DO_NOTEBOOK.ipynb">
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
