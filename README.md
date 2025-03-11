# Api_DetectionPhishing

O projeto **Api_DetectionPhishing** é uma API desenvolvida em Flask que utiliza aprendizado de máquina para detectar URLs de phishing. Ele processa URLs e retorna uma resposta indicando se são maliciosas ou legítimas. O objetivo principal é fornecer uma interface simples e eficiente para prever a segurança de URLs.

![modelo](https://github.com/user-attachments/assets/a56259da-c170-4ecc-8ed2-ee3e3da56d19)
# Figura 1.Processo de Detecção de URLs de Phishing utilizando TF-IDF e Aprendizado de Máquina.Fonte: Autor feita pelo Canva.

O processo de detecção de URLs de phishing foi aprimorado ao transformar as URLs em dados numéricos utilizando a técnica de TF-IDF. Isso foi feito para melhorar o desempenho do modelo, pois trabalhar com dados estruturados é mais eficaz do que lidar com strings de texto bruto, como os caracteres das URLs. Ao usar o algoritmo de Árvore de Decisão, a API foi capaz de classificar as URLs com maior precisão, separando as URLs legítimas das maliciosas com base nas características extraídas. Para mais detalhes sobre a implementação e a metodologia, consulte o notebook no [GitHub](https://github.com/MatheuseduPinheiro/DetectorPhishingNotebook/blob/main/detection_phishing.ipynb). As imagens utilizadas para ilustrar o processo foram retiradas dos seguintes sites: [Rany Elhousieny PhD](https://www.linkedin.com/pulse/understanding-tf-idf-rany-elhousieny-phd%E1%B4%AC%E1%B4%AE%E1%B4%B0/) e [Flaticon](https://www.flaticon.com/br/icone-gratis/aprendizagem-profunda_8637101?related_id=8637101).


## 1. Entrada dos Dados:
- **URL**: Conjunto de URLs (legítimas e de phishing).
- **Label**: Rótulos associados (0 para legítima, 1 para phishing).

---

## 2. Transformação dos Dados:
- As URLs passam pelo processo de vetorização utilizando **TF-IDF** para gerar representações numéricas.
- **Fórmula do TF-IDF** destacada:

$ w_{x,y} = tf_{x,y} \times \log \left( \frac{N}{df_x} \right) $

Onde:
- $ tf_{x,y} $: Frequência do termo $ x $ no documento $ y $.
- $ df_x $: Número de documentos contendo o termo $ x $.
- $ N $: Número total de documentos.

---

## 3. Divisão de Dados:
- Os dados são divididos em **treinamento** e **teste**.

---

## 4. Treinamento do Modelo:
- Um modelo de aprendizado de máquina é treinado utilizando os dados processados para realizar a classificação.

---

## 5. Previsão e Interface:
- O modelo realiza previsões sobre novas URLs.
- Os resultados são apresentados na interface, indicando se a URL é **segura** ou **maliciosa**.


## Estrutura do Projeto

```plaintext
Api_DetectionPhishing/
├── controller/              # Lógica principal da API (ex.: rotas Flask)
│   └── __pycache__/         # Arquivos cache do Python
├── datasets/                # Dados de treinamento e testes usados no projeto
├── dump/                    # Modelos treinados e vetorizadores (arquivos .pkl)
├── model/                   # Scripts relacionados ao treinamento do modelo
├── static/                  # Arquivos estáticos para o frontend
│   ├── css/                 # Arquivos CSS
│   └── images/              # Imagens para interface
└── templates/               # Templates HTML para interface Flask

## Tecnologias Utilizadas

- **Flask**: Framework para desenvolvimento da API e renderização da interface web.  
- **Scikit-learn**: Biblioteca de aprendizado de máquina para treinamento e predição.  
- **TF-IDF**: Método de vetorização para transformar texto (URLs) em representações numéricas.  
- **Joblib**: Ferramenta para salvar e carregar modelos treinados.  
- **HTML e CSS**: Para desenvolvimento da interface de usuário.  
- **Python 3.9+**: Linguagem principal utilizada no projeto.  
