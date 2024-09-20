import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score 
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

print("="*10,"Predição dos Phishing`s","="*10)
print("="*45)


print("="*10,"Importando biblioteca","="*12)

df = pd.read_csv("datasets/PhiUSIIL_Phishing_URL_Dataset.csv")


print(df.head(5))


print("="*12,"Verificar a distribuição dos dados e estatísticas","="*12)
print(df.describe())

print(df.info())


# Cálculo da porcentagem de valores nulos
equacao = (df.isnull().sum() / len(df)) * 100

# Imprimindo o resultado formatado corretamente
print("="*12, f"Porcentagem de valores nulos:\n{equacao}", "="*12)


def dadosObjects(df):
    # Verifica se algum dado no DataFrame é do tipo 'object'
    if any(df.dtypes == 'object'):
        print("Existem colunas do tipo 'object':")
        print(df.dtypes[df.dtypes == 'object'])
    else:
        print("Não há colunas do tipo 'object'.")

def dadosfloots(df):
    df_float = df.dtypes[df.dtypes == 'float64']
    for i in range(len(df.columns)):
        if df.dtypes[df.columns[i]] == 'float64':
            print("Existem colunas do tipo 'float64':")
            print(df_float)

def dadosints(df):
    df_int = df.dtypes[df.dtypes == 'int64']
    for i in range(len(df.columns)):
        if df.dtypes[df.columns[i]] == 'int64':
            print("Existem colunas do tipo 'float64':")
            print(df_int)


#Chamar os objcts
print("="*12,"Dados do tipo object","="*12)
dadosObjects(df)

#Chamar os floats
print("="*12,"Dados do tipo float","="*12)
dadosfloots(df)

#Chamamos os Interiros
print("="*12,"Dados do tipo inteiro","="*12)
print(df)

df_url = df[['URL']]
print("="*12,"Definindo URL como minha única característica","="*12)
print(df_url)

print("="*12,"Separar Característica dos Rotulos","="*12)

# Selecionar colunas de características e rótulo
# Característica (X): Apenas a URL
x = df['URL'].values

# Rótulos (y): Diversos rótulos indicativos
y = df[['label']].values

# Verificar os primeiros valores
print(x[:5])

print(y[:5])

# Transformar URLs em características numéricas
vectorizer = TfidfVectorizer()
x_transformed = vectorizer.fit_transform(x)
print("="*12,"Transformar URLs em características numéricas","="*12)
# Dividir os dados em treino e teste
x_train, x_test, y_train, y_test = train_test_split(x_transformed, y, test_size=0.2, random_state=42)

# Criar o classificador KNN
knn = KNeighborsClassifier(n_neighbors=5)

# Treinar o classificador
knn.fit(x_train, y_train)

# Fazer previsões
y_pred = knn.predict(x_test)

# Avaliar o desempenho
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')


# Função para prever se uma URL é phishing
def predict_phishing(url):
    # Carregar o modelo e o vectorizer
    model = joblib.load(f'{path}\\phishing_model.pkl')
    vectorizer = joblib.load(f'{path}\\vectorizer.pkl')
    
    # Transformar a URL em características numéricas
    url_transformed = vectorizer.transform([url])
    
    # Fazer a previsão
    prediction = model.predict(url_transformed)
    return prediction[0]

#input_url = ""
#result = predict_phishing(input_url)
#if result == 1:
#    print("A URL é phishing.")
#else:
#    print("A URL é segura.")


# Definir o caminho do diretório
path = r''

# Salvar o modelo e o vectorizer no diretório específico
joblib.dump(knn, f'{path}\\phishing_model.pkl')
joblib.dump(vectorizer, f'{path}\\vectorizer.pkl')