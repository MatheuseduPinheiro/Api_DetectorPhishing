import pandas as pd

print("="*10,"Predição dos Phishing`s","="*10)
print("="*45)


print("="*10,"Importando biblioteca","="*12)

df = pd.read_csv("datasets/PhiUSIIL_Phishing_URL_Dataset.csv")


print(df.head(5))


print("="*12,"Verificar a distribuição dos dados e estatísticas","="*12)
print(df.describe())

print("="*12,"Fuçar")
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

# Chame a função passando o DataFrame como argumento

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


df.info()