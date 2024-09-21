from flask import redirect, request, url_for, Blueprint, render_template
import joblib

# Criando o Blueprint
app_blueprint = Blueprint('app', __name__)

# Rota raiz que redireciona para a página principal
@app_blueprint.route('/')
def root():
    return redirect(url_for('app.home'))

# Rota para a página principal
@app_blueprint.route('/DetectionPhishing/home')
def home():
    return render_template('index.html')

# Rota para a página de entrada do programa
@app_blueprint.route('/DetectionPhishing/home/detector', methods=['GET'])
def detector():
    return render_template('detector.html')

# Rota para fazer a previsão usando o modelo
@app_blueprint.route('/DetectionPhishing/home/detector/predict', methods=['POST'])
def predict():
    try:
        # Coleta do dado de entrada
        input_url = request.form.get('input', type=str)
        
        # Faz a previsão
        result = predict_phishing(input_url)

        # Retorna a resposta ao usuário
        if result == 1:
            message = "A URL é phishing."
        else:
            message = "A URL é segura."
        
        return render_template('detector.html', message=message)

    except Exception as e:
        # Retorna uma mensagem de erro em caso de falha
        return render_template('detector.html', message=f"Ocorreu um erro: {str(e)}")

# Função para prever se uma URL é phishing
def predict_phishing(url):
    # Carregar o modelo e o vectorizer da pasta dump
    model = joblib.load('dump/phishing_model.pkl')
    vectorizer = joblib.load('dump/vectorizer.pkl')

    # Transformar a URL em características numéricas
    url_transformed = vectorizer.transform([url])

    # Fazer a previsão
    prediction = model.predict(url_transformed)
    return prediction[0]  # Retorna 1 para phishing, 0 para seguro
