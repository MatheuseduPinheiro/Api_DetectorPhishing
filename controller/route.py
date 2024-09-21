from flask import Blueprint, redirect, url_for, render_template

# Criando o Blueprint
app_blueprint = Blueprint('app', __name__)

# Rota raiz que redireciona para a página principal
@app_blueprint.route('/')
def root():
    return redirect(url_for('app.home'))

# Rota para a página principal
@app_blueprint.route('/DetectionFishing/home')
def home():
    return render_template('index.html')

# Rota para a página de entrada do programa
@app_blueprint.route('/DetectionFishing/home/detector', methods=['GET'])
def detector():
    return render_template('detector.html')


