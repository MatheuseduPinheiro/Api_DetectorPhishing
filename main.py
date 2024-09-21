from flask import Flask
from controller.route import app_blueprint

# Inicializa a aplicação Flask
app = Flask(__name__)

# Registra o Blueprint na aplicação
app.register_blueprint(app_blueprint)

# Executa a aplicação Flask
if __name__ == '__main__':
    # Configura para escutar no IP e porta desejados
    app.run(host='127.0.0.1', port=5000, debug=True)
