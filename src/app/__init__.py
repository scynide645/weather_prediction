from flask import Flask
from .model_loader import load_model

def create_app():
    app = Flask(__name__)
    app.model = load_model()

    from .routes import main
    app.register_blueprint(main)

    return app
