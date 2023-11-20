from flask import Flask
from .routes import main
from .sockets import socketio


def create_app():
    """
    Create and configure the Flask app.

    Returns:
        Flask: The configured Flask app.
    """
    app = Flask(__name__)

    # Set configuration options for the Flask app
    app.config['DEBUG'] = True
    app.config['SECRET_KEY'] = 'thisIsASecretKey'

    with app.app_context():
        # Register the main blueprint containing your routes
        app.register_blueprint(main)

        # Initialize Socket.IO with the Flask app
        socketio.init_app(app)

    return app
