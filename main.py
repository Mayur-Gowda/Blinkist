"""
Open Blinkist: Text summarization tool.
The main Python script. Creates a Flask app.

"""

import os
from summarize import create_app, socketio

app = create_app()

if __name__ == "__main__":
    # Check if uploads folder exists, if not create it
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # Run the Flask app with Socket.IO support
    # Set allow_unsafe_werkzeug=True to allow WebSocket transport
    socketio.run(app, allow_unsafe_werkzeug=True)


# Flesch-Kincaid Grade: 8.2 Gunning Fog Index: 9.5 Coleman-Liau Index: 8.94
# For Facebook BART-Base
