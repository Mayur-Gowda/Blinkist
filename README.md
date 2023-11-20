# Open Blinkist

Book Summarizer and Audio Converter
The Book Summarizer and Audio Converter is a project designed to help users summarize large volumes of text, such as books or documents, using advanced natural language processing models. It employs BART (Bidirectional and Auto-Regressive Transformers) models to break down the input text into manageable chunks and generate concise summaries for each chunk. Additionally, the project provides the functionality to convert the summarized text into an audio format, offering users an alternative way to consume the content.

## Table of Contents
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Technologies Used](#technologies-used)
- [File Hierarchy](#file-hierarchy)

<a name="features"></a>
## Features
- Text Summarization: Utilizes BART models to summarize lengthy text into digestible chunks, allowing users to grasp the main points efficiently.
- Audio Conversion: Converts the summarized text into audio files, enabling users to listen to the content, making it accessible for those preferring an auditory learning experience.
- Real-time Communication: Incorporates real-time communication between the frontend and backend using Flask-SocketIO, providing users with live updates on the summarization process.
- Readability Metrics: Calculates readability metrics such as Flesch-Kincaid Grade Level, Gunning Fog Index, and Coleman-Liau Index to evaluate the complexity of the summarized text.

<a name="getting-started"></a>
## Getting Started
### Local Installation Guide
Follow these steps to set up the Book Summarizer and Audio Converter project on your local machine.

<a name="prerequisites"></a>
### Prerequisites
Ensure you have the following installed on your system:
- Python (version 3.10)
- Virtualenv (for creating a virtual environment)

<a name="installation"></a>
### Installation Options
Choose the method that best suits you to get started with Blinkist:

#### 1. Download and Install:
Download the repository using the ZIP file.
Extract the contents to your local machine.
Navigate to the project directory in your preferred terminal.

#### 2. Clone the Repository:
```
git clone https://github.com/Mayur-Gowda/Blinkist.git

cd Blinkist

python -m venv venv

.\venv\Scripts\activate

pip install -r requirements.txt

python main.py
```

<a name="technologies-used"></a>
### Technologies Used
- Flask: The web framework used for the backend server.
- Flask-SocketIO: Enables real-time communication between the frontend and backend.
- BART Models: Powerful transformers used for text summarization.
- Pyttsx3: Library for converting text to speech and generating audio files.

<a name="file-hierarchy"></a>
### File Hierarchy
```
.
│   main.py
│   requirements.txt
│
├───summarize
│   │   functions.py
│   │   routes.py
│   │   sockets.py
│   │   __init__.py
│   │
│   ├───static
│   │   ├───css
│   │   │       base.css
│   │   │       create.css
│   │   │       home.css
│   │   │       summary_page.css
│   │   │
│   │   └───img
│   │           bg.jpg
│   │           Github.png
│   │
│   └───templates
│           base.html
│           create.html
│           home.html
│           summarize_page.html
│
└───uploads
```
