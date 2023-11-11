import time
import eventlet
from flask import Flask, render_template, request, redirect, url_for, send_file
from flask_socketio import SocketIO, emit
from werkzeug.utils import secure_filename
import PyPDF2
import re
import os
from transformers import pipeline
from textstat import textstat as ts


summarizer = pipeline("summarization", model="facebook/bart-base")

app = Flask(__name__)
socketio = SocketIO(app, ping_interval=720, ping_timeout=900)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'txt', 'pdf'}

global pdf_path
global processed_text
global chunk
global all_summaries

eventlet.monkey_patch()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess(path: str):
    """ Creates processed text from the pdf file

  Takes a path string for the pdf and converts it into a text.
  It then preprocesses removing newlines, punctuations and other characters.

  Parameters:
  -----------
  path : str
    The path for the pdf file

  """

    # List contains the keywords for removing unnecessary pages (Incomplete)
    remove_lt = ['CONTENTS', 'Contents', 'contents',
                 'PREFACE', 'Preface', 'preface',
                 'ILLUSTRATIONS', 'Illustrations', 'illustrations',
                 'COPYRIGHT INFORMATION', 'Copyright Information', 'copyright information',
                 'FOREWORD', 'Foreword', 'foreword',
                 'ACKNOWLEDGEMENTS', 'Acknowledgments', 'acknowledgments']

    # Text Processing
    text = ""

    # Extract text from the pdf
    with open(path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]

            # Extract text from the page
            page_text = page.extract_text()

            # Check for the keywords in the page
            found_common_element = any(element in page_text for element in remove_lt)

            # If no keywords were found on the page, add it to the text
            if not found_common_element:
                text += page_text

    # Remove newlines from the text and replace with space
    text = text.replace('\n', ' ').replace('\r', ' ')
    # Remove copyright Symbols
    text = re.sub(r'©.*', '', text)
    # Remove fonts
    font_code_pattern = r'/[A-Z0-9]+'
    text = re.sub(font_code_pattern, '', text)
    # Remove excessive spaces in the text
    text = ' '.join(text.split())

    emit('pr-status', {'text': "Preprocessing Complete.\n\n", 'num': 1})
    return text


def text_chunking(new_text: str):
    """ Splits a long piece of text into smaller chunks based on sentence boundaries while
    ensuring that each chunk's word count does not exceed a maximum limit

  The text is split into chunks using regular expressions that search for sentence ending punctuation.

  Parameter:
  ----------
  new_text: str
    The preprocessed text of the file.

  """

    # Define the max no. of chunks
    max_chunk = 500

    # Split text into sentences using re
    sentences = re.split(r'(?<=[.!?])\s+', new_text)
    current_chunk = 0
    chunks = []

    # Split the sentence into chunks
    for sentence in sentences:
        if len(chunks) == current_chunk + 1:
            if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
                chunks[current_chunk].extend(sentence.split(' '))
            else:
                current_chunk += 1
                chunks.append(sentence.split(' '))
        else:
            chunks.append(sentence.split(' '))

    for chunk_id in range(len(chunks)):
        chunks[chunk_id] = ' '.join(chunks[chunk_id])
    emit('chunking', {'text': "Total chunks of text are: " + str(len(chunks)) + "\n\n", 'num': 1})
    return chunks


def model_summary(chunks: list):
    """ Creates summary of the chunks and combines them into a single summary.

  Summarizes a chunk and combines it into a bigger summary.

  Parameter:
  ----------
  chunks: list
    List containing the sub-chunks of the sentences.

  """

    summaries = []
    count = 0

    # Summarize each chunk
    for chnk in chunks:
        print(f"Summarizing Chunk NO: {count + 1}")
        emit('summ_chunk', {'text': 'Summarizing Chunk NO.:' + str(count + 1),
                            'num': 2, 'count': count + 1}, callback=ack)
        time.sleep(0.1)
        res = summarizer(chnk, max_length=150, min_length=30, do_sample=False)
        summaries += res
        count += 1
    emit('chunk_done', {'text': '\nEach chunk summarized.'})
    return summaries


def prep_b4_save(text: str):
    """ Formats the text to proper readability.

  Parameter:
  ----------
  text: str
    Summary text for formatting

  """

    text = re.sub('Gods', 'God\'s', text)
    text = re.sub('yours', 'your\'s', text)
    text = re.sub('dont', 'don\'t', text)
    text = re.sub('doesnt', 'doesn\'t', text)
    text = re.sub('isnt', 'isn\'t', text)
    text = re.sub('havent', 'haven\'t', text)
    text = re.sub('hasnt', 'hasn\'t', text)
    text = re.sub('wouldnt', 'wouldn\'t', text)
    text = re.sub('theyre', 'they\'re', text)
    text = re.sub('youve', 'you\'ve', text)
    text = re.sub('arent', 'aren\'t', text)
    text = re.sub('youre', 'you\'re', text)
    text = re.sub('cant', 'can\'t', text)
    text = re.sub('whore', 'who\'re', text)
    text = re.sub('whos', 'who\'s', text)
    text = re.sub('whatre', 'what\'re', text)
    text = re.sub('whats', 'what\'s', text)
    text = re.sub('hadnt', 'hadn\'t', text)
    text = re.sub('didnt', 'didn\'t', text)
    text = re.sub('couldnt', 'couldn\'t', text)
    text = re.sub('theyll', 'they\'ll', text)
    text = re.sub('youd', 'you\'d', text)
    return text


def calculate_scores(text: str):
    """ Calculates the readability and the coherence of the summary.

  Uses Flesch-Kincaid Grade, Gunning Fog index, Coleman-Liau index.

  Parameter:
  ----------
  text: str
    The post-processed text of the file

  """

    # Flesch-Kincaid Grade Level, Gunning Fog Index, and Coleman-Liau Index readability metrics to evaluate
    # the readability of your text or summaries.
    # These metrics provide an estimate of the grade level required to understand a piece of text.
    # Lower grade levels indicate easier readability

    emit('end_phase', {'text': "Calculating scores...\n", 'num': 2})

    # The Flesch-Kincaid Grade Level is based on the average number of syllables per word and the average
    # number of words per sentence
    # Ideal-Score: 8-10
    grade_level = ts.flesch_kincaid_grade(text)

    # The Gunning Fog Index is based on the number of complex words (words with three or more syllables)
    # and the average sentence length
    # Ideal-Score: 8-10
    fog_index = ts.gunning_fog(text)

    # The Coleman-Liau Index is based on the number of letters, words, and sentences in the text
    # Ideal Range: 8-12
    coleman_liau = ts.coleman_liau_index(text)

    emit('end_phase', {'text': f"Flesch-Kincaid Grade: {grade_level} \
          Gunning Fog Index: {fog_index} \
          Coleman-Liau Index: {coleman_liau}", 'num': 3})


def save_to_text(text: str, name: str):
    """ Creates a text file of the summary produced

  Creates a text file and saves it in the current working directory

  Parameter:
  ----------
  text: str
    The summarized text that is needed to save into the text file.

  name: str
    The name of the book (Name of the book when input is taken here.)

  """
    file_name = f"{name}_summary.txt"
    with open(file_name, 'w') as txt:
        txt.write(text)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/create', methods=['GET', 'POST'])
def create():
    if request.method == 'POST':
        # Check if the post request has any file parts
        if not request.files:
            return redirect(request.url)

        file = request.files['file']

        # If the user does not select a file, the browser should also submit an empty part without a filename
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            global pdf_path
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            pdf_path = file_path
            return redirect(url_for('summarize'))

    return render_template("create.html")


@app.route('/summarize')
def summarize():
    return render_template('summarize_page.html')


@socketio.on('connect')
def connect():
    print("Connected")
    emit('status', {'text': 'Summarization started<br><br>Preprocessing Text ...'})


@socketio.on('preprocess')
def pre(text):
    global processed_text
    print(text['text'])

    # Preprocess the text
    processed_text = preprocess(pdf_path)

    socketio.emit('pr-status', {'text': 'Chunking text.', 'num': 2})


@socketio.on('chunk')
def chunking(text):
    global chunk
    print(text['text'])

    # Chunk the large text into small parts so it can be supplied to the model
    chunk = text_chunking(processed_text)


@socketio.on('chunk_summarize')
def chunk_sum(text):
    global all_summaries
    print(text['text'])
    emit('summ_chunk', {'text': "Summarizing the text. Please wait .......\n", 'num': 1}, callback=ack)
    time.sleep(0.1)

    # Passing the chunks to the model for the summarization
    all_summaries = model_summary(chunk)


@socketio.on('post_process')
def post_process(text):
    print(text['text'])
    emit('end_phase', {'text': '\nPost Processing', 'num': 1})
    # Combine all chunks of summaries to a single one
    joined_summary = ' '.join([summ['summary_text'] for summ in all_summaries])

    # This ignores the "apostrophe" which is little problematic (raises error when saving to pdf)
    txt_to_save = (joined_summary.encode('latin1', 'ignore')).decode("latin1")

    # Kind of  post-processing.
    txt_to_save_prep = prep_b4_save(txt_to_save)

    emit('summary', {'text': txt_to_save_prep})

    # Calculate the readability factors of the summary
    calculate_scores(txt_to_save_prep)

    # Splitting the path based on "/" to get the name of the book or pdf
    spl = pdf_path.split('/')

    # Summary is added at the end i.e. book name is the_alchemist, so it becomes -> the_alchemist_summary.pdf etc.
    file_name = spl[-1][:-4]

    # Save the summary to a text file
    save_to_text(txt_to_save_prep, file_name)


def ack():
    print('Sent to the client')


if __name__ == "__main__":
    socketio.run(app, allow_unsafe_werkzeug=True, debug=True)
