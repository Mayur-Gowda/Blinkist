"""This module defines the socket paths facilitating real-time communication between the frontend and backend."""

import eventlet
from .functions import *
from .routes import pdf_path
from flask_socketio import SocketIO, emit

eventlet.monkey_patch()

global processed_text
global chunk
global all_summaries
global file_name
global txt_to_save_prep


download_path = {
    'path': ''
}


socketio = SocketIO(ping_interval=1440, ping_timeout=1800)


@socketio.on('connect')
def connect():
    print("Connected")
    emit('status', {'text': 'Summarization started<br><br>Preprocessing Text ...'})


@socketio.on('preprocess')
def pre(text):
    global processed_text
    print(text['text'])

    # Preprocess the text
    processed_text = preprocess(pdf_path['path'])

    socketio.emit('pr-status', {'text': 'Chunking text.', 'num': 2})


@socketio.on('chunk')
def chunking(text):
    global chunk
    print(text['text'])

    # Chunk the large text into small parts, so it can be supplied to the model
    chunk = text_chunking(processed_text)


@socketio.on('chunk_summarize')
def chunk_sum(text):
    global all_summaries
    print(text['text'])
    emit('summ_chunk', {'text': "Summarizing the text. Please wait .......<br>", 'num': 1}, callback=ack)
    time.sleep(0.1)

    # Passing the chunks to the model for the summarization
    all_summaries = model_summary(chunk)


@socketio.on('post_process')
def post_process(text):
    global file_name
    global txt_to_save_prep
    print(text['text'])
    emit('end_phase', {'text': 'Post Processing<br><br>', 'num': 1})
    # Combine all chunks of summaries to a single one
    joined_summary = ' '.join([summ['summary_text'] for summ in all_summaries])

    # This ignores the "apostrophe" which is little problematic (raises error when saving to pdf)
    txt_to_save = (joined_summary.encode('latin1', 'ignore')).decode("latin1")

    # Kind of  post-processing.
    txt_to_save_prep = prep_b4_save(txt_to_save)

    # Calculate the readability factors of the summary
    calculate_scores(txt_to_save_prep)

    # Splitting the path based on "/" to get the name of the book or pdf
    spl = pdf_path['path'].split('/')

    # Summary is added at the end i.e. book name is the_alchemist, so it becomes -> the_alchemist_summary.pdf etc.
    file_name = spl[-1][:-4]

    download_path['path'] = file_name

    # Save the summary to a text file
    save_to_text(txt_to_save_prep, file_name)

    emit('summary', {'text': txt_to_save_prep})
