"""This module houses key functions for text summarization and audio generation."""

import PyPDF2
import re
import time
import pyttsx3
from flask_socketio import emit
from transformers import pipeline
from textstat import textstat as ts


ALLOWED_EXTENSIONS = {'pdf'}

# Using the Facebook BART model for summarization (Two options: bart-base & bart-large-cnn)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def ack():
    """Callback function for acknowledgment."""
    print('Sent to the client')


def allowed_file(filename):
    """
    Check if the file extension is allowed.

    Parameters:
    -----------
    filename : str
        The name of the file.

    Returns:
    --------
    bool
        True if the file extension is allowed, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess(path: str):
    """
    Create processed text from a PDF file.

    Takes a path string for the PDF and converts it to clean text
    and then preprocesses it by removing newlines, punctuations, and other characters.

    Parameters:
    -----------
    path : str
        The path for the PDF file.

    Returns:
    --------
    str
        The preprocessed text.
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
    # Remove links (URLs)
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove excessive spaces in the text
    text = ' '.join(text.split())

    emit('pr-status', {'text': "Preprocessing Complete.<br><br.", 'num': 1})
    return text


def text_chunking(new_text: str):
    """
    Split a long piece of text into smaller chunks based on sentence boundaries
    while ensuring that each chunk's word count does not exceed a maximum limit.

    Parameters:
    -----------
    new_text : str
        The preprocessed text of the file.

    Returns:
    --------
    list
        List containing sub-chunks of the sentences.
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
    emit('chunking', {'text': "Total chunks of text are: " + str(len(chunks)) + "<br><br>", 'num': 1})
    return chunks


def model_summary(chunks: list):
    """
    Create a summary of the chunks and combine them into a single summary.

    Summarizes a chunk and combines it into a bigger summary.

    Parameters:
    -----------
    chunks : list
        List containing the sub-chunks of the sentences.

    Returns:
    --------
    list
        List containing summaries of each chunk.
    """

    summaries = []
    count = 0

    # Summarize each chunk
    for chnk in chunks:
        print(f"Summarizing Chunk NO: {count + 1}")
        emit('summ_chunk', {'text': 'Summarizing Chunk NO.:' + str(count + 1),
                            'num': 2, 'count': count + 1}, callback=ack)
        time.sleep(0.1)
        res = summarizer(chnk, max_length=60, min_length=20, do_sample=False)
        summaries += res
        count += 1
    emit('chunk_done', {'text': '\nEach chunk summarized.'})
    return summaries


def prep_b4_save(text: str):
    """
    Format the text for proper readability.

    Parameters:
    -----------
    text : str
        Summary text for formatting.

    Returns:
    --------
    str
        Formatted text.
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
    """
    Calculate the readability and coherence of the summary.

    Uses Flesch-Kincaid Grade, Gunning Fog index, Coleman-Liau index.

    Parameters:
    -----------
    text : str
        The post-processed text of the file.
    """

    # Flesch-Kincaid Grade Level, Gunning Fog Index, and Coleman-Liau Index readability metrics to evaluate
    # the readability of your text or summaries.
    # These metrics provide an estimate of the grade level required to understand a piece of text.
    # Lower grade levels indicate easier readability

    emit('end_phase', {'text': "Calculating scores...<br>", 'num': 2})

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

    emit('end_phase', {'text': f"Flesch-Kincaid Grade: {grade_level}<br> \
          Gunning Fog Index: {fog_index}<br> \
          Coleman-Liau Index: {coleman_liau}<br><br>", 'num': 3})


def save_to_text(text: str, name: str):
    """
    Create a text file of the summary produced.

    Creates a text file and saves it in the current working directory.

    Parameters:
    -----------
    text : str
        The summarized text that is needed to save into the text file.

    name : str
        The name of the book (Name of the book when input is taken here).
    """
    with open(f"{name}_summary.txt", 'w') as txt:
        txt.write(text)


def gen_audio(file_name, text):
    """
    Generate an audio file from the text using text-to-speech.

    Parameters:
    -----------
    file_name : str
        Name of the audio file to be generated.

    text : str
        Text to be converted into audio.
    """
    # Using pyttsx3 Text-to-Speech
    engine = pyttsx3.init()

    # Adjust speech speed
    engine.setProperty('rate', 150)

    # Save location of the file
    engine.save_to_file(text, f"uploads/{file_name}_audiobook.mp3")
    engine.runAndWait()
