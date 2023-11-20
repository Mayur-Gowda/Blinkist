"""This module contains the accessible routes by the flask application"""

import os
import shutil
from flask import render_template, redirect, request, Blueprint, url_for, current_app, send_file, jsonify
from .functions import allowed_file, gen_audio
from werkzeug.utils import secure_filename

main = Blueprint("main", __name__)

UPLOAD_FOLDER = 'uploads'

pdf_path = {
    'path': ''
}

download_path = {
    'path': '',
    'file_name': '',
    'audio_path': ''
}


@main.route('/')
def home():
    """Render the home page."""
    return render_template('home.html')


@main.route('/create', methods=['GET', 'POST'])
def create():
    """Render the create page and handle file upload and redirect to the summarization page."""
    if request.method == 'POST':
        current_app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
        if not request.files:
            return redirect(request.url)
        print(request.files['file'])
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            pdf_path['path'] = file_path
            spl = file_path.split('\\')
            file_name = spl[-1][:-4]
            download_path['file_name'] = f"{file_name}"
            download_path['path'] = f"uploads/{file_name}_summary.txt"
            return redirect(url_for('main.summarize'))

    return render_template("create.html", layout='create')


@main.route('/summarize')
def summarize():
    """Render the summarization page."""
    return render_template('summarize_page.html')


@main.route('/summ_download', methods=['POST', 'GET'])
def download():
    """Download the summary file."""
    full_path = os.path.abspath(download_path['path'])
    return send_file(full_path,
                     download_name=f"{download_path['file_name']}_summary.txt",
                     as_attachment=True)


@main.route('/audio')
def audio():
    """Render the audio generation page."""
    return render_template('create.html', layout='audio')


@main.route('/audio_gen')
def audio_gen():
    """Generate audio from the summary and return a JSON response."""
    file_name = download_path['file_name']
    with open(download_path['path']) as txt:
        text = txt.read()

    gen_audio(file_name, text)

    file_path = f"uploads/{file_name}_audioBook.mp3"
    download_path['audio_path'] = file_path

    response_data = {
        'status': 'success',
        'message': 'Audio file generated successfully',
        'file_path': file_path
    }

    return jsonify(response_data)


@main.route('/audio_download')
def audio_download():
    """Download the generated audio file."""
    full_path = os.path.abspath(download_path['audio_path'])
    return send_file(full_path,
                     download_name=f"{download_path['file_name']}_audiobook.mp3",
                     as_attachment=True)


@main.route('/admin')
def admin():
    """Admin route used for clearing the contents of the 'uploads' folder."""
    folder_path = 'uploads'
    shutil.rmtree(folder_path, ignore_errors=True)
    os.makedirs(folder_path, exist_ok=True)
    return redirect(url_for('main.home'))
