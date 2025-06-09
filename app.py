from flask import Flask, request, jsonify, send_file
import pyttsx3
import tempfile
import os
import speech_recognition as sr
from transformers import pipeline
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Initialize components
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B")
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mandatory health check endpoint for Choreo
@app.route('/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def text_to_speech(text):
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    engine.save_to_file(text, path)
    engine.runAndWait()
    return path

def speech_to_text(audio_path):
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Speech recognition service unavailable"

def get_llm_response(prompt):
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return result[0]["generated_text"]

@app.route('/api/chat', methods=['POST'])
def chat():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    response = get_llm_response(prompt)
    return jsonify({"response": response})

@app.route('/api/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(audio_path)
        
        text = speech_to_text(audio_path)
        os.remove(audio_path)
        return jsonify({"text": text})
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    audio_path = text_to_speech(text)
    return send_file(audio_path, mimetype='audio/mp3')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Choreo will provide PORT
    app.run(host='0.0.0.0', port=port)
