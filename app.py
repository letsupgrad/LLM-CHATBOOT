from flask import Flask, request, jsonify
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

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def text_to_speech(text):
    """Convert text to speech and return audio file path"""
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    engine.save_to_file(text, path)
    engine.runAndWait()
    return path

def speech_to_text(audio_path):
    """Convert speech to text using speech_recognition"""
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            return "Could not understand audio"
        except sr.RequestError:
            return "Speech recognition service unavailable"

def get_llm_response(prompt):
    """Generate response using local LLM"""
    result = generator(prompt, max_length=100, num_return_sequences=1)
    return result[0]["generated_text"]

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
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
        os.remove(audio_path)  # Clean up
        
        return jsonify({"text": text})
    
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/api/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    audio_path = text_to_speech(text)
    
    # Return the audio file
    return send_file(audio_path, mimetype='audio/mp3', as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
