import whisper
from flask import Flask, request, jsonify
from flask_cors import CORS
from pydub import AudioSegment
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Cargar el modelo Whisper
model = whisper.load_model("base")

@app.route('/procesar_audio', methods=['POST'])
def procesar_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No se envió ningún archivo"}), 400

        audio_file = request.files['audio']
        audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(audio_path)

        audio = AudioSegment.from_file(audio_path, format="webm")
        converted_path = os.path.join(UPLOAD_FOLDER, "audio_converted.wav")
        audio.export(converted_path, format="wav")

        # Usar Whisper para transcribir el audio en español
        result = model.transcribe(converted_path, language="es")
        transcripcion = result['text']

        return jsonify({
            "message": "Audio procesado y convertido exitosamente",
            "path": converted_path,
            "transcripcion": transcripcion
        })

    except Exception as e:
        print(f"Error al procesar el audio: {str(e)}")
        return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
