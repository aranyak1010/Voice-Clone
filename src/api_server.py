from flask import Flask, request, jsonify, send_file
import numpy as np
import io
import base64
import soundfile as sf
from pathlib import Path
import tempfile
import logging
from .inference import VoiceAPI

app = Flask(__name__)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Voice API
CONFIG_PATH = "config/config.yaml"
MODELS_DIR = "checkpoints"
voice_api = VoiceAPI(CONFIG_PATH, MODELS_DIR)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "voice-clone-api"})

@app.route('/speakers', methods=['GET'])
def get_speakers():
    """Get list of available speakers."""
    try:
        speakers = voice_api.list_speakers()
        return jsonify({
            "speakers": speakers,
            "count": len(speakers),
            "current_speaker": voice_api.get_current_speaker()
        })
    except Exception as e:
        logger.error(f"Error getting speakers: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/speaker/<speaker_id>', methods=['POST'])
def select_speaker(speaker_id):
    """Select a speaker for synthesis."""
    try:
        success = voice_api.select_speaker(speaker_id)
        if success:
            return jsonify({
                "message": f"Speaker {speaker_id} selected successfully",
                "current_speaker": speaker_id
            })
        else:
            return jsonify({"error": f"Speaker {speaker_id} not found or failed to load"}), 404
    except Exception as e:
        logger.error(f"Error selecting speaker: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/synthesize', methods=['POST'])
def synthesize_speech():
    """Synthesize speech from text."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({"error": "Text field is required"}), 400
        
        text = data['text']
        output_format = data.get('format', 'wav')  # wav or base64
        
        if not voice_api.get_current_speaker():
            return jsonify({"error": "No speaker selected. Please select a speaker first."}), 400
        
        # Synthesize audio
        audio = voice_api.synthesize(text)
        
        if output_format == 'base64':
            # Return audio as base64
            buffer = io.BytesIO()
            sf.write(buffer, audio, 22050, format='WAV')
            buffer.seek(0)
            audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            return jsonify({
                "audio_base64": audio_base64,
                "text": text,
                "speaker": voice_api.get_current_speaker(),
                "sample_rate": 22050
            })
        
        else:
            # Return audio file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio, 22050)
                return send_file(
                    tmp_file.name,
                    as_attachment=True,
                    download_name=f'synthesized_{voice_api.get_current_speaker()}.wav',
                    mimetype='audio/wav'
                )
    
    except Exception as e:
        logger.error(f"Error during synthesis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/synthesize/batch', methods=['POST'])
def synthesize_batch():
    """Synthesize multiple texts in batch."""
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({"error": "texts field is required"}), 400
        
        texts = data['texts']
        if not isinstance(texts, list):
            return jsonify({"error": "texts must be a list"}), 400
        
        if not voice_api.get_current_speaker():
            return jsonify({"error": "No speaker selected. Please select a speaker first."}), 400
        
        results = []
        for i, text in enumerate(texts):
            try:
                audio = voice_api.synthesize(text)
                
                # Convert to base64
                buffer = io.BytesIO()
                sf.write(buffer, audio, 22050, format='WAV')
                buffer.seek(0)
                audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                
                results.append({
                    "index": i,
                    "text": text,
                    "audio_base64": audio_base64,
                    "success": True
                })
            except Exception as e:
                results.append({
                    "index": i,
                    "text": text,
                    "error": str(e),
                    "success": False
                })
        
        return jsonify({
            "results": results,
            "speaker": voice_api.get_current_speaker(),
            "total": len(texts),
            "successful": sum(1 for r in results if r['success'])
        })
    
    except Exception as e:
        logger.error(f"Error during batch synthesis: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/info', methods=['GET'])
def get_info():
    """Get API information."""
    return jsonify({
        "service": "Voice Clone API",
        "version": "1.0.0",
        "current_speaker": voice_api.get_current_speaker(),
        "available_speakers": voice_api.list_speakers(),
        "endpoints": {
            "/health": "Health check",
            "/speakers": "Get available speakers",
            "/speaker/<id>": "Select speaker",
            "/synthesize": "Synthesize single text",
            "/synthesize/batch": "Synthesize multiple texts",
            "/info": "API information"
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
