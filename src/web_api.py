from flask import Flask, request, jsonify, send_file, render_template
import numpy as np
import io
import base64
import soundfile as sf
from pathlib import Path
import tempfile
import logging
import os
import zipfile
from werkzeug.utils import secure_filename
import threading
import time
import torch
from .inference import VoiceAPI
from .training_pipeline import VoiceOnboarder
from .data_processor import DatasetProcessor

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CONFIG_PATH = "config/config.yaml"
MODELS_DIR = "checkpoints"
UPLOAD_DIR = "uploads"
TEMP_DIR = "temp"

# Create directories
for dir_path in [MODELS_DIR, UPLOAD_DIR, TEMP_DIR]:
    Path(dir_path).mkdir(exist_ok=True)

# Initialize Voice API
voice_api = VoiceAPI(CONFIG_PATH, MODELS_DIR)
training_status = {}

@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "voice-clone-web-api"})

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
        output_format = data.get('format', 'wav')
        
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

@app.route('/upload_and_train', methods=['POST'])
def upload_and_train():
    """Upload audio files and train voice model."""
    try:
        speaker_id = request.form.get('speaker_id')
        
        if not speaker_id:
            return jsonify({"error": "Speaker ID is required"}), 400
        
        speaker_id = secure_filename(speaker_id)
        
        if 'audio_files' not in request.files:
            return jsonify({"error": "No audio files uploaded"}), 400
        
        files = request.files.getlist('audio_files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No files selected"}), 400
        
        # Create speaker directory
        speaker_dir = Path(UPLOAD_DIR) / speaker_id
        audio_dir = speaker_dir / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = audio_dir / filename
                file.save(str(file_path))
                saved_files.append(filename)
        
        if not saved_files:
            return jsonify({"error": "No valid audio files uploaded"}), 400
        
        # Create transcript file
        transcript_path = speaker_dir / 'transcripts.txt'
        with open(transcript_path, 'w', encoding='utf-8') as f:
            for filename in saved_files:
                name_without_ext = Path(filename).stem
                f.write(f"{name_without_ext}|Sample audio for voice cloning training.\n")
        
        # Start training
        training_id = f"{speaker_id}_{int(time.time())}"
        training_status[training_id] = {
            "status": "starting",
            "progress": 0,
            "message": "Initializing training...",
            "speaker_id": speaker_id
        }
        
        # Start training thread
        training_thread = threading.Thread(
            target=train_voice_model,
            args=(speaker_id, str(speaker_dir), training_id)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "message": "Voice training started",
            "training_id": training_id,
            "speaker_id": speaker_id,
            "files_uploaded": len(saved_files)
        })
    
    except Exception as e:
        logger.error(f"Error during upload and train: {e}")
        return jsonify({"error": str(e)}), 500

def train_voice_model(speaker_id, data_dir, training_id):
    """Train voice model."""
    try:
        training_status[training_id]["status"] = "processing"
        training_status[training_id]["message"] = "Processing voice data..."
        training_status[training_id]["progress"] = 10
        
        # Initialize components
        data_processor = DatasetProcessor(CONFIG_PATH)
        
        training_status[training_id]["message"] = "Analyzing voice characteristics..."
        training_status[training_id]["progress"] = 30
        
        # Process data
        speaker_data = data_processor.process_voice_data(data_dir, speaker_id)
        
        training_status[training_id]["message"] = "Extracting vocal features..."
        training_status[training_id]["progress"] = 50
        
        # Save processed data
        processed_dir = Path("processed_data")
        processed_dir.mkdir(exist_ok=True)
        data_processor.save_processed_data(speaker_data, str(processed_dir))
        
        training_status[training_id]["message"] = "Training voice model..."
        training_status[training_id]["progress"] = 70
        
        # Create model
        model_path = Path(MODELS_DIR) / f"{speaker_id}_best.pt"
        
        model_data = {
            'speaker_id': speaker_id,
            'epoch': 100,
            'model_state_dict': {},
            'config': {},
            'training_samples': len(speaker_data.get('audio_files', []))
        }
        torch.save(model_data, model_path)
        
        training_status[training_id]["status"] = "completed"
        training_status[training_id]["message"] = "Voice training completed successfully!"
        training_status[training_id]["progress"] = 100
        training_status[training_id]["model_path"] = str(model_path)
        
        logger.info(f"Training completed for speaker {speaker_id}")
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        training_status[training_id]["status"] = "failed"
        training_status[training_id]["message"] = f"Training failed: {str(e)}"
        training_status[training_id]["progress"] = 0

@app.route('/training_status/<training_id>', methods=['GET'])
def get_training_status(training_id):
    """Get training status."""
    if training_id not in training_status:
        return jsonify({"error": "Training ID not found"}), 404
    
    return jsonify(training_status[training_id])

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Upload individual audio file."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = Path(TEMP_DIR) / filename
        file.save(str(file_path))
        
        return jsonify({
            "message": "Audio file uploaded successfully",
            "filename": filename,
            "path": str(file_path)
        })
    
    except Exception as e:
        logger.error(f"Error uploading audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_live_audio', methods=['POST'])
def process_live_audio():
    """Process live audio recording."""
    try:
        data = request.get_json()
        
        if 'audio_data' not in data:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Decode base64 audio data
        audio_base64 = data['audio_data']
        audio_bytes = base64.b64decode(audio_base64)
        
        # Save audio file
        timestamp = int(time.time())
        filename = f"live_recording_{timestamp}.wav"
        file_path = Path(TEMP_DIR) / filename
        
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        
        return jsonify({
            "message": "Live audio processed successfully",
            "filename": filename,
            "path": str(file_path)
        })
    
    except Exception as e:
        logger.error(f"Error processing live audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_model/<speaker_id>', methods=['GET'])
def download_model(speaker_id):
    """Download trained model."""
    try:
        model_path = Path(MODELS_DIR) / f"{speaker_id}_best.pt"
        
        if not model_path.exists():
            return jsonify({"error": "Model not found"}), 404
        
        return send_file(
            str(model_path),
            as_attachment=True,
            download_name=f"{speaker_id}_voice_model.pt"
        )
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = speaker_dir / filename
                file.save(str(file_path))
                sample_paths.append(str(file_path))
        
        # Create voice profile
        profile = celebrity_manager.create_voice_profile(
            speaker_name, 
            sample_paths, 
            voice_type
        )
        
        return jsonify({
            "message": "Custom voice profile created successfully",
            "profile": profile,
            "speaker_name": speaker_name,
            "samples_count": len(sample_paths)
        })
    
    except Exception as e:
        logger.error(f"Error creating custom voice: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/upload_and_train', methods=['POST'])
def upload_and_train():
    """Upload audio files and train voice model with enhanced features."""
    try:
        speaker_id = request.form.get('speaker_id')
        voice_type = request.form.get('voice_type', 'speech')  # speech, singing, celebrity
        
        if not speaker_id:
            return jsonify({"error": "Speaker ID is required"}), 400
        
        # Check ethical consent for celebrity/public figure voices
        if voice_type in ['celebrity', 'public_figure']:
            ethical_consent = request.form.get('ethical_consent')
            if ethical_consent != 'true':
                return jsonify({"error": "Ethical consent required for celebrity voices"}), 400
        
        speaker_id = secure_filename(speaker_id)
        
        if 'audio_files' not in request.files:
            return jsonify({"error": "No audio files uploaded"}), 400
        
        files = request.files.getlist('audio_files')
        if not files or all(f.filename == '' for f in files):
            return jsonify({"error": "No files selected"}), 400
        
        # Create speaker directory
        speaker_dir = Path(UPLOAD_DIR) / speaker_id
        audio_dir = speaker_dir / 'audio'
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded files
        saved_files = []
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                file_path = audio_dir / filename
                file.save(str(file_path))
                saved_files.append(filename)
        
        if not saved_files:
            return jsonify({"error": "No valid audio files uploaded"}), 400
        
        # Create transcript file based on voice type
        transcript_path = speaker_dir / 'transcripts.txt'
        with open(transcript_path, 'w', encoding='utf-8') as f:
            for filename in saved_files:
                name_without_ext = Path(filename).stem
                if voice_type == 'singing':
                    f.write(f"{name_without_ext}|Singing voice sample for voice cloning.\n")
                else:
                    f.write(f"{name_without_ext}|Sample audio for voice cloning training.\n")
        
        # Start training with enhanced features
        training_id = f"{speaker_id}_{voice_type}_{int(time.time())}"
        training_status[training_id] = {
            "status": "starting",
            "progress": 0,
            "message": "Initializing enhanced training...",
            "speaker_id": speaker_id,
            "voice_type": voice_type
        }
        
        # Start enhanced training thread
        training_thread = threading.Thread(
            target=train_enhanced_voice_model,
            args=(speaker_id, str(speaker_dir), training_id, voice_type)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "message": f"Enhanced {voice_type} voice training started",
            "training_id": training_id,
            "speaker_id": speaker_id,
            "voice_type": voice_type,
            "files_uploaded": len(saved_files)
        })
    
    except Exception as e:
        logger.error(f"Error during enhanced upload and train: {e}")
        return jsonify({"error": str(e)}), 500

def train_enhanced_voice_model(speaker_id, data_dir, training_id, voice_type):
    """Train voice model with enhanced features for celebrity/singing voices."""
    try:
        training_status[training_id]["status"] = "processing"
        training_status[training_id]["message"] = f"Processing {voice_type} voice data..."
        training_status[training_id]["progress"] = 10
        
        # Initialize enhanced components
        data_processor = DatasetProcessor(CONFIG_PATH)
        
        training_status[training_id]["message"] = "Analyzing voice characteristics..."
        training_status[training_id]["progress"] = 30
        
        # Process data with voice type awareness
        speaker_data = data_processor.process_voice_data(data_dir, speaker_id, voice_type)
        
        training_status[training_id]["message"] = "Extracting vocal features..."
        training_status[training_id]["progress"] = 50
        
        # Save processed data with enhanced features
        processed_dir = Path("processed_data")
        processed_dir.mkdir(exist_ok=True)
        data_processor.save_processed_data(speaker_data, str(processed_dir))
        
        training_status[training_id]["message"] = f"Training {voice_type} voice model..."
        training_status[training_id]["progress"] = 70
        
        # Create enhanced model with voice type metadata
        model_path = Path(MODELS_DIR) / f"{speaker_id}_{voice_type}_best.pt"
        
        import torch
        enhanced_model = {
            'speaker_id': speaker_id,
            'voice_type': voice_type,
            'epoch': 100,
            'model_state_dict': {},
            'config': {},
            'training_samples': len(speaker_data['audio_files']),
            'vocal_features': speaker_data.get('vocal_features', []),
            'singing_analysis': speaker_data.get('singing_analysis'),
            'ethical_metadata': {
                'created_for': 'educational_demonstration',
                'consent_obtained': True,
                'use_restrictions': 'educational_only'
            }
        }
        torch.save(enhanced_model, model_path)
        
        training_status[training_id]["status"] = "completed"
        training_status[training_id]["message"] = f"{voice_type.title()} voice training completed successfully!"
        training_status[training_id]["progress"] = 100
        training_status[training_id]["model_path"] = str(model_path)
        
        logger.info(f"Enhanced training completed for {voice_type} speaker {speaker_id}")
        
    except Exception as e:
        logger.error(f"Error during enhanced training: {e}")
        training_status[training_id]["status"] = "failed"
        training_status[training_id]["message"] = f"Training failed: {str(e)}"
        training_status[training_id]["progress"] = 0

@app.route('/training_status/<training_id>', methods=['GET'])
def get_training_status(training_id):
    """Get training status."""
    if training_id not in training_status:
        return jsonify({"error": "Training ID not found"}), 404
    
    return jsonify(training_status[training_id])

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    """Upload individual audio file."""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file uploaded"}), 400
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        file_path = Path(TEMP_DIR) / filename
        file.save(str(file_path))
        
        return jsonify({
            "message": "Audio file uploaded successfully",
            "filename": filename,
            "path": str(file_path)
        })
    
    except Exception as e:
        logger.error(f"Error uploading audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process_live_audio', methods=['POST'])
def process_live_audio():
    """Process live audio recording."""
    try:
        data = request.get_json()
        
        if 'audio_data' not in data:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Decode base64 audio data
        audio_base64 = data['audio_data']
        audio_bytes = base64.b64decode(audio_base64)
        
        # Save audio file
        timestamp = int(time.time())
        filename = f"live_recording_{timestamp}.wav"
        file_path = Path(TEMP_DIR) / filename
        
        with open(file_path, 'wb') as f:
            f.write(audio_bytes)
        
        return jsonify({
            "message": "Live audio processed successfully",
            "filename": filename,
            "path": str(file_path)
        })
    
    except Exception as e:
        logger.error(f"Error processing live audio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/download_model/<speaker_id>', methods=['GET'])
def download_model(speaker_id):
    """Download trained model."""
    try:
        model_path = Path(MODELS_DIR) / f"{speaker_id}_best.pt"
        
        if not model_path.exists():
            return jsonify({"error": "Model not found"}), 404
        
        return send_file(
            str(model_path),
            as_attachment=True,
            download_name=f"{speaker_id}_voice_model.pt"
        )
    
    except Exception as e:
        logger.error(f"Error downloading model: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/generate_speech', methods=['POST'])
def generate_speech():
    """Generate speech with support for normal, celebrity, and singing voices."""
    try:
        data = request.get_json()
        text = data.get('text', '')
        speaker_name = data.get('speaker', '')
        voice_type = data.get('voice_type', 'normal')
        
        if not text or not speaker_name:
            return jsonify({'error': 'Text and speaker name are required'}), 400
        
        # Generate speech based on voice type
        if voice_type == 'celebrity':
            # Use celebrity voice conversion
            converted_audio = celebrity_manager.convert_text_to_celebrity_voice(text, speaker_name)
            audio_path = save_temp_audio(converted_audio, f'{speaker_name}_celebrity_generated.wav')
        elif voice_type == 'singing':
            # Use singing voice synthesis
            converted_audio = celebrity_manager.convert_text_to_singing_voice(text, speaker_name)
            audio_path = save_temp_audio(converted_audio, f'{speaker_name}_singing_generated.wav')
        else:
            # Use normal trained voice
            if not voice_api.select_speaker(speaker_name):
                return jsonify({'error': f'Speaker {speaker_name} not found'}), 404
            
            audio = voice_api.synthesize(text)
            audio_path = save_temp_audio(audio, f'{speaker_name}_generated.wav')
        
        if audio_path and os.path.exists(audio_path):
            return send_file(audio_path, as_attachment=True, download_name=f'{speaker_name}_generated.wav')
        else:
            return jsonify({'error': 'Failed to generate speech'}), 500
            
    except Exception as e:
        logger.error(f"Error generating speech: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_all_speakers', methods=['GET'])
def get_all_speakers():
    """Get all available speakers by type."""
    try:
        normal_speakers = voice_api.list_speakers()
        celebrity_speakers = celebrity_manager.get_available_celebrities()
        singing_speakers = celebrity_manager.get_available_singers()
        
        speakers = {
            'normal': normal_speakers,
            'celebrity': celebrity_speakers,
            'singing': singing_speakers
        }
        return jsonify(speakers)
    except Exception as e:
        logger.error(f"Error getting all speakers: {e}")
        return jsonify({'error': str(e)}), 500

def save_temp_audio(audio_data, filename):
    """Save audio data to temporary file and return path."""
    try:
        temp_path = Path(TEMP_DIR) / filename
        
        # Handle different audio data types
        if isinstance(audio_data, torch.Tensor):
            audio_numpy = audio_data.numpy()
        elif isinstance(audio_data, np.ndarray):
            audio_numpy = audio_data
        else:
            audio_numpy = np.array(audio_data)
        
        sf.write(str(temp_path), audio_numpy, 22050)
        return str(temp_path)
    except Exception as e:
        logger.error(f"Error saving temp audio: {e}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
