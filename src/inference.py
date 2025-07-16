import torch
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import yaml
import json
from typing import List, Optional
import logging
from .model import Tacotron2
from .data_processor import TextProcessor, AudioProcessor

class VoiceInference:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor(config_path)
        
        # Initialize empty model (will be loaded when speaker is selected)
        self.model = None
        self.current_speaker = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_speaker_model(self, model_path: str, speaker_id: str):
        """Load a specific speaker's model."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model with config from checkpoint
            model_config = checkpoint.get('config', {}).get('model', self.config['model'])
            self.model = Tacotron2(model_config).to(self.device)
            
            # Load model weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.current_speaker = speaker_id
            self.logger.info(f'Loaded model for speaker: {speaker_id}')
            
        except Exception as e:
            self.logger.error(f'Error loading model: {e}')
            raise e
    
    def text_to_mel(self, text: str) -> torch.Tensor:
        """Convert text to mel spectrogram."""
        if self.model is None:
            raise ValueError("No model loaded. Please load a speaker model first.")
        
        # Process text
        text_sequence = self.text_processor.text_to_sequence(text)
        text_tensor = torch.LongTensor(text_sequence).unsqueeze(0).to(self.device)
        text_lengths = torch.LongTensor([len(text_sequence)]).to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            mel_outputs, mel_outputs_postnet, gate_outputs, alignments = self.model.inference(
                text_tensor, text_lengths
            )
        
        return mel_outputs_postnet.squeeze(0)
    
    def mel_to_audio(self, mel_spectrogram: torch.Tensor) -> np.ndarray:
        """Convert mel spectrogram to audio using Griffin-Lim algorithm."""
        # Convert mel to linear spectrogram (simplified approach)
        mel_spec = mel_spectrogram.cpu().numpy()
        
        # Inverse mel scale
        mel_basis = librosa.filters.mel(
            sr=self.config['audio']['sample_rate'],
            n_fft=self.config['audio']['n_fft'],
            n_mels=self.config['audio']['n_mel_channels'],
            fmin=self.config['audio']['mel_fmin'],
            fmax=self.config['audio']['mel_fmax']
        )
        
        # Convert to linear scale
        linear_spec = np.dot(np.linalg.pinv(mel_basis), mel_spec)
        
        # Convert to magnitude spectrogram
        magnitude = np.exp(linear_spec)
        
        # Griffin-Lim algorithm
        audio = librosa.griffinlim(
            magnitude,
            n_iter=60,
            hop_length=self.config['audio']['hop_length'],
            win_length=self.config['audio']['win_length']
        )
        
        return audio
    
    def synthesize_speech(self, text: str, output_path: Optional[str] = None) -> np.ndarray:
        """Complete text-to-speech synthesis."""
        if self.model is None:
            raise ValueError("No model loaded. Please load a speaker model first.")
        
        self.logger.info(f'Synthesizing speech for: "{text}"')
        
        try:
            # Text to mel
            mel_spectrogram = self.text_to_mel(text)
            
            # Mel to audio
            audio = self.mel_to_audio(mel_spectrogram)
            
            # Save audio if path provided
            if output_path:
                sf.write(
                    output_path, 
                    audio, 
                    self.config['audio']['sample_rate']
                )
                self.logger.info(f'Audio saved to: {output_path}')
            
            return audio
            
        except Exception as e:
            self.logger.error(f'Error during synthesis: {e}')
            raise e
    
    def get_available_speakers(self, models_dir: str) -> List[str]:
        """Get list of available speaker models."""
        models_path = Path(models_dir)
        speakers = []
        
        for model_file in models_path.glob('*.pt'):
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                speaker_id = checkpoint.get('speaker_id', model_file.stem)
                speakers.append(speaker_id)
            except:
                continue
        
        return speakers

# Enhanced model with inference method
class Tacotron2Enhanced(Tacotron2):
    def inference(self, inputs, input_lengths):
        """Inference method for text-to-speech generation."""
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, input_lengths)
        
        mel_outputs, gate_outputs, alignments = self.decoder.inference(encoder_outputs)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        return mel_outputs, mel_outputs_postnet, gate_outputs, alignments

# Enhanced decoder with inference method
class DecoderInference:
    def inference(self, memory):
        """Inference method for decoder."""
        device = memory.device
        B = memory.size(0)
        
        # Initialize decoder states
        decoder_input = self.get_go_frame(memory)
        self.initialize_decoder_states(memory, mask=None)
        
        mel_outputs, gate_outputs, alignments = [], [], []
        
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, attention_weights = self.decode(decoder_input)
            
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [attention_weights]
            
            # Check for end of sequence
            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) >= self.max_decoder_steps:
                self.logger.warning("Reached maximum decoder steps")
                break
            
            decoder_input = mel_output
        
        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        
        return mel_outputs, gate_outputs, alignments

class VoiceAPI:
    def __init__(self, config_path: str, models_dir: str):
        self.inference_engine = VoiceInference(config_path)
        self.models_dir = models_dir
        self.available_speakers = self.inference_engine.get_available_speakers(models_dir)
    
    def list_speakers(self) -> List[str]:
        """Return list of available speakers."""
        return self.available_speakers
    
    def select_speaker(self, speaker_id: str) -> bool:
        """Select a speaker for synthesis."""
        if speaker_id not in self.available_speakers:
            return False
        
        model_path = Path(self.models_dir) / f'{speaker_id}_best.pt'
        if not model_path.exists():
            # Try to find any model for this speaker
            model_files = list(Path(self.models_dir).glob(f'{speaker_id}*.pt'))
            if not model_files:
                return False
            model_path = model_files[0]
        
        try:
            self.inference_engine.load_speaker_model(str(model_path), speaker_id)
            return True
        except:
            return False
    
    def synthesize(self, text: str, output_path: Optional[str] = None) -> np.ndarray:
        """Synthesize speech from text."""
        return self.inference_engine.synthesize_speech(text, output_path)
    
    def get_current_speaker(self) -> Optional[str]:
        """Get currently selected speaker."""
        return self.inference_engine.current_speaker
