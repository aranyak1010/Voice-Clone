import os
import librosa
import numpy as np
import torch
import torchaudio
from pathlib import Path
import json
import re
from typing import List, Tuple, Dict
from pydub import AudioSegment
import yaml
import torchaudio.functional as F
from scipy.signal import find_peaks

class AudioProcessor:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sample_rate = self.config['audio']['sample_rate']
        self.n_fft = self.config['audio']['n_fft']
        self.hop_length = self.config['audio']['hop_length']
        self.win_length = self.config['audio']['win_length']
        self.n_mel_channels = self.config['audio']['n_mel_channels']
    
    def load_audio(self, audio_path: str) -> torch.Tensor:
        """Load and preprocess audio file."""
        # Load audio
        audio, sr = torchaudio.load(audio_path)
        
        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        
        return audio.squeeze(0)
    
    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to mel spectrogram."""
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mel_channels,
            f_min=self.config['audio']['mel_fmin'],
            f_max=self.config['audio']['mel_fmax']
        )
        
        mel_spec = mel_transform(audio)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))
        return mel_spec
    
    def normalize_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range."""
        return audio / torch.max(torch.abs(audio))
    
    def extract_vocal_features(self, audio: torch.Tensor) -> Dict:
        """Extract vocal characteristics for voice cloning."""
        features = {}
        
        # Fundamental frequency (F0) extraction
        f0 = self.extract_f0(audio)
        features['f0_mean'] = torch.mean(f0[f0 > 0])
        features['f0_std'] = torch.std(f0[f0 > 0])
        features['f0_range'] = torch.max(f0) - torch.min(f0[f0 > 0])
        
        # Spectral features
        stft = torch.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length, 
                         win_length=self.win_length, return_complex=True)
        magnitude = torch.abs(stft)
        
        # Spectral centroid
        freqs = torch.fft.fftfreq(self.n_fft, 1/self.sample_rate)[:self.n_fft//2+1]
        spectral_centroid = torch.sum(magnitude * freqs.unsqueeze(1), dim=0) / torch.sum(magnitude, dim=0)
        features['spectral_centroid_mean'] = torch.mean(spectral_centroid)
        
        # Spectral rolloff
        cumulative_magnitude = torch.cumsum(magnitude, dim=0)
        total_magnitude = torch.sum(magnitude, dim=0)
        rolloff_threshold = 0.85 * total_magnitude
        rolloff_indices = torch.argmax((cumulative_magnitude >= rolloff_threshold).float(), dim=0)
        spectral_rolloff = freqs[rolloff_indices]
        features['spectral_rolloff_mean'] = torch.mean(spectral_rolloff)
        
        # Zero crossing rate
        zcr = torch.mean(torch.abs(torch.diff(torch.sign(audio))))
        features['zero_crossing_rate'] = zcr
        
        return features
    
    def extract_f0(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract fundamental frequency using autocorrelation."""
        # Convert to numpy for librosa processing
        audio_np = audio.numpy()
        f0 = librosa.yin(audio_np, fmin=80, fmax=400, sr=self.sample_rate)
        return torch.from_numpy(f0)
    
    def is_singing_voice(self, audio: torch.Tensor) -> bool:
        """Detect if audio contains singing voice vs speaking voice."""
        features = self.extract_vocal_features(audio)
        
        # Singing typically has:
        # - Higher F0 variance
        # - Sustained tones
        # - More melodic patterns
        
        f0_variation = features['f0_std'] / features['f0_mean']
        
        # Simple heuristic - can be improved with ML model
        is_singing = f0_variation > 0.3 and features['f0_range'] > 100
        
        return is_singing
    
    def separate_vocals(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Separate vocals from background music using simple spectral subtraction."""
        # Convert to stereo if mono
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0).repeat(2, 1)
        
        # Simple vocal isolation using center channel extraction
        if audio.shape[0] == 2:
            # Subtract left from right to isolate center vocals
            vocals = (audio[0] + audio[1]) / 2
            background = (audio[0] - audio[1]) / 2
        else:
            vocals = audio[0]
            background = torch.zeros_like(vocals)
        
        return vocals, background

class TextProcessor:
    def __init__(self):
        self.cleaners = ['english_cleaners']
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters except punctuation
        text = re.sub(r'[^\w\s.,!?;:]', '', text)
        
        return text.strip()
    
    def text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of integers."""
        # Simple character-level encoding
        char_to_id = {char: idx for idx, char in enumerate(" abcdefghijklmnopqrstuvwxyz.,!?;:")}
        
        cleaned_text = self.clean_text(text)
        sequence = [char_to_id.get(char, 0) for char in cleaned_text]
        
        return sequence
    
    def extract_phonemes(self, text: str) -> List[str]:
        """Extract phonemes from text for better voice cloning."""
        try:
            from phonemizer import phonemize
            phonemes = phonemize(text, language='en-us', backend='espeak')
            return phonemes.split()
        except ImportError:
            # Fallback to character-based if phonemizer not available
            return list(self.clean_text(text))

class SingingVoiceProcessor:
    """Processor for singing voice conversion."""
    
    def __init__(self):
        self.singer_styles = {
            'classical_male': {
                'name': 'Classical Male Singer',
                'f0_range': [80, 300],
                'vibrato_rate': 5.5,
                'vocal_technique': 'operatic'
            },
            'classical_female': {
                'name': 'Classical Female Singer',
                'f0_range': [200, 800],
                'vibrato_rate': 6.0,
                'vocal_technique': 'operatic'
            },
            'jazz_singer': {
                'name': 'Jazz Singer',
                'f0_range': [100, 400],
                'vibrato_rate': 4.0,
                'vocal_technique': 'breathy'
            },
            'rock_singer': {
                'name': 'Rock Singer',
                'f0_range': [80, 500],
                'vibrato_rate': 3.0,
                'vocal_technique': 'powerful'
            },
            'pop_singer': {
                'name': 'Pop Singer',
                'f0_range': [150, 450],
                'vibrato_rate': 4.5,
                'vocal_technique': 'smooth'
            }
        }
    
    def analyze_singing_style(self, audio: torch.Tensor) -> Dict:
        """Analyze singing style characteristics."""
        audio_processor = AudioProcessor('config/config.yaml')
        features = audio_processor.extract_vocal_features(audio)
        
        # Detect vibrato
        f0 = audio_processor.extract_f0(audio)
        vibrato_rate = self._detect_vibrato(f0)
        
        # Classify vocal technique
        vocal_technique = self._classify_vocal_technique(features)
        
        return {
            'vibrato_rate': vibrato_rate,
            'vocal_technique': vocal_technique,
            'f0_characteristics': features,
            'recommended_style': self._recommend_singer_style(features, vibrato_rate)
        }
    
    def _detect_vibrato(self, f0: torch.Tensor) -> float:
        """Detect vibrato rate in Hz."""
        # Remove silent frames
        voiced_f0 = f0[f0 > 0]
        if len(voiced_f0) < 10:
            return 0.0
        
        # Simple vibrato detection using autocorrelation
        f0_np = voiced_f0.numpy()
        autocorr = np.correlate(f0_np, f0_np, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Find peaks in autocorrelation
        peaks, _ = find_peaks(autocorr[1:], height=0.3 * np.max(autocorr))
        
        if len(peaks) > 0:
            # Convert to Hz (assuming hop_length corresponds to time)
            vibrato_rate = 22050 / (256 * peaks[0])  # Approximate conversion
            return min(vibrato_rate, 10.0)  # Cap at reasonable vibrato rate
        
        return 0.0
    
    def _classify_vocal_technique(self, features: Dict) -> str:
        """Classify vocal technique based on features."""
        zcr = features.get('zero_crossing_rate', 0)
        spectral_centroid = features.get('spectral_centroid_mean', 0)
        
        if zcr > 0.1:
            return 'breathy'
        elif spectral_centroid > 2000:
            return 'bright'
        elif features.get('f0_range', 0) > 200:
            return 'powerful'
        else:
            return 'smooth'
    
    def _recommend_singer_style(self, features: Dict, vibrato_rate: float) -> str:
        """Recommend singer style based on analysis."""
        f0_mean = features.get('f0_mean', 150)
        f0_range = features.get('f0_range', 100)
        
        if vibrato_rate > 5.0 and f0_range > 300:
            return 'classical_male' if f0_mean < 200 else 'classical_female'
        elif vibrato_rate > 3.0 and vibrato_rate < 5.0:
            return 'jazz_singer'
        elif f0_range > 250:
            return 'rock_singer'
        else:
            return 'pop_singer'
