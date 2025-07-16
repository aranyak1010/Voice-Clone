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

class CelebrityVoiceProcessor:
    """Processor for celebrity voice models with ethical guidelines."""
    
    def __init__(self):
        self.celebrity_profiles = {
            # Political figures - for educational/satirical purposes only
            'politician_1': {
                'name': 'Political Speaker A',
                'characteristics': {
                    'f0_mean': 120.0,
                    'speaking_rate': 'moderate',
                    'accent': 'indian_english',
                    'voice_type': 'authoritative'
                }
            },
            'politician_2': {
                'name': 'Political Speaker B', 
                'characteristics': {
                    'f0_mean': 140.0,
                    'speaking_rate': 'fast',
                    'accent': 'indian_english',
                    'voice_type': 'energetic'
                }
            },
            # Actors - for demonstration purposes only
            'actor_1': {
                'name': 'Bollywood Actor A',
                'characteristics': {
                    'f0_mean': 110.0,
                    'speaking_rate': 'slow',
                    'accent': 'hindi_english',
                    'voice_type': 'baritone'
                }
            },
            # Singers - for voice conversion demo
            'classical_singer': {
                'name': 'Classical Singer',
                'characteristics': {
                    'f0_range': [80, 350],
                    'vocal_style': 'classical',
                    'vibrato_rate': 5.5
                }
            },
            'modern_singer': {
                'name': 'Modern Singer',
                'characteristics': {
                    'f0_range': [100, 400],
                    'vocal_style': 'contemporary',
                    'vibrato_rate': 4.0
                }
            }
        }
    
    def get_celebrity_list(self) -> List[Dict]:
        """Get list of available celebrity voice profiles."""
        return [
            {
                'id': key,
                'name': profile['name'],
                'type': self._get_celebrity_type(key)
            }
            for key, profile in self.celebrity_profiles.items()
        ]
    
    def _get_celebrity_type(self, celebrity_id: str) -> str:
        """Determine celebrity type for UI categorization."""
        if 'politician' in celebrity_id:
            return 'political'
        elif 'actor' in celebrity_id:
            return 'entertainment'
        elif 'singer' in celebrity_id:
            return 'music'
        else:
            return 'general'
    
    def get_voice_characteristics(self, celebrity_id: str) -> Dict:
        """Get voice characteristics for a celebrity."""
        return self.celebrity_profiles.get(celebrity_id, {}).get('characteristics', {})

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

class DatasetProcessor:
    def __init__(self, config_path: str):
        self.audio_processor = AudioProcessor(config_path)
        self.text_processor = TextProcessor()
        self.celebrity_processor = CelebrityVoiceProcessor()
        self.singing_processor = SingingVoiceProcessor()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def process_voice_data(self, data_dir: str, speaker_id: str, voice_type: str = 'speech') -> Dict:
        """Process voice recordings with enhanced features for celebrity/singing voices."""
        speaker_data = {
            'speaker_id': speaker_id,
            'voice_type': voice_type,
            'audio_files': [],
            'transcripts': [],
            'mel_spectrograms': [],
            'text_sequences': [],
            'vocal_features': [],
            'singing_analysis': None
        }
        
        audio_dir = Path(data_dir) / speaker_id / 'audio'
        transcript_path = Path(data_dir) / speaker_id / 'transcripts.txt'
        
        # Load transcripts
        transcripts = {}
        if transcript_path.exists():
            with open(transcript_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '|' in line:
                        filename, transcript = line.strip().split('|', 1)
                        transcripts[filename] = transcript
        
        # Process audio files with enhanced features
        for audio_file in audio_dir.glob('*.wav'):
            filename = audio_file.stem
            
            if filename in transcripts or voice_type == 'singing':
                try:
                    # Process audio
                    audio = self.audio_processor.load_audio(str(audio_file))
                    audio = self.audio_processor.normalize_audio(audio)
                    
                    # Extract vocal features
                    vocal_features = self.audio_processor.extract_vocal_features(audio)
                    
                    # Check if singing voice
                    is_singing = self.audio_processor.is_singing_voice(audio)
                    
                    if is_singing or voice_type == 'singing':
                        # Process as singing voice
                        vocals, background = self.audio_processor.separate_vocals(audio)
                        singing_analysis = self.singing_processor.analyze_singing_style(vocals)
                        speaker_data['singing_analysis'] = singing_analysis
                        audio = vocals  # Use separated vocals for training
                    
                    mel_spec = self.audio_processor.audio_to_mel(audio)
                    
                    # Process text if available
                    if filename in transcripts:
                        transcript = transcripts[filename]
                        text_sequence = self.text_processor.text_to_sequence(transcript)
                    else:
                        # For singing, create placeholder text sequence
                        transcript = "singing voice sample"
                        text_sequence = self.text_processor.text_to_sequence(transcript)
                    
                    speaker_data['audio_files'].append(str(audio_file))
                    speaker_data['transcripts'].append(transcript)
                    speaker_data['mel_spectrograms'].append(mel_spec)
                    speaker_data['text_sequences'].append(text_sequence)
                    speaker_data['vocal_features'].append(vocal_features)
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
        
        return speaker_data
    
    def save_processed_data(self, speaker_data: Dict, output_dir: str):
        """Save processed data with enhanced features."""
        output_path = Path(output_dir) / speaker_data['speaker_id']
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata with enhanced features
        metadata = {
            'speaker_id': speaker_data['speaker_id'],
            'voice_type': speaker_data.get('voice_type', 'speech'),
            'num_samples': len(speaker_data['audio_files']),
            'audio_files': speaker_data['audio_files'],
            'transcripts': speaker_data['transcripts'],
            'singing_analysis': speaker_data.get('singing_analysis'),
            'average_vocal_features': self._average_vocal_features(speaker_data['vocal_features'])
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save features
        torch.save({
            'mel_spectrograms': speaker_data['mel_spectrograms'],
            'text_sequences': speaker_data['text_sequences'],
            'vocal_features': speaker_data['vocal_features']
        }, output_path / 'features.pt')
    
    def _average_vocal_features(self, vocal_features_list: List[Dict]) -> Dict:
        """Calculate average vocal features across all samples."""
        if not vocal_features_list:
            return {}
        
        avg_features = {}
        for key in vocal_features_list[0].keys():
            try:
                values = [features[key] for features in vocal_features_list if key in features]
                if values:
                    avg_features[key] = float(torch.mean(torch.stack(values)))
            except:
                avg_features[key] = None
        
        return avg_features
    
    def process_celebrity_voice(self, celebrity_id: str, audio_samples: List[str]) -> Dict:
        """Process celebrity voice samples with ethical guidelines."""
        # Add disclaimer and ethical guidelines
        disclaimer = {
            'warning': 'This is for educational/demonstration purposes only.',
            'ethics': 'Do not use for impersonation, fraud, or harmful purposes.',
            'consent': 'Ensure you have proper rights to use this voice data.'
        }
        
        celebrity_characteristics = self.celebrity_processor.get_voice_characteristics(celebrity_id)
        
        # Process samples with celebrity-specific parameters
        processed_data = {
            'celebrity_id': celebrity_id,
            'disclaimer': disclaimer,
            'characteristics': celebrity_characteristics,
            'samples': []
        }
        
        for audio_path in audio_samples:
            try:
                audio = self.audio_processor.load_audio(audio_path)
                features = self.audio_processor.extract_vocal_features(audio)
                processed_data['samples'].append({
                    'audio_path': audio_path,
                    'features': features
                })
            except Exception as e:
                print(f"Error processing celebrity sample {audio_path}: {e}")
        
        return processed_data
