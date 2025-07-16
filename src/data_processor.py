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

class DatasetProcessor:
    def __init__(self, config_path: str):
        self.audio_processor = AudioProcessor(config_path)
        self.text_processor = TextProcessor()
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def process_voice_data(self, data_dir: str, speaker_id: str) -> Dict:
        """Process voice recordings and transcripts for a specific speaker."""
        speaker_data = {
            'speaker_id': speaker_id,
            'audio_files': [],
            'transcripts': [],
            'mel_spectrograms': [],
            'text_sequences': []
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
        
        # Process audio files
        for audio_file in audio_dir.glob('*.wav'):
            filename = audio_file.stem
            
            if filename in transcripts:
                try:
                    # Process audio
                    audio = self.audio_processor.load_audio(str(audio_file))
                    audio = self.audio_processor.normalize_audio(audio)
                    mel_spec = self.audio_processor.audio_to_mel(audio)
                    
                    # Process text
                    transcript = transcripts[filename]
                    text_sequence = self.text_processor.text_to_sequence(transcript)
                    
                    speaker_data['audio_files'].append(str(audio_file))
                    speaker_data['transcripts'].append(transcript)
                    speaker_data['mel_spectrograms'].append(mel_spec)
                    speaker_data['text_sequences'].append(text_sequence)
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
        
        return speaker_data
    
    def save_processed_data(self, speaker_data: Dict, output_dir: str):
        """Save processed data to disk."""
        output_path = Path(output_dir) / speaker_data['speaker_id']
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            'speaker_id': speaker_data['speaker_id'],
            'num_samples': len(speaker_data['audio_files']),
            'audio_files': speaker_data['audio_files'],
            'transcripts': speaker_data['transcripts']
        }
        
        with open(output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save mel spectrograms and text sequences
        torch.save({
            'mel_spectrograms': speaker_data['mel_spectrograms'],
            'text_sequences': speaker_data['text_sequences']
        }, output_path / 'features.pt')
