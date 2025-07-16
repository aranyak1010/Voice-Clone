import json
import torch
from pathlib import Path
from typing import Dict, List, Optional
import warnings
from .data_processor import CelebrityVoiceProcessor, SingingVoiceProcessor

class CelebrityVoiceManager:
    """Manager for celebrity voice models with strict ethical guidelines."""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.celebrity_processor = CelebrityVoiceProcessor()
        self.singing_processor = SingingVoiceProcessor()
        
        # Ethical guidelines and warnings
        self.ethical_guidelines = {
            'consent_required': True,
            'educational_use_only': True,
            'no_impersonation': True,
            'no_fraud': True,
            'attribution_required': True
        }
        
        self.disclaimer = """
        IMPORTANT ETHICAL NOTICE:
        
        1. This technology is for EDUCATIONAL and DEMONSTRATION purposes only
        2. Do NOT use for impersonation, fraud, or deceptive practices
        3. Respect personality rights and intellectual property
        4. Obtain proper consent before using anyone's voice
        5. Use responsibly and ethically
        6. Consider the potential harm of synthetic voice technology
        
        By using this feature, you agree to these ethical guidelines.
        """
    
    def show_disclaimer(self):
        """Display ethical disclaimer."""
        print(self.disclaimer)
        warnings.warn("Celebrity voice cloning requires ethical consideration", UserWarning)
    
    def get_available_celebrities(self) -> List[Dict]:
        """Get list of available celebrity voice profiles."""
        self.show_disclaimer()
        
        celebrities = self.celebrity_processor.get_celebrity_list()
        
        # Add ethical markers to each celebrity
        for celebrity in celebrities:
            celebrity['ethical_use_only'] = True
            celebrity['consent_status'] = 'demonstration_only'
        
        return celebrities
    
    def get_available_singers(self) -> List[Dict]:
        """Get list of available singing voice styles."""
        return [
            {
                'id': style_id,
                'name': style_info['name'],
                'type': 'singing',
                'vocal_technique': style_info['vocal_technique'],
                'ethical_use_only': True
            }
            for style_id, style_info in self.singing_processor.singer_styles.items()
        ]
    
    def load_celebrity_model(self, celebrity_id: str) -> Optional[Dict]:
        """Load celebrity voice model with ethical checks."""
        self.show_disclaimer()
        
        model_path = self.models_dir / f"celebrity_{celebrity_id}.pt"
        
        if not model_path.exists():
            # Return characteristics for voice conversion instead of trained model
            return self.celebrity_processor.get_voice_characteristics(celebrity_id)
        
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Add ethical metadata
            checkpoint['ethical_guidelines'] = self.ethical_guidelines
            checkpoint['disclaimer_shown'] = True
            checkpoint['use_case'] = 'educational_demonstration'
            
            return checkpoint
        except Exception as e:
            print(f"Error loading celebrity model: {e}")
            return None
    
    def convert_to_celebrity_voice(self, input_audio: torch.Tensor, 
                                  celebrity_id: str) -> torch.Tensor:
        """Convert input audio to celebrity-like voice using voice conversion."""
        self.show_disclaimer()
        
        # Get celebrity characteristics
        characteristics = self.celebrity_processor.get_voice_characteristics(celebrity_id)
        
        # Simple voice conversion based on characteristics
        converted_audio = self._apply_voice_characteristics(input_audio, characteristics)
        
        return converted_audio
    
    def convert_to_singing_style(self, input_audio: torch.Tensor, 
                                singer_style: str) -> torch.Tensor:
        """Convert speech to singing in specified style."""
        # Get singing style characteristics
        style_info = self.singing_processor.singer_styles.get(singer_style, {})
        
        # Apply singing conversion
        singing_audio = self._apply_singing_conversion(input_audio, style_info)
        
        return singing_audio
    
    def _apply_voice_characteristics(self, audio: torch.Tensor, 
                                   characteristics: Dict) -> torch.Tensor:
        """Apply voice characteristics for celebrity voice conversion."""
        # Simplified voice conversion - in production, use advanced techniques
        converted = audio.clone()
        
        # Adjust pitch based on F0 characteristics
        if 'f0_mean' in characteristics:
            target_f0 = characteristics['f0_mean']
            # Apply pitch shifting (simplified)
            pitch_shift_factor = target_f0 / 150.0  # Assume 150Hz as baseline
            converted = self._pitch_shift(converted, pitch_shift_factor)
        
        # Apply formant shifting for voice timbre
        if 'voice_type' in characteristics:
            voice_type = characteristics['voice_type']
            converted = self._apply_voice_type(converted, voice_type)
        
        return converted
    
    def _apply_singing_conversion(self, audio: torch.Tensor, 
                                style_info: Dict) -> torch.Tensor:
        """Convert speech to singing with specified style."""
        singing_audio = audio.clone()
        
        # Add vibrato if specified
        if 'vibrato_rate' in style_info:
            vibrato_rate = style_info['vibrato_rate']
            singing_audio = self._add_vibrato(singing_audio, vibrato_rate)
        
        # Adjust pitch range for singing
        if 'f0_range' in style_info:
            f0_range = style_info['f0_range']
            singing_audio = self._adjust_pitch_range(singing_audio, f0_range)
        
        return singing_audio
    
    def _pitch_shift(self, audio: torch.Tensor, factor: float) -> torch.Tensor:
        """Apply pitch shifting to audio."""
        # Simplified pitch shifting - use librosa or advanced methods in production
        if factor == 1.0:
            return audio
        
        # Simple resampling-based pitch shift (not ideal but functional)
        stretched = torch.nn.functional.interpolate(
            audio.unsqueeze(0).unsqueeze(0),
            scale_factor=1.0/factor,
            mode='linear',
            align_corners=False
        ).squeeze()
        
        return stretched
    
    def _apply_voice_type(self, audio: torch.Tensor, voice_type: str) -> torch.Tensor:
        """Apply voice type characteristics."""
        if voice_type == 'authoritative':
            # Lower pitch, more resonance
            return self._pitch_shift(audio, 0.9)
        elif voice_type == 'energetic':
            # Higher pitch, more brightness
            return self._pitch_shift(audio, 1.1)
        elif voice_type == 'baritone':
            # Much lower pitch
            return self._pitch_shift(audio, 0.8)
        
        return audio
    
    def _add_vibrato(self, audio: torch.Tensor, vibrato_rate: float) -> torch.Tensor:
        """Add vibrato effect to audio."""
        sample_rate = 22050
        t = torch.arange(len(audio), dtype=torch.float32) / sample_rate
        
        # Create vibrato modulation
        vibrato = 0.02 * torch.sin(2 * torch.pi * vibrato_rate * t)
        
        # Apply vibrato as amplitude modulation (simplified)
        vibrato_audio = audio * (1 + vibrato)
        
        return vibrato_audio
    
    def _adjust_pitch_range(self, audio: torch.Tensor, f0_range: List[float]) -> torch.Tensor:
        """Adjust pitch range for singing."""
        # Simplified pitch range adjustment
        min_f0, max_f0 = f0_range
        
        # Apply dynamic pitch modification based on range
        if max_f0 > 400:  # High range singing
            return self._pitch_shift(audio, 1.2)
        elif max_f0 < 200:  # Low range singing
            return self._pitch_shift(audio, 0.8)
        
        return audio
    
    def create_voice_profile(self, speaker_name: str, audio_samples: List[str], 
                           voice_type: str = 'speech') -> Dict:
        """Create a voice profile from audio samples."""
        self.show_disclaimer()
        
        if not audio_samples:
            raise ValueError("No audio samples provided")
        
        profile = {
            'speaker_name': speaker_name,
            'voice_type': voice_type,
            'samples_count': len(audio_samples),
            'ethical_consent': False,  # Must be explicitly set to True
            'created_for': 'educational_demonstration',
            'samples': audio_samples
        }
        
        # Save profile with ethical metadata
        profile_path = self.models_dir / f"profile_{speaker_name}.json"
        with open(profile_path, 'w') as f:
            json.dump(profile, f, indent=2)
        
        return profile
    
    def validate_ethical_use(self, use_case: str) -> bool:
        """Validate if the use case is ethical."""
        ethical_uses = [
            'educational_demonstration',
            'research',
            'artistic_creation',
            'accessibility_tools',
            'voice_restoration'
        ]
        
        unethical_uses = [
            'impersonation',
            'fraud',
            'deception',
            'harassment',
            'non_consensual_use'
        ]
        
        if use_case in unethical_uses:
            warnings.warn(f"Use case '{use_case}' is not ethical", UserWarning)
            return False
        
        return use_case in ethical_uses
