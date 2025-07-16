import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import numpy as np
import librosa
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from .data_processor import CelebrityVoiceProcessor, SingingVoiceProcessor

class VoiceEvaluator:
    """Evaluation mechanisms for voice similarity and quality assessment."""
    
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 256
        self.n_fft = 1024
        
    def extract_voice_embeddings(self, audio: torch.Tensor) -> np.ndarray:
        """Extract voice embeddings for similarity comparison."""
        # Convert to numpy for librosa processing
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
        
        # Extract MFCC features as voice embeddings
        mfccs = librosa.feature.mfcc(
            y=audio_np, 
            sr=self.sample_rate, 
            n_mfcc=13,
            hop_length=self.hop_length,
            n_fft=self.n_fft
        )
        
        # Extract spectral features
        spectral_centroids = librosa.feature.spectral_centroid(
            y=audio_np, sr=self.sample_rate, hop_length=self.hop_length
        )
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=audio_np, sr=self.sample_rate, hop_length=self.hop_length
        )
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=audio_np, sr=self.sample_rate, hop_length=self.hop_length
        )
        zero_crossing_rate = librosa.feature.zero_crossing_rate(
            y=audio_np, hop_length=self.hop_length
        )
        
        # Combine features
        features = np.vstack([
            mfccs,
            spectral_centroids,
            spectral_bandwidth,
            spectral_rolloff,
            zero_crossing_rate
        ])
        
        # Return mean and std as embedding
        embedding = np.hstack([
            np.mean(features, axis=1),
            np.std(features, axis=1)
        ])
        
        return embedding
    
    def calculate_voice_similarity(self, original_audio: torch.Tensor, 
                                 converted_audio: torch.Tensor) -> Dict[str, float]:
        """Calculate similarity between original and converted voice."""
        # Extract embeddings
        original_embedding = self.extract_voice_embeddings(original_audio)
        converted_embedding = self.extract_voice_embeddings(converted_audio)
        
        # Calculate cosine similarity
        cosine_sim = 1 - cosine(original_embedding, converted_embedding)
        
        # Calculate Euclidean distance (normalized)
        euclidean_dist = np.linalg.norm(original_embedding - converted_embedding)
        euclidean_sim = 1 / (1 + euclidean_dist)
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(original_embedding, converted_embedding)[0, 1]
        
        # F0 similarity
        f0_original = self._extract_f0(original_audio)
        f0_converted = self._extract_f0(converted_audio)
        f0_similarity = self._calculate_f0_similarity(f0_original, f0_converted)
        
        # Spectral similarity
        spectral_sim = self._calculate_spectral_similarity(original_audio, converted_audio)
        
        return {
            'cosine_similarity': float(cosine_sim),
            'euclidean_similarity': float(euclidean_sim),
            'correlation': float(correlation) if not np.isnan(correlation) else 0.0,
            'f0_similarity': float(f0_similarity),
            'spectral_similarity': float(spectral_sim),
            'overall_similarity': float(np.mean([cosine_sim, euclidean_sim, f0_similarity, spectral_sim]))
        }
    
    def assess_audio_quality(self, audio: torch.Tensor) -> Dict[str, float]:
        """Assess the quality of generated audio."""
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
        
        # Signal-to-noise ratio estimation
        snr = self._estimate_snr(audio_np)
        
        # Dynamic range
        dynamic_range = np.max(audio_np) - np.min(audio_np)
        
        # Zero crossing rate (measure of noisiness)
        zcr = librosa.feature.zero_crossing_rate(audio_np)[0]
        avg_zcr = np.mean(zcr)
        
        # Spectral quality metrics
        spectral_quality = self._assess_spectral_quality(audio_np)
        
        # Harmonic-to-noise ratio
        harmonic_ratio = self._calculate_harmonic_ratio(audio_np)
        
        # Overall quality score (0-100)
        quality_score = self._calculate_quality_score(
            snr, dynamic_range, avg_zcr, spectral_quality, harmonic_ratio
        )
        
        return {
            'snr_db': float(snr),
            'dynamic_range': float(dynamic_range),
            'zero_crossing_rate': float(avg_zcr),
            'spectral_quality': float(spectral_quality),
            'harmonic_ratio': float(harmonic_ratio),
            'overall_quality_score': float(quality_score)
        }
    
    def _extract_f0(self, audio: torch.Tensor) -> np.ndarray:
        """Extract fundamental frequency."""
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio
        f0 = librosa.yin(audio_np, fmin=80, fmax=400, sr=self.sample_rate)
        return f0
    
    def _calculate_f0_similarity(self, f0_1: np.ndarray, f0_2: np.ndarray) -> float:
        """Calculate F0 similarity between two audio signals."""
        # Remove unvoiced frames (f0 = 0)
        voiced_1 = f0_1[f0_1 > 0]
        voiced_2 = f0_2[f0_2 > 0]
        
        if len(voiced_1) == 0 or len(voiced_2) == 0:
            return 0.0
        
        # Calculate statistics
        mean_1, std_1 = np.mean(voiced_1), np.std(voiced_1)
        mean_2, std_2 = np.mean(voiced_2), np.std(voiced_2)
        
        # Mean similarity
        mean_diff = abs(mean_1 - mean_2) / max(mean_1, mean_2)
        mean_sim = 1 - min(mean_diff, 1.0)
        
        # Variance similarity
        var_diff = abs(std_1 - std_2) / max(std_1, std_2)
        var_sim = 1 - min(var_diff, 1.0)
        
        return (mean_sim + var_sim) / 2
    
    def _calculate_spectral_similarity(self, audio1: torch.Tensor, audio2: torch.Tensor) -> float:
        """Calculate spectral similarity between two audio signals."""
        audio1_np = audio1.numpy() if isinstance(audio1, torch.Tensor) else audio1
        audio2_np = audio2.numpy() if isinstance(audio2, torch.Tensor) else audio2
        
        # Extract mel spectrograms
        mel1 = librosa.feature.melspectrogram(y=audio1_np, sr=self.sample_rate)
        mel2 = librosa.feature.melspectrogram(y=audio2_np, sr=self.sample_rate)
        
        # Ensure same dimensions
        min_frames = min(mel1.shape[1], mel2.shape[1])
        mel1 = mel1[:, :min_frames]
        mel2 = mel2[:, :min_frames]
        
        # Calculate cosine similarity
        mel1_flat = mel1.flatten()
        mel2_flat = mel2.flatten()
        
        similarity = cosine_similarity([mel1_flat], [mel2_flat])[0][0]
        return similarity
    
    def _estimate_snr(self, audio: np.ndarray) -> float:
        """Estimate signal-to-noise ratio."""
        # Simple SNR estimation using signal variance
        signal_power = np.var(audio)
        noise_power = np.var(audio - np.mean(audio))  # Simplified noise estimation
        
        if noise_power == 0:
            return 100.0  # Very high SNR
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        return snr_db
    
    def _assess_spectral_quality(self, audio: np.ndarray) -> float:
        """Assess spectral quality of audio."""
        # Calculate spectral centroid variability
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
        centroid_var = np.var(spectral_centroids)
        
        # Good quality audio has moderate spectral centroid variance
        # Normalize to 0-1 scale
        quality = 1 / (1 + centroid_var / 1000)
        return quality
    
    def _calculate_harmonic_ratio(self, audio: np.ndarray) -> float:
        """Calculate harmonic-to-noise ratio."""
        # Extract harmonic and percussive components
        harmonic, percussive = librosa.effects.hpss(audio)
        
        harmonic_power = np.sum(harmonic ** 2)
        percussive_power = np.sum(percussive ** 2)
        
        if percussive_power == 0:
            return 100.0
        
        hnr = harmonic_power / percussive_power
        return hnr
    
    def _calculate_quality_score(self, snr: float, dynamic_range: float, 
                               zcr: float, spectral_quality: float, 
                               harmonic_ratio: float) -> float:
        """Calculate overall quality score (0-100)."""
        # Normalize individual metrics to 0-1 scale
        snr_norm = min(max(snr / 30, 0), 1)  # SNR up to 30dB
        dr_norm = min(max(dynamic_range / 2, 0), 1)  # Dynamic range up to 2
        zcr_norm = 1 - min(zcr / 0.1, 1)  # Lower ZCR is better
        spec_norm = spectral_quality
        hnr_norm = min(harmonic_ratio / 10, 1)  # HNR up to 10
        
        # Weighted average
        weights = [0.25, 0.15, 0.15, 0.25, 0.20]
        quality_score = np.average(
            [snr_norm, dr_norm, zcr_norm, spec_norm, hnr_norm], 
            weights=weights
        ) * 100
        
        return quality_score
    
    def generate_evaluation_report(self, original_audio: torch.Tensor, 
                                 converted_audio: torch.Tensor, 
                                 target_voice: str) -> Dict:
        """Generate comprehensive evaluation report."""
        similarity_metrics = self.calculate_voice_similarity(original_audio, converted_audio)
        quality_metrics = self.assess_audio_quality(converted_audio)
        
        # Performance grading
        similarity_grade = self._grade_similarity(similarity_metrics['overall_similarity'])
        quality_grade = self._grade_quality(quality_metrics['overall_quality_score'])
        
        report = {
            'target_voice': target_voice,
            'evaluation_timestamp': str(torch.datetime.now()),
            'similarity_metrics': similarity_metrics,
            'quality_metrics': quality_metrics,
            'performance_grades': {
                'similarity_grade': similarity_grade,
                'quality_grade': quality_grade,
                'overall_grade': self._calculate_overall_grade(similarity_grade, quality_grade)
            },
            'recommendations': self._generate_recommendations(similarity_metrics, quality_metrics)
        }
        
        return report
    
    def _grade_similarity(self, similarity_score: float) -> str:
        """Grade similarity performance."""
        if similarity_score >= 0.9:
            return 'A+ (Excellent)'
        elif similarity_score >= 0.8:
            return 'A (Very Good)'
        elif similarity_score >= 0.7:
            return 'B (Good)'
        elif similarity_score >= 0.6:
            return 'C (Fair)'
        elif similarity_score >= 0.5:
            return 'D (Poor)'
        else:
            return 'F (Very Poor)'
    
    def _grade_quality(self, quality_score: float) -> str:
        """Grade audio quality."""
        if quality_score >= 90:
            return 'A+ (Excellent)'
        elif quality_score >= 80:
            return 'A (Very Good)'
        elif quality_score >= 70:
            return 'B (Good)'
        elif quality_score >= 60:
            return 'C (Fair)'
        elif quality_score >= 50:
            return 'D (Poor)'
        else:
            return 'F (Very Poor)'
    
    def _calculate_overall_grade(self, sim_grade: str, qual_grade: str) -> str:
        """Calculate overall performance grade."""
        grade_values = {
            'A+': 4.0, 'A': 3.7, 'B': 3.0, 'C': 2.0, 'D': 1.0, 'F': 0.0
        }
        
        sim_val = grade_values.get(sim_grade.split()[0], 0.0)
        qual_val = grade_values.get(qual_grade.split()[0], 0.0)
        
        avg_val = (sim_val + qual_val) / 2
        
        if avg_val >= 3.8:
            return 'A+ (Excellent)'
        elif avg_val >= 3.5:
            return 'A (Very Good)'
        elif avg_val >= 2.5:
            return 'B (Good)'
        elif avg_val >= 1.5:
            return 'C (Fair)'
        elif avg_val >= 0.5:
            return 'D (Poor)'
        else:
            return 'F (Very Poor)'
    
    def _generate_recommendations(self, similarity_metrics: Dict, 
                                quality_metrics: Dict) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        if similarity_metrics['f0_similarity'] < 0.7:
            recommendations.append(
                "Consider improving pitch matching - the fundamental frequency doesn't match well"
            )
        
        if similarity_metrics['spectral_similarity'] < 0.7:
            recommendations.append(
                "Spectral characteristics need improvement - consider better formant matching"
            )
        
        if quality_metrics['snr_db'] < 15:
            recommendations.append(
                "Audio quality could be improved - reduce background noise"
            )
        
        if quality_metrics['zero_crossing_rate'] > 0.05:
            recommendations.append(
                "Audio contains too much noise - apply noise reduction"
            )
        
        if quality_metrics['harmonic_ratio'] < 2.0:
            recommendations.append(
                "Harmonic content is low - improve voice clarity"
            )
        
        if not recommendations:
            recommendations.append("Excellent conversion quality! No specific improvements needed.")
        
        return recommendations

class CelebrityVoiceManager:
    """Manager for celebrity voice models with strict ethical guidelines."""
    
    def __init__(self, models_dir: str):
        self.models_dir = Path(models_dir)
        self.celebrity_processor = CelebrityVoiceProcessor()
        self.singing_processor = SingingVoiceProcessor()
        self.evaluator = VoiceEvaluator()
        
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
                                  celebrity_id: str) -> Tuple[torch.Tensor, Dict]:
        """Convert input audio to celebrity-like voice with evaluation."""
        self.show_disclaimer()
        
        # Get celebrity characteristics
        characteristics = self.celebrity_processor.get_voice_characteristics(celebrity_id)
        
        # Simple voice conversion based on characteristics
        converted_audio = self._apply_voice_characteristics(input_audio, characteristics)
        
        # Generate evaluation report
        evaluation_report = self.evaluator.generate_evaluation_report(
            input_audio, converted_audio, celebrity_id
        )
        
        # Save evaluation report
        self._save_evaluation_report(evaluation_report, celebrity_id)
        
        return converted_audio, evaluation_report
    
    def convert_to_singing_style(self, input_audio: torch.Tensor, 
                                singer_style: str) -> Tuple[torch.Tensor, Dict]:
        """Convert speech to singing with evaluation."""
        # Get singing style characteristics
        style_info = self.singing_processor.singer_styles.get(singer_style, {})
        
        # Apply singing conversion
        singing_audio = self._apply_singing_conversion(input_audio, style_info)
        
        # Generate evaluation report
        evaluation_report = self.evaluator.generate_evaluation_report(
            input_audio, singing_audio, singer_style
        )
        
        # Save evaluation report
        self._save_evaluation_report(evaluation_report, singer_style)
        
        return singing_audio, evaluation_report
    
    def _save_evaluation_report(self, report: Dict, voice_id: str):
        """Save evaluation report to file."""
        timestamp = str(torch.datetime.now()).replace(':', '-').replace(' ', '_')
        report_file = self.evaluation_dir / f"evaluation_{voice_id}_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def generate_documentation(self) -> str:
        """Generate comprehensive documentation for the voice cloning system."""
        doc_content = self._create_system_documentation()
        
        doc_file = self.docs_dir / 'voice_cloning_documentation.md'
        with open(doc_file, 'w') as f:
            f.write(doc_content)
        
        return str(doc_file)
    
    def _create_system_documentation(self) -> str:
        """Create comprehensive system documentation."""
        return """
# Voice Cloning System Documentation

## Overview
This voice cloning system provides educational demonstration of voice conversion technology with strict ethical guidelines.

## Features

### 1. Celebrity Voice Conversion
- **Purpose**: Educational demonstration only
- **Ethical Guidelines**: Strict consent and usage policies
- **Quality Metrics**: Comprehensive similarity and quality assessment

### 2. Singing Voice Conversion
- **Styles Available**: Classical, Jazz, Rock, Pop
- **Analysis**: Vibrato detection, vocal technique classification
- **Evaluation**: Pitch accuracy, harmonic content assessment

### 3. Evaluation Metrics

#### Similarity Metrics:
- **Cosine Similarity**: Measures overall voice characteristic similarity
- **F0 Similarity**: Compares fundamental frequency patterns
- **Spectral Similarity**: Analyzes frequency content matching
- **Correlation**: Statistical correlation between voice features

#### Quality Metrics:
- **SNR (Signal-to-Noise Ratio)**: Audio clarity measurement
- **Dynamic Range**: Audio level variation assessment
- **Zero Crossing Rate**: Noise level indicator
- **Harmonic Ratio**: Voice clarity vs noise content

### 4. Performance Grading
- **A+ (90-100%)**: Excellent conversion quality
- **A (80-89%)**: Very good quality with minor issues
- **B (70-79%)**: Good quality with some noticeable differences
- **C (60-69%)**: Fair quality with significant differences
- **D (50-59%)**: Poor quality with major issues
- **F (<50%)**: Very poor quality, needs significant improvement

## Ethical Guidelines

### IMPORTANT NOTICE:
1. **Educational Use Only**: This technology is for learning and demonstration
2. **No Impersonation**: Do not use for fraudulent purposes
3. **Consent Required**: Always obtain permission before using someone's voice
4. **Respect Rights**: Honor personality and intellectual property rights
5. **Consider Impact**: Think about potential misuse and societal harm

### Prohibited Uses:
- Fraud or deception
- Non-consensual impersonation
- Creating misleading content
- Harassment or abuse
- Commercial use without proper licensing

### Permitted Uses:
- Educational research
- Academic demonstrations
- Accessibility tools (with consent)
- Artistic creation (with proper attribution)
- Voice restoration for medical purposes

## Technical Implementation

### Voice Conversion Pipeline:
1. **Audio Preprocessing**: Normalization, noise reduction
2. **Feature Extraction**: MFCC, spectral features, F0 analysis
3. **Voice Characteristic Analysis**: Pitch, timbre, speaking style
4. **Conversion Application**: Pitch shifting, formant modification
5. **Quality Assessment**: Similarity and quality evaluation
6. **Report Generation**: Comprehensive performance analysis

### Evaluation Framework:
```python
# Example evaluation usage
evaluator = VoiceEvaluator()
similarity = evaluator.calculate_voice_similarity(original, converted)
quality = evaluator.assess_audio_quality(converted)
report = evaluator.generate_evaluation_report(original, converted, target_voice)
```

## Best Practices

### For Better Results:
1. **High-Quality Input**: Use clear, noise-free audio
2. **Appropriate Length**: 5-30 seconds for optimal conversion
3. **Clear Speech**: Avoid mumbling or background noise
4. **Consistent Volume**: Maintain steady audio levels

### Ethical Usage:
1. **Always Disclose**: Make it clear when audio is synthetic
2. **Get Consent**: Obtain permission before using someone's voice
3. **Consider Context**: Think about how the content might be perceived
4. **Report Misuse**: Flag unethical applications when encountered

## Troubleshooting

### Common Issues:
- **Low Similarity Score**: Try higher quality input audio
- **Poor Audio Quality**: Check for background noise or distortion
- **Pitch Problems**: Ensure input audio has clear vocal content
- **Spectral Mismatch**: Consider voice compatibility with target style

### Support:
For technical issues or ethical concerns, contact the development team.

## Version History
- v1.0: Initial release with basic voice conversion
- v1.1: Added evaluation metrics and quality assessment
- v1.2: Enhanced ethical guidelines and documentation

## Disclaimer
This technology is provided for educational purposes only. Users are responsible for ethical use and compliance with applicable laws and regulations.
"""
