"""Audio agent for processing audio files with feature extraction and analysis."""

import os
import warnings
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch

from .base_agent import BaseAgent, FileAnalysis

# Suppress librosa warnings
warnings.filterwarnings('ignore')

class AudioAgent(BaseAgent):
    """Audio agent for processing and analyzing audio files."""

    def __init__(self) -> None:
        self._supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
        self.agent_name = "audio_agent"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models for audio analysis."""
        try:
            # For feature extraction and embeddings
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Warning: Could not load audio models: {str(e)}")
            self.processor = None
            self.model = None

    @property
    def supported_formats(self) -> List[str]:
        return self._supported_formats
    
    def _load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and convert to mono 16kHz WAV format."""
        try:
            # Convert to WAV if needed
            if not file_path.lower().endswith('.wav'):
                audio = AudioSegment.from_file(file_path)
                audio = audio.set_channels(1)  # Convert to mono
                audio = audio.set_frame_rate(16000)  # Resample to 16kHz
                temp_path = os.path.splitext(file_path)[0] + '_temp.wav'
                audio.export(temp_path, format='wav')
                y, sr = librosa.load(temp_path, sr=16000, mono=True)
                os.remove(temp_path)
            else:
                y, sr = librosa.load(file_path, sr=16000, mono=True)
            
            return y, sr
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")
    
    def _extract_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract audio features using librosa."""
        features = {}
        
        # Basic features
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        
        # Spectral features
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        
        features.update({
            'spectral_centroid': float(spectral_centroid),
            'spectral_bandwidth': float(spectral_bandwidth),
            'spectral_rolloff': float(spectral_rolloff),
        })
        
        # Zero crossing rate
        features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        
        # RMS energy
        features['rms_energy'] = float(np.mean(librosa.feature.rms(y=y)))
        
        # MFCCs (mean of first 13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i, coeff in enumerate(mfccs):
            features[f'mfcc_{i+1}_mean'] = float(np.mean(coeff))
        
        return features
    
    def _generate_embedding(self, y: np.ndarray) -> List[float]:
        """Generate audio embedding using Wav2Vec2."""
        if self.processor is None or self.model is None:
            return []
            
        try:
            # Process audio
            input_values = self.processor(
                y, 
                sampling_rate=16000, 
                return_tensors="pt",
                padding=True
            ).input_values.to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(input_values)
                # Use mean of the last hidden state as the audio embedding
                embedding = torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()
                return embedding.tolist()
        except Exception as e:
            print(f"Error generating audio embedding: {str(e)}")
            return []
    
    def _generate_summary(self, features: Dict[str, float]) -> str:
        """Generate a human-readable summary of audio features."""
        duration = features.get('duration', 0)
        energy = features.get('rms_energy', 0)
        spectral_centroid = features.get('spectral_centroid', 0)
        
        # Basic summary based on features
        summary = ["Audio file detected"]
        
        if duration > 0:
            summary.append(f"Duration: {duration:.1f} seconds")
        
        if energy > 0.1:  # Arbitrary threshold
            summary.append("High energy content")
        
        if spectral_centroid > 2000:  # Higher values indicate more high-frequency content
            summary.append("Contains high-frequency components")
        
        return ". ".join(summary) + "."
    
    def _generate_tags(self, features: Dict[str, float]) -> List[str]:
        """Generate descriptive tags based on audio features."""
        tags = ["audio"]
        
        # Add tags based on features
        if features.get('duration', 0) > 300:  # Longer than 5 minutes
            tags.append("long")
        
        if features.get('rms_energy', 0) > 0.1:
            tags.append("loud")
        
        if features.get('spectral_centroid', 0) > 2000:
            tags.extend(["high_freq", "bright"])
        else:
            tags.append("warm")
        
        return tags

    async def analyze(self, file_path: str, metadata: Optional[Dict] = None) -> FileAnalysis:
        """Analyze an audio file and return its features and embedding."""
        metadata = metadata or {}
        
        try:
            # Load and preprocess audio
            y, sr = self._load_audio(file_path)
            
            # Extract features
            features = self._extract_features(y, sr)
            
            # Generate embedding
            embedding = self._generate_embedding(y)
            
            # Generate summary and tags
            summary = self._generate_summary(features)
            tags = self._generate_tags(features)
            
            # Update metadata with extracted features
            metadata.update({
                'audio_features': features,
                'sample_rate': sr,
                'samples': len(y),
                'agent': self.agent_name,
                'agent_version': '1.0.0',
            })
            
            return FileAnalysis(
                summary=summary,
                tags=tags,
                embedding=embedding,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error analyzing audio file: {str(e)}")
            return FileAnalysis(
                summary="Error analyzing audio file",
                tags=["error", "audio"],
                embedding=[],
                metadata={
                    'error': str(e),
                    'agent': self.agent_name,
                    'agent_version': '1.0.0',
                }
            )
