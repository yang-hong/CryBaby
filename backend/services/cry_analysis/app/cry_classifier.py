#!/usr/bin/env python3
"""
CryBaby - Baby Cry Classifier

A simple, self-contained module for classifying baby cries using 
YAMNet embeddings + Logistic Regression.

Usage:
    from cry_classifier import CryClassifier
    
    classifier = CryClassifier('artifacts_option2/yamnet_lr_full.joblib')
    result = classifier.predict('data/raw/hungry/hung_yasin_lapar_11.wav')
    print(result)
"""

import joblib
import numpy as np
import soundfile as sf
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from typing import Dict, Tuple
from pathlib import Path


class CryClassifier:
    """
    Baby cry classifier using YAMNet embeddings + Logistic Regression
    
    Attributes:
        model_path: Path to the trained model (.joblib file)
        confidence_threshold: Minimum confidence for non-unknown prediction (default: 0.6)
        yamnet: Loaded YAMNet model from TensorFlow Hub
        scaler: StandardScaler for feature normalization
        clf: Trained classifier (Logistic Regression)
        labels: List of class labels
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.6):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to trained model joblib file
            confidence_threshold: Minimum confidence for prediction (0.0 - 1.0)
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If confidence_threshold is not between 0 and 1
        """
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        if not 0 <= confidence_threshold <= 1:
            raise ValueError(f"confidence_threshold must be between 0 and 1, got {confidence_threshold}")
        
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Load the trained model bundle
        print(f"Loading model from {model_path}...")
        self.bundle = joblib.load(model_path)
        self.scaler = self.bundle['scaler']
        self.clf = self.bundle['clf']
        self.labels = self.bundle['labels']
        print(f"Model loaded! Labels: {self.labels}")
        
        # Load YAMNet model from TensorFlow Hub
        print("Loading YAMNet from TensorFlow Hub...")
        self.yamnet = hub.load("https://tfhub.dev/google/yamnet/1")
        print("YAMNet loaded successfully!")
    
    def predict(self, audio_path: str, return_features: bool = False) -> Dict:
        """
        Predict baby cry type from audio file
        
        Args:
            audio_path: Path to audio file (.wav)
            return_features: If True, include extracted features in response
            
        Returns:
            Dictionary containing:
                - predicted_label: str (e.g., 'hungry', 'uncomfortable', 'unknown')
                - probabilities: dict of label -> probability
                - confidence_level: str ('high', 'medium', 'low')
                - explanation: str (human-readable explanation)
                - max_probability: float (highest probability value)
                - features: np.ndarray (optional, if return_features=True)
                
        Example:
            >>> classifier = CryClassifier('model.joblib')
            >>> result = classifier.predict('baby_cry.wav')
            >>> print(f"Prediction: {result['predicted_label']}")
            >>> print(f"Confidence: {result['confidence_level']}")
        """
        # Load and preprocess audio
        audio = self._load_audio(audio_path)
        
        # Extract YAMNet features
        features = self._extract_features(audio)
        
        # Make prediction
        result = self._predict_from_features(features)
        
        # Optionally include features
        if return_features:
            result['features'] = features
        
        return result
    
    def _load_audio(self, path: str, target_sr: int = 16000) -> np.ndarray:
        """
        Load audio file and resample to 16kHz mono
        
        Args:
            path: Path to audio file
            target_sr: Target sample rate (default: 16000 Hz)
            
        Returns:
            Audio waveform as numpy array (float32, range: -1.0 to 1.0)
            
        Raises:
            FileNotFoundError: If audio file doesn't exist
            Exception: If audio file cannot be read
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        try:
            # Read audio file
            audio, sr = sf.read(path, dtype='float32', always_2d=False)
            
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            
            # Resample if needed
            if sr != target_sr:
                audio = librosa.resample(
                    audio, 
                    orig_sr=sr, 
                    target_sr=target_sr, 
                    res_type="kaiser_best"
                )
            
            # Clip to [-1, 1] range and ensure float32
            return np.clip(audio, -1.0, 1.0).astype(np.float32)
            
        except Exception as e:
            raise Exception(f"Error loading audio from {path}: {e}")
    
    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract YAMNet embeddings from audio
        
        Args:
            audio: Audio waveform as numpy array (16kHz, mono, float32)
            
        Returns:
            1024-dimensional feature vector (averaged across time frames)
        """
        # Convert to TensorFlow tensor
        waveform = tf.convert_to_tensor(audio, dtype=tf.float32)
        
        # Get YAMNet embeddings
        # Returns: (scores, embeddings, spectrogram)
        _, embeddings, _ = self.yamnet(waveform)
        
        # Average embeddings across time frames
        # Shape: (num_frames, 1024) -> (1024,)
        features = embeddings.numpy().mean(axis=0).astype(np.float32)
        
        return features
    
    def _predict_from_features(self, features: np.ndarray) -> Dict:
        """
        Make prediction from extracted features
        
        Args:
            features: 1024-dimensional feature vector
            
        Returns:
            Dictionary with prediction results
        """
        # Reshape and scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get class probabilities
        probabilities = self.clf.predict_proba(features_scaled)[0]
        
        # Build probability dictionary
        prob_dict = {
            label: float(prob) 
            for label, prob in zip(self.labels, probabilities)
        }
        
        # Determine predicted label and confidence
        max_prob = float(max(probabilities))
        max_idx = int(probabilities.argmax())
        
        # Apply confidence threshold - be more conservative for very extreme predictions
        second_max_prob = float(sorted(probabilities)[-2]) if len(probabilities) > 1 else 0.0
        
        # Debug logging
        print(f"DEBUG: max_prob={max_prob:.6f}, second_max_prob={second_max_prob:.2e}")
        
        # More robust out-of-distribution detection for overconfident models
        is_extremely_confident = max_prob >= 0.99  # 99%+ confidence
        is_second_prob_negligible = second_max_prob < 1e-10  # Second probability essentially zero
        confidence_ratio = max_prob / (second_max_prob + 1e-100)  # Ratio of first to second
        
        is_low_confidence = max_prob < self.confidence_threshold
        
        print(f"DEBUG: is_extremely_confident={is_extremely_confident}, is_second_prob_negligible={is_second_prob_negligible}, confidence_ratio={confidence_ratio:.2e}")
        
        # SIMPLIFIED: If probability is exactly 1.0, it's definitely overconfident for real-world data
        if max_prob >= 1.0:
            print("DEBUG: Detected perfect confidence (1.0), marking as unknown")
            predicted_label = "unknown"
            confidence_level = "uncertain"
            explanation = (
                f"The model returned perfect confidence ({max_prob:.1%}), which often indicates "
                f"the audio might not match the expected baby cry patterns. Please try recording "
                f"actual baby crying sounds for better results."
            )
        # If model is extremely confident (99%+) with negligible second probability, likely out-of-distribution
        elif is_extremely_confident and is_second_prob_negligible and confidence_ratio > 1000000:
            predicted_label = "unknown"
            confidence_level = "uncertain"
            explanation = (
                f"The model returned perfect confidence ({max_prob:.1%}), which often indicates "
                f"the audio might not match the expected baby cry patterns. Please try recording "
                f"actual baby crying sounds for better results."
            )
        elif is_low_confidence:
            predicted_label = "unknown"
            confidence_level = "low"
            explanation = (
                f"The model is uncertain (max confidence: {max_prob:.1%}). "
                f"This might be a cry type outside the trained categories "
                f"(hungry, uncomfortable), such as tired, bored, or gassy."
            )
        else:
            predicted_label = self.labels[max_idx]
            
            # Determine confidence level based on probability
            if max_prob >= 0.8:
                confidence_level = "high"
                explanation = (
                    f"High confidence prediction. Based on the cry pattern, "
                    f"your baby appears to be {predicted_label}."
                )
            elif max_prob >= 0.65:
                confidence_level = "medium"
                explanation = (
                    f"Moderate confidence. Your baby is likely {predicted_label}."
                )
            else:
                confidence_level = "low"
                explanation = (
                    f"Low confidence. Your baby might be {predicted_label}, "
                    f"but other needs are also possible."
                )
        
        return {
            'predicted_label': predicted_label,
            'probabilities': prob_dict,
            'confidence_level': confidence_level,
            'explanation': explanation,
            'max_probability': max_prob
        }
    
    def predict_batch(self, audio_paths: list) -> list:
        """
        Predict multiple audio files in batch
        
        Args:
            audio_paths: List of paths to audio files
            
        Returns:
            List of prediction dictionaries
            
        Example:
            >>> files = ['cry1.wav', 'cry2.wav', 'cry3.wav']
            >>> results = classifier.predict_batch(files)
            >>> for path, result in zip(files, results):
            ...     print(f"{path}: {result['predicted_label']}")
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
            except Exception as e:
                # Include error in results
                results.append({
                    'error': str(e),
                    'audio_path': audio_path
                })
        return results
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model metadata
        """
        return {
            'model_path': self.model_path,
            'labels': self.labels,
            'confidence_threshold': self.confidence_threshold,
            'classifier_type': type(self.clf).__name__,
            'feature_dimensions': 1024,
            'yamnet_url': 'https://tfhub.dev/google/yamnet/1'
        }


def main():
    """
    Command-line interface for testing the classifier
    
    Usage:
        python cry_classifier.py data/raw/hungry/hung_yasin_lapar_11.wav
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python cry_classifier.py <audio_file.wav>")
        print("\nExample:")
        print("  python cry_classifier.py data/raw/hungry/hung_yasin_lapar_11.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    model_path = 'artifacts_option2/yamnet_lr_full.joblib'
    
    # Initialize classifier
    classifier = CryClassifier(model_path, confidence_threshold=0.6)
    
    # Make prediction
    print(f"\n{'='*60}")
    print(f"Analyzing: {audio_path}")
    print(f"{'='*60}\n")
    
    result = classifier.predict(audio_path)
    
    # Display results
    print(f"Predicted Label: {result['predicted_label'].upper()}")
    print(f"Confidence Level: {result['confidence_level'].upper()}")
    print(f"\nProbabilities:")
    for label, prob in sorted(result['probabilities'].items(), 
                             key=lambda x: x[1], 
                             reverse=True):
        bar = '█' * int(prob * 40)
        print(f"  {label:15s}: {prob:6.1%} {bar}")
    
    print(f"\nExplanation:")
    print(f"  {result['explanation']}")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    main()
