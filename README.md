# CryBaby - AI-Powered Baby Cry Analysis

A comprehensive iOS application that uses machine learning to analyze baby cries and provide intelligent insights to help parents understand their baby's needs.

## 🎯 Project Overview

CryBaby combines cutting-edge machine learning with intuitive mobile design to help parents:
- **Analyze baby cries** using AI to identify if baby is hungry, uncomfortable, or needs other attention
- **Track daily activities** including feeding, diaper changes, and sleep patterns
- **Receive AI-powered advice** based on patterns and expert knowledge
- **Improve the system** through feedback and model retraining

## 🏗️ System Architecture

### Components
- **iOS App**: Native SwiftUI application for iPhone/iPad
- **FastAPI Backend**: Python-based API service for ML processing
- **ML Model**: Logistic Regression classifier using YAMNet embeddings

### Key Features
- Real-time cry recording and analysis
- Confidence-based predictions with "unknown" classification
- Daily activity tracking and pattern analysis
- Personalized AI recommendations
- User feedback system for continuous improvement

## 📱 iOS App Features

### 1. Cry Analysis Tab
- One-tap recording with large, accessible button
- Real-time audio processing and analysis
- Results with confidence scores and actionable suggestions
- Feedback collection for model improvement

### 2. Daily Activities Tab
- Track feeding times, types, and amounts
- Record diaper changes with types
- Monitor sleep sessions and patterns
- Visual summaries and historical data

### 3. AI Advice Tab
- Personalized recommendations based on collected data
- Pattern identification and insights
- Time-based analysis (daily, weekly, monthly views)
- Expert knowledge integration

### 4. Settings Tab
- Backend configuration and connectivity
- Confidence threshold adjustment
- Data export and privacy controls
- App information and version details

## 🔧 Technical Implementation

### Machine Learning Pipeline
1. **Audio Capture**: 16kHz mono WAV format recording
2. **Feature Extraction**: YAMNet embedding generation
3. **Classification**: Logistic Regression with confidence scoring
4. **Threshold Application**: "Unknown" classification for low-confidence predictions

### Backend API
```python
# Main endpoints
POST /api/v1/cry/predict          # Analyze single audio file
POST /api/v1/cry/predict-batch    # Batch processing
POST /api/v1/cry/feedback         # Submit training feedback
GET  /health                      # Service health check
```

### Data Models
- **CryAnalysisResult**: Classification results with confidence and suggestions
- **BabyActivity**: Daily tracking data structure
- **FeedbackRequest**: User correction and improvement data

## 🚀 Quick Start

### Prerequisites
- iOS 15.0+ and Xcode 15.0+
- Python 3.11+ and Docker
- Mac with Apple Silicon or Intel processor

### Backend Setup
```bash
cd backend/services/cry_analysis
docker compose up -d
```

### iOS Development
1. Open `ios/CryBaby/CryBaby.xcodeproj` in Xcode
2. Select iPhone simulator
3. Build and run (⌘+R)

### Testing
Backend health check: `curl http://localhost:8000/health`

## 📊 Model Performance

The ML model achieves:
- **Cross-validation F1 Score**: ~80% (macro average)
- **Classes**: "hungry", "uncomfortable", "unknown"
- **Confidence Threshold**: 0.6 (configurable)
- **Audio Requirements**: WAV format, 16kHz, mono, max 30 seconds

## 🔒 Privacy & Security

- **Local Processing**: Audio processing prioritizes local storage
- **Minimal Data Collection**: Only necessary information stored
- **User Control**: Full data export and deletion capabilities
- **Transparent AI**: Clear explanations of predictions and confidence levels

## 🛠️ Development

### Project Structure
```
cryBaby/
├── ios/CryBaby/                 # iOS Swift app
├── backend/services/cry_analysis/  # FastAPI backend
├── cry_classifier.py            # ML classifier module
├── artifacts_option2/           # Trained models
└── docs/                        # Documentation
```

### Key Technologies
- **iOS**: SwiftUI, AVFoundation, Core Data
- **Backend**: FastAPI, TensorFlow, scikit-learn, librosa
- **ML**: YAMNet, Logistic Regression, audio processing

## 📈 Future Enhancements

- **Offline Processing**: Local ML inference capability
- **Multiple Profiles**: Support for families with multiple children
- **Health Integration**: Apple HealthKit connectivity
- **Advanced Analytics**: Predictive insights and trend analysis
- **Voice Commands**: Hands-free operation during busy parenting

## 🤝 Contributing

This project demonstrates the integration of:
- Modern iOS development with SwiftUI
- Machine learning pipeline with production-ready deployment
- Real-time audio processing and analysis
- User-centered design for accessibility and ease of use

## 📄 License

This project is developed for educational and demonstration purposes, showcasing best practices in mobile app development, machine learning integration, and user experience design.
