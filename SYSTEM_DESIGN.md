# CryBaby - System Design Document

## Overview
CryBaby is a comprehensive baby monitoring application that uses machine learning to analyze baby cries and provides AI-powered insights to parents. The system consists of a mobile iOS app, a FastAPI backend service, and ML models for cry classification.

## Architecture

### Frontend (iOS App)
- **Technology**: Native iOS (SwiftUI)
- **Platform**: iOS 15.0+
- **Features**:
  - Real-time cry recording and analysis
  - Daily activity tracking (feeding, diaper changes, sleep)
  - AI-powered parenting advice
  - Model retraining feedback system

### Backend (API Service)
- **Technology**: FastAPI (Python)
- **Framework**: Python 3.11+ with FastAPI
- **ML Integration**: scikit-learn, TensorFlow, YAMNet
- **Features**:
  - RESTful API for cry analysis
  - Real-time audio processing
  - Model serving and inference
  - Feedback collection for model improvement

### Data Storage
- **Local Storage**: iOS Core Data for app data
- **Model Storage**: Local file system for ML models
- **Audio Processing**: Temporary file storage for analysis

## Core Components

### 1. Cry Analysis Engine
- **Model**: Logistic Regression with YAMNet features
- **Input**: 16kHz mono audio (WAV format)
- **Output**: Classification probabilities for "hungry", "uncomfortable", "unknown"
- **Confidence Threshold**: Configurable (default 0.6)

### 2. Daily Activity Tracker
- **Feeding Records**: Type, time, duration, notes
- **Diaper Changes**: Type and timing
- **Sleep Sessions**: Duration and type classification
- **Cry Episodes**: Linked to analysis results

### 3. AI Advice System
- **Pattern Analysis**: Based on historical data
- **Personalized Recommendations**: Tailored to individual baby patterns
- **Expert Knowledge Integration**: External parenting resources

### 4. Model Retraining Pipeline
- **Feedback Collection**: User corrections and labels
- **Data Management**: Secure handling of user audio data
- **Incremental Learning**: Support for model updates

## API Endpoints

### Cry Analysis
- `POST /api/v1/cry/predict` - Analyze single audio file
- `POST /api/v1/cry/predict-batch` - Batch processing
- `POST /api/v1/cry/feedback` - Submit feedback for training

### System
- `GET /health` - Health check
- `GET /api/v1/model/info` - Model information
- `GET /api/v1/model/test` - Test endpoint

## Security & Privacy
- **Audio Data**: Processed locally when possible
- **Personal Data**: Stored securely on device
- **Feedback**: Anonymized before training use
- **Permissions**: Clear microphone usage disclosure

## Deployment Strategy

### Development
- **Backend**: Docker Compose with local development
- **Frontend**: Xcode simulator and device testing
- **Model**: Local file serving

### Production
- **Backend**: Containerized deployment (Docker/Kubernetes)
- **Frontend**: App Store distribution
- **Monitoring**: Health checks and logging

## Technology Decisions

### Why Native iOS?
- **Audio Quality**: Superior control over microphone access
- **Performance**: Real-time processing requirements
- **Integration**: Seamless iOS ecosystem integration
- **Reliability**: Consistent behavior across devices

### Why FastAPI?
- **Performance**: High-speed async processing
- **Type Safety**: Python type hints and validation
- **Documentation**: Auto-generated API docs
- **ML Integration**: Excellent Python ML ecosystem support

## Data Flow

1. **Recording**: User records audio in iOS app
2. **Upload**: Audio sent to FastAPI backend
3. **Processing**: YAMNet feature extraction + ML prediction
4. **Response**: Results with confidence scores and suggestions
5. **Storage**: Local storage of analysis results
6. **Feedback**: Optional user correction for model improvement

## Future Enhancements
- **Offline Mode**: Local ML processing capability
- **Multiple Children**: Support for family profiles
- **Health Integration**: Apple HealthKit integration
- **Notifications**: Smart reminders and alerts
- **Advanced Analytics**: Trend analysis and predictions
