# CryBaby Implementation Guide

This guide provides step-by-step instructions for implementing the CryBaby application, from initial setup to deployment.

## 📋 Table of Contents
1. [Project Setup](#project-setup)
2. [Backend Implementation](#backend-implementation)
3. [iOS App Development](#ios-app-development)
4. [Testing & Validation](#testing--validation)
5. [Deployment](#deployment)

## 🚀 Project Setup

### Prerequisites
- **macOS** with Xcode 15.0+
- **Python 3.11+** with pip
- **Docker** and Docker Compose
- **Git** for version control

### Initial Setup
```bash
# Clone or create the project structure
mkdir cryBaby && cd cryBaby

# Create directory structure
mkdir -p {ios,backend,frontend,docs,artifacts_option2}
mkdir -p backend/services/cry_analysis/{app,tests}
mkdir -p ios/CryBaby/CryBaby/{Views,Models,Services,Assets.xcassets}
```

## 🔧 Backend Implementation

### 1. ML Classifier Setup

The core ML functionality is in `cry_classifier.py`:

```python
from cry_classifier import CryClassifier

# Initialize classifier
classifier = CryClassifier(
    model_path='artifacts_option2/yamnet_lr_full.joblib',
    confidence_threshold=0.6
)

# Make prediction
result = classifier.predict('audio_file.wav')
```

### 2. FastAPI Backend

Create `backend/services/cry_analysis/app/main.py`:

```python
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from cry_classifier import CryClassifier

app = FastAPI(title="CryBaby API", version="1.0.0")

# Add CORS for iOS app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

classifier = CryClassifier(model_path, confidence_threshold=0.6)

@app.post("/api/v1/cry/predict")
async def predict_cry(audio: UploadFile = File(...)):
    # Process uploaded audio file
    # Return prediction results
    pass
```

### 3. Docker Configuration

**Dockerfile**:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y \
    libsndfile1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app/ /app/
EXPOSE 8000
CMD ["gunicorn", "main:app", "-w", "2", "-k", "uvicorn.workers.UvicornWorker"]
```

**docker-compose.yml**:
```yaml
version: '3.8'
services:
  cry_analysis:
    build: .
    ports:
      - "8000:8000"
    environment:
      MODEL_PATH: /model/yamnet_lr_full.joblib
      CONFIDENCE_THRESHOLD: 0.6
    volumes:
      - ../../../artifacts_option2/yamnet_lr_full.joblib:/model/
      - ../../../cry_classifier.py:/app/
      - ./app:/app
```

## 📱 iOS App Development

### 1. Project Structure

```
ios/CryBaby/CryBaby/
├── CryBabyApp.swift          # Main app entry point
├── ContentView.swift         # Tab-based navigation
├── Views/                    # UI Components
│   ├── CryAnalysisView.swift
│   ├── ActivityInputView.swift
│   ├── AIAdviceView.swift
│   ├── FeedbackView.swift
│   └── SettingsView.swift
├── Models/                   # Data models
│   ├── CryAnalysisModels.swift
│   └── ActivityModels.swift
└── Services/                 # Business logic
    ├── AudioRecorder.swift
    ├── CryAnalyzer.swift
    └── ActivityManager.swift
```

### 2. Core Views Implementation

**CryAnalysisView** - Main recording interface:
```swift
struct CryAnalysisView: View {
    @StateObject private var audioRecorder = AudioRecorder()
    @StateObject private var cryAnalyzer = CryAnalyzer()
    @State private var isRecording = false
    @State private var analysisResult: CryAnalysisResult?
    
    var body: some View {
        VStack {
            // Recording button and status
            // Analysis results display
            // Feedback collection
        }
    }
}
```

**AudioRecorder** - Record audio for analysis:
```swift
class AudioRecorder: NSObject, ObservableObject, AVAudioRecorderDelegate {
    @Published var isRecording = false
    @Published var statusText = "Tap to record"
    
    private var audioRecorder: AVAudioRecorder?
    private var audioSession = AVAudioSession.sharedInstance()
    
    func startRecording() {
        // Setup audio session and start recording
    }
    
    func stopRecording(completion: @escaping (URL) -> Void) {
        // Stop recording and return file URL
    }
}
```

### 3. Backend Integration

**CryAnalyzer** - API communication:
```swift
class CryAnalyzer: ObservableObject {
    private let backendURL = "http://127.0.0.1:8000"
    
    func analyzeCry(audioURL: URL, completion: @escaping (CryAnalysisResult) -> Void) {
        // Upload audio to backend
        // Parse response and return results
    }
}
```

### 4. Data Models

**CryAnalysisResult** - ML prediction results:
```swift
struct CryAnalysisResult: Codable {
    let cryType: CryType
    let confidence: Double
    let probabilities: [String: Double]
    let timestamp: Date
    let audioDuration: TimeInterval
    
    var suggestion: String? {
        // Return contextual advice based on prediction
    }
}
```

## 🧪 Testing & Validation

### 1. Backend Testing

```python
# Test the classifier directly
def test_classifier():
    classifier = CryClassifier('model_path')
    result = classifier.predict('test_audio.wav')
    assert result['predicted_label'] in ['hungry', 'uncomfortable', 'unknown']

# Test API endpoints
def test_api():
    response = client.post("/api/v1/cry/predict", files={"audio": test_file})
    assert response.status_code == 200
    assert "predicted_label" in response.json()
```

### 2. iOS Testing

```swift
// Unit tests for models and services
func testCryAnalysisResult() {
    let result = CryAnalysisResult(
        cryType: .hungry,
        confidence: 0.85,
        probabilities: ["hungry": 0.85, "uncomfortable": 0.15],
        timestamp: Date(),
        audioDuration: 5.0
    )
    
    XCTAssertEqual(result.cryType, .hungry)
    XCTAssertGreaterThan(result.confidence, 0.8)
}
```

### 3. Integration Testing

Test the complete flow:
1. Record audio in iOS app
2. Upload to backend API
3. Verify ML prediction
4. Display results in UI
5. Submit feedback if needed

## 🚀 Deployment

### 1. Backend Deployment

**Production Docker**:
```bash
# Build production image
docker build -t crybaby-backend .

# Run with production settings
docker run -d \
  --name crybaby-api \
  -p 8000:8000 \
  -e MODEL_PATH=/model/yamnet_lr_full.joblib \
  crybaby-backend
```

### 2. iOS App Store

1. **Prepare for Release**:
   - Update bundle identifier
   - Set production backend URL
   - Configure code signing

2. **Build & Archive**:
   - Select "Any iOS Device" target
   - Product → Archive
   - Upload to App Store Connect

3. **App Store Requirements**:
   - Privacy policy for microphone usage
   - App description and screenshots
   - Age rating and content warnings

## 🔧 Configuration

### Environment Variables

**Backend**:
- `MODEL_PATH`: Path to ML model file
- `CONFIDENCE_THRESHOLD`: Prediction confidence threshold
- `DEBUG`: Enable debug logging

**iOS**:
- Backend URL configuration in Settings
- Confidence threshold adjustment
- Data export options

## 📊 Monitoring

### Backend Monitoring
- Health check endpoint: `/health`
- Model loading status
- API response times and error rates

### iOS Monitoring
- Crash reporting and analytics
- User interaction tracking
- Performance metrics

## 🛠️ Maintenance

### Regular Tasks
1. **Model Updates**: Retrain with new feedback data
2. **Security Updates**: Keep dependencies current
3. **Performance Monitoring**: Track app performance
4. **User Feedback**: Respond to app store reviews

### Troubleshooting
- Check backend connectivity in iOS settings
- Verify microphone permissions
- Monitor backend logs for errors
- Test with different audio input sources
