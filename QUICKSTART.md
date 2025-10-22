# CryBaby Quick Start Guide

Get up and running with CryBaby in under 10 minutes!

## 🚀 Prerequisites

- **macOS** with Xcode 15.0+
- **Python 3.11+** and pip
- **Docker Desktop** (for backend)
- **Git** (optional)

## ⚡ Quick Setup

### 1. Backend Setup (2 minutes)

```bash
# Navigate to backend directory
cd backend/services/cry_analysis

# Start the FastAPI service with Docker
docker compose up -d

# Verify it's running
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "cry_analysis",
  "model_loaded": true,
  "timestamp": "2025-10-20T..."
}
```

### 2. iOS App Setup (3 minutes)

```bash
# Navigate to iOS directory
cd ../../ios/CryBaby

# Open project in Xcode
open CryBaby.xcodeproj
```

In Xcode:
1. Select iPhone 15 Simulator (or your device)
2. Press **⌘+R** to build and run
3. Grant microphone permissions when prompted

### 3. Test the System (2 minutes)

1. **In the iOS app**: Go to "Cry Analysis" tab
2. **Tap the record button** and record a short audio
3. **View the results** - should show prediction with confidence scores

## 🧪 Alternative Testing (Web UI)

If you prefer to test via web browser:

```bash
# Start the web frontend (separate terminal)
cd frontend/web-ui
python3 server.py
```

Then open: http://localhost:3000

## 🔧 Troubleshooting

### Backend Issues

**Docker not starting:**
```bash
# Check if Docker Desktop is running
docker --version

# If not installed:
# Download Docker Desktop from docker.com
```

**Port 8000 busy:**
```bash
# Find what's using the port
lsof -i :8000

# Kill the process (replace PID)
kill -9 <PID>
```

**Model not loading:**
```bash
# Check if model file exists
ls -la ../../artifacts_option2/yamnet_lr_full.joblib

# If missing, ensure you have the trained model file
```

### iOS Issues

**Build errors in Xcode:**
- Clean build folder: **⌘+Shift+K**
- Restart Xcode
- Check that all .swift files are included in project

**Microphone permission:**
- Go to iOS Settings > Privacy & Security > Microphone
- Enable CryBaby app

**App not connecting to backend:**
- Ensure backend is running: `curl http://localhost:8000/health`
- Check iOS simulator can reach localhost
- Verify backend URL in app settings

## 📱 App Features Overview

Once everything is running, you can test:

### 🎵 Cry Analysis
- Record baby cries using the microphone
- View AI predictions (hungry/uncomfortable/unknown)
- See confidence scores and suggestions
- Provide feedback to improve the model

### 📅 Daily Activities
- Add feeding times and amounts
- Record diaper changes
- Track sleep sessions
- View daily summaries

### 🤖 AI Advice
- Get personalized recommendations
- View pattern analysis
- Access expert insights

### ⚙️ Settings
- Adjust confidence thresholds
- Configure backend URL
- Manage app preferences

## 🔄 Development Workflow

### Making Changes

**Backend changes:**
```bash
# Restart service after code changes
docker compose restart cry_analysis

# View logs
docker compose logs -f cry_analysis
```

**iOS changes:**
- Save in Xcode (auto-compiles)
- Press **⌘+R** to reload in simulator

### Debugging

**Backend debugging:**
```bash
# Run without Docker for easier debugging
cd backend/services/cry_analysis/app
python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**iOS debugging:**
- Use Xcode debugger and breakpoints
- Check Console for network errors
- Test with different audio files

## 📊 Expected Results

After setup, you should see:

1. **Backend**: Healthy status response at http://localhost:8000/health
2. **iOS App**: Four tabs (Cry Analysis, Daily Activities, AI Advice, Settings)
3. **Recording**: Microphone button activates and records audio
4. **Analysis**: Results appear with confidence scores and suggestions

## 🆘 Need Help?

**Common Issues:**
- Backend not responding → Check Docker status
- iOS build fails → Clean and rebuild in Xcode
- Audio upload fails → Check file format (WAV only)
- No predictions → Verify model file exists

**Next Steps:**
- Try recording different types of audio
- Test the feedback system
- Explore the daily activity tracking
- Customize settings and thresholds

You should now have a fully functional CryBaby system! 🎉
