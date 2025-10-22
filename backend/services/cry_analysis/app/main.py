#!/usr/bin/env python3
"""
FastAPI backend for CryBaby application
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import logging
from datetime import datetime
from typing import Optional
import sys

# Add the current directory to Python path to import cry_classifier
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from cry_classifier import CryClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CryBaby API",
    description="Baby cry analysis using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global classifier instance
classifier: Optional[CryClassifier] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the ML model on startup"""
    global classifier
    try:
        model_path = os.getenv("MODEL_PATH", "/model/yamnet_lr_full.joblib")
        confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))
        
        logger.info(f"Loading model from: {model_path}")
        classifier = CryClassifier(model_path, confidence_threshold=confidence_threshold)
        logger.info("Model loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "CryBaby API is running!", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "cry_analysis",
        "model_loaded": classifier is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/v1/model/info")
async def get_model_info():
    """
    Get information about the loaded ML model
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model_info = classifier.get_model_info()
        return {
            "success": True,
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")

@app.post("/api/v1/cry/predict")
async def predict_cry(audio: UploadFile = File(...)):
    """
    Predict baby cry type from audio file
    
    - **audio**: WAV audio file of baby crying (max 30 seconds)
    - Returns prediction with probabilities and confidence level
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not audio.filename.endswith(('.wav', '.WAV')):
        raise HTTPException(
            status_code=400, 
            detail="Only WAV files are supported"
        )
    
    # Save uploaded file temporarily
    temp_path = f"/tmp/{audio.filename}"
    
    try:
        # Read and save file
        content = await audio.read()
        
        # Check file size (max 10MB)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File too large. Maximum size is 10MB."
            )
        
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Get file info
        import soundfile as sf
        audio_data, sr = sf.read(temp_path)
        duration_seconds = len(audio_data) / sr
        
        # Check duration (max 30 seconds)
        if duration_seconds > 30:
            raise HTTPException(
                status_code=400,
                detail="Audio too long. Maximum duration is 30 seconds."
            )
        
        # Make prediction
        logger.info(f"Processing audio: {audio.filename} ({duration_seconds:.1f}s)")
        result = classifier.predict(temp_path)
        
        # Add metadata to result
        result.update({
            "filename": audio.filename,
            "duration_seconds": duration_seconds,
            "file_size_bytes": len(content),
            "model_version": "v1.0.0",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"Prediction: {result['predicted_label']} ({result['confidence_level']})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing audio: {str(e)}"
        )
    
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.post("/api/v1/cry/predict-batch")
async def predict_batch(
    files: list[UploadFile] = File(...)
):
    """
    Predict multiple audio files in batch
    
    - **files**: List of WAV audio files
    - Returns list of predictions
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 files per batch request"
        )
    
    results = []
    temp_files = []
    
    try:
        for audio_file in files:
            # Validate file type
            if not audio_file.filename.endswith(('.wav', '.WAV')):
                results.append({
                    "filename": audio_file.filename,
                    "error": "Only WAV files are supported"
                })
                continue
            
            # Save file temporarily
            temp_path = f"/tmp/{audio_file.filename}"
            content = await audio_file.read()
            
            with open(temp_path, "wb") as f:
                f.write(content)
            
            temp_files.append(temp_path)
            
            try:
                # Make prediction
                result = classifier.predict(temp_path)
                result["filename"] = audio_file.filename
                results.append(result)
                
            except Exception as e:
                results.append({
                    "filename": audio_file.filename,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "results": results,
            "total_files": len(files),
            "processed": len([r for r in results if "error" not in r]),
            "errors": len([r for r in results if "error" in r])
        }
        
    finally:
        # Cleanup temp files
        for temp_path in temp_files:
            if os.path.exists(temp_path):
                os.remove(temp_path)


@app.post("/api/v1/cry/feedback")
async def submit_feedback(feedback_data: dict):
    """
    Submit feedback for model improvement
    """
    # This endpoint would handle feedback submission
    # For now, just log the feedback
    logger.info(f"Received feedback: {feedback_data}")
    
    return {
        "success": True,
        "message": "Feedback received",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/v1/model/test")
async def test_model():
    """
    Test the model with a sample file (for debugging)
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Use a sample file if available
        sample_file = "/data/hungry/hung_yasin_lapar_11.wav"
        
        if not os.path.exists(sample_file):
            raise HTTPException(
                status_code=404,
                detail="Sample file not found. Please upload an audio file instead."
            )
        
        result = classifier.predict(sample_file)
        result["test_file"] = sample_file
        result["timestamp"] = datetime.utcnow().isoformat()
        
        return result
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        raise HTTPException(status_code=500, detail=f"Error testing model: {str(e)}")


@app.post("/api/v1/ai/advice")
async def get_ai_advice(advice_request: dict):
    """
    Generate AI-powered parenting advice based on cry analysis and activity data
    
    This endpoint would ideally connect to:
    - OpenAI GPT API for general parenting advice
    - Expert pediatric knowledge bases
    - Pattern analysis from collected data
    """
    try:
        # Parse request data
        cry_analysis_results = advice_request.get("cryAnalysisResults", [])
        baby_activities = advice_request.get("babyActivities", [])
        
        # For now, return mock advice based on patterns
        # TODO: Integrate with external AI services like OpenAI
        
        if not cry_analysis_results and not baby_activities:
            return {
                "success": True,
                "advice": {
                    "patterns": ["No data available yet"],
                    "recommendation": "Start recording your baby's activities and cries to get personalized insights.",
                    "dataPoints": 0,
                    "confidence": 0.0,
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
        
        # Analyze patterns from cry data
        patterns = []
        if cry_analysis_results:
            cry_types = [r.get("cryType", "unknown") for r in cry_analysis_results]
            hungry_count = cry_types.count("hungry")
            uncomfortable_count = cry_types.count("uncomfortable")
            
            if hungry_count > 0:
                patterns.append(f"Your baby shows hungry cries {hungry_count} time(s)")
            if uncomfortable_count > 0:
                patterns.append(f"Your baby shows uncomfortable signals {uncomfortable_count} time(s)")
        
        # Generate recommendation based on patterns
        recommendation = "Based on your baby's patterns, consider tracking feeding times and comfort needs more closely."
        
        if patterns:
            if any("hungry" in p for p in patterns):
                recommendation += " Try establishing a consistent feeding schedule."
            if any("uncomfortable" in p for p in patterns):
                recommendation += " Check for signs of overstimulation or discomfort."
        else:
            recommendation = "Continue monitoring your baby's patterns for more personalized insights."
        
        return {
            "success": True,
            "advice": {
                "patterns": patterns,
                "recommendation": recommendation,
                "dataPoints": len(cry_analysis_results) + len(baby_activities),
                "confidence": 0.7,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating advice: {e}")
        return {
            "success": False,
            "error": f"Failed to generate advice: {str(e)}"
        }


@app.post("/api/v1/ai/advisor")
async def get_ai_advisor_response(advisor_request: dict):
    """
    Advanced AI advisor endpoint for conversational baby advice
    
    Handles complex queries with full context including:
    - Baby profile information
    - Recent cry analysis results
    - Activity history
    - Parent questions
    """
    try:
        # Parse request data
        question = advisor_request.get("question", "")
        context_data = advisor_request.get("context", {})
        conversation_memory = advisor_request.get("conversationMemory", [])
        
        baby_profile = context_data.get("babyProfile", {})
        recent_cry_analyses = context_data.get("recentCryAnalyses", [])
        recent_activities = context_data.get("recentActivities", [])
        time_range = context_data.get("timeRange", "week")
        
        logger.info(f"AI Advisor request - Question: {question[:100]}..., Context: {len(recent_cry_analyses)} cries, {len(recent_activities)} activities, Memory: {len(conversation_memory)} items")
        
        # Add safety guardrails
        if is_urgent_medical_query(question):
            return generate_urgent_response(question)
        
        # Analyze the context and generate intelligent response
        analysis_result = analyze_baby_context(
            baby_profile=baby_profile,
            cry_analyses=recent_cry_analyses,
            activities=recent_activities,
            time_range=time_range,
            conversation_memory=conversation_memory
        )
        
        # Generate response based on question and analysis
        if question:
            response = generate_contextual_response(question, analysis_result, baby_profile, conversation_memory)
        else:
            response = generate_general_insights(analysis_result, baby_profile)
        
        return {
            "success": True,
            "advice": {
                "recommendation": response["recommendation"],
                "confidence": response.get("confidence", 0.7),
                "sources": response.get("sources", ["Pattern Analysis", "Pediatric Guidelines"]),
                "patterns": response.get("patterns", []),
                "timestamp": datetime.utcnow().isoformat(),
                "context_analysis": analysis_result
            }
        }
        
    except Exception as e:
        logger.error(f"Error in AI advisor: {e}")
        return {
            "success": False,
            "error": f"Failed to generate advisor response: {str(e)}"
        }


def is_urgent_medical_query(question: str) -> bool:
    """Check if the question indicates an urgent medical concern"""
    urgent_keywords = ["emergency", "urgent", "immediately", "call doctor", "hospital", "ambulance", "severe", "critical"]
    question_lower = question.lower()
    return any(keyword in question_lower for keyword in urgent_keywords)


def generate_urgent_response(question: str) -> dict:
    """Generate urgent medical response"""
    return {
        "success": True,
        "advice": {
            "recommendation": "⚠️ URGENT: This sounds like it may require immediate medical attention. Please contact your pediatrician or go to the emergency room if your baby is in distress. Do not wait for AI responses in emergency situations.",
            "confidence": 1.0,
            "sources": ["Emergency Protocol"],
            "patterns": ["Urgent medical concern detected"],
            "timestamp": datetime.utcnow().isoformat(),
            "urgency": "high"
        }
    }


def analyze_baby_context(baby_profile: dict, cry_analyses: list, activities: list, time_range: str, conversation_memory: list = None) -> dict:
    """
    Analyze baby's context to identify patterns and insights
    """
    analysis = {
        "cry_patterns": {},
        "activity_patterns": {},
        "recommendations": [],
        "health_indicators": {},
        "feeding_insights": {},
        "sleep_insights": {}
    }
    
    # Analyze cry patterns
    if cry_analyses:
        cry_types = [cry.get("cryType", "unknown") for cry in cry_analyses]
        analysis["cry_patterns"] = {
            "hungry_count": cry_types.count("hungry"),
            "uncomfortable_count": cry_types.count("uncomfortable"),
            "unknown_count": cry_types.count("unknown"),
            "total_cries": len(cry_types),
            "dominant_pattern": max(set(cry_types), key=cry_types.count) if cry_types else "none"
        }
    
    # Analyze baby age and feeding recommendations
    if baby_profile:
        age_in_days = baby_profile.get("ageInDays", 0)
        feeding_method = baby_profile.get("feedingMethod", "unknown")
        
        analysis["age_insights"] = {
            "age_days": age_in_days,
            "development_stage": get_development_stage(age_in_days),
            "feeding_needs": get_feeding_recommendations(age_in_days, feeding_method)
        }
    
    # Analyze activity patterns
    if activities:
        analysis["activity_patterns"] = analyze_activity_data(activities)
    
    return analysis


def get_development_stage(age_days: int) -> str:
    """Determine developmental stage based on age"""
    if age_days < 30:
        return "newborn"
    elif age_days < 120:
        return "infant_early"
    elif age_days < 365:
        return "infant_advanced"
    else:
        return "toddler"


def get_feeding_recommendations(age_days: int, feeding_method: str) -> dict:
    """Get age-appropriate feeding recommendations"""
    if age_days < 30:
        return {
            "frequency": "8-12 times per day",
            "duration": "15-30 minutes per feed",
            "signs": "Early hunger cues: lip-smacking, hand-to-mouth, rooting"
        }
    elif age_days < 120:
        return {
            "frequency": "6-8 times per day",
            "duration": "20-40 minutes per feed",
            "signs": "Watch for regular hunger patterns and growth spurts"
        }
    else:
        return {
            "frequency": "4-6 times per day",
            "duration": "Variable based on solids",
            "signs": "Interest in solids, self-feeding attempts"
        }


def analyze_activity_data(activities: list) -> dict:
    """Analyze baby activity data for patterns"""
    # This would analyze feeding, sleep, and diaper change patterns
    return {
        "feeding_frequency": len([a for a in activities if "feeding" in str(a)]),
        "sleep_sessions": len([a for a in activities if "sleep" in str(a)]),
        "diaper_changes": len([a for a in activities if "diaper" in str(a)])
    }


def generate_contextual_response(question: str, analysis: dict, baby_profile: dict, conversation_memory: list = None) -> dict:
    """Generate a contextual response based on the question and analysis"""
    question_lower = question.lower()
    
    # Determine response type based on question keywords
    if any(word in question_lower for word in ["hungry", "feeding", "eat", "food"]):
        return generate_feeding_response(analysis, baby_profile)
    elif any(word in question_lower for word in ["sleep", "nap", "tired", "bedtime"]):
        return generate_sleep_response(analysis, baby_profile)
    elif any(word in question_lower for word in ["cry", "fussy", "upset", "comfort"]):
        return generate_cry_response(analysis, baby_profile)
    else:
        return generate_general_response(question, analysis, baby_profile)


def generate_feeding_response(analysis: dict, baby_profile: dict) -> dict:
    """Generate feeding-related advice"""
    cry_patterns = analysis.get("cry_patterns", {})
    age_insights = analysis.get("age_insights", {})
    
    hungry_count = cry_patterns.get("hungry_count", 0)
    total_cries = cry_patterns.get("total_cries", 0)
    
    if hungry_count > 0 and total_cries > 0:
        hunger_ratio = hungry_count / total_cries
        if hunger_ratio > 0.6:
            recommendation = "Your baby shows frequent hunger cues. Consider reviewing feeding schedule or amounts. Ensure proper feeding technique and burping."
        elif hunger_ratio > 0.3:
            recommendation = "Some hunger cries detected. Monitor feeding times and ensure adequate milk/formula intake."
        else:
            recommendation = "Hunger cries are minimal. Continue current feeding routine."
    else:
        recommendation = "No recent hunger patterns detected. Maintain regular feeding schedule appropriate for your baby's age."
    
    feeding_needs = age_insights.get("feeding_needs", {})
    if feeding_needs:
        recommendation += f" For {feeding_needs.get('frequency', 'regular')} feeding schedule."
    
    return {
        "recommendation": recommendation,
        "confidence": 0.8,
        "sources": ["Cry Pattern Analysis", "Age-Appropriate Guidelines"],
        "patterns": [f"Hunger cries: {hungry_count}/{total_cries}"],
        "category": "feeding"
    }


def generate_sleep_response(analysis: dict, baby_profile: dict) -> dict:
    """Generate sleep-related advice"""
    age_insights = analysis.get("age_insights", {})
    age_days = age_insights.get("age_days", 0)
    
    if age_days < 90:
        recommendation = "Newborns need 14-17 hours of sleep per day. Establish a calming bedtime routine and watch for sleep cues like yawning or eye-rubbing."
    elif age_days < 365:
        recommendation = "Infants typically need 12-15 hours of sleep. Maintain consistent nap times and bedtime routines. Avoid overstimulation before sleep."
    else:
        recommendation = "Toddlers need 11-14 hours of sleep. Establish clear bedtime boundaries and consistent routines."
    
    return {
        "recommendation": recommendation,
        "confidence": 0.8,
        "sources": ["Sleep Guidelines", "Age Development"],
        "patterns": [],
        "category": "sleep"
    }


def generate_cry_response(analysis: dict, baby_profile: dict) -> dict:
    """Generate crying-related advice"""
    cry_patterns = analysis.get("cry_patterns", {})
    dominant_pattern = cry_patterns.get("dominant_pattern", "unknown")
    unknown_count = cry_patterns.get("unknown_count", 0)
    total_cries = cry_patterns.get("total_cries", 0)
    
    if unknown_count > total_cries * 0.3:
        recommendation = "Many cries couldn't be classified. Check environmental factors: temperature, diaper condition, overstimulation, or need for comfort/contact."
    elif dominant_pattern == "hungry":
        recommendation = "Your baby's cries primarily indicate hunger. Try feeding on demand and establish a more regular feeding routine."
    elif dominant_pattern == "uncomfortable":
        recommendation = "Uncomfortable cries are common. Check for diaper changes, gas, or need for position changes. Consider swaddling or gentle rocking."
    else:
        recommendation = "Babies cry for various reasons. Try the basics: feeding, diaper change, burping, or comfort. Sometimes babies cry to release stress."
    
    return {
        "recommendation": recommendation,
        "confidence": 0.7,
        "sources": ["Cry Analysis", "Comfort Strategies"],
        "patterns": [f"Primary pattern: {dominant_pattern}"],
        "category": "comfort"
    }


def generate_general_response(question: str, analysis: dict, baby_profile: dict) -> dict:
    """Generate general advice response"""
    age_insights = analysis.get("age_insights", {})
    development_stage = age_insights.get("development_stage", "unknown")
    
    recommendation = f"Based on your baby's development stage ({development_stage}), I'd recommend consulting with your pediatrician for specific concerns. General guidance includes monitoring feeding, sleep, and comfort needs while tracking your baby's unique patterns."
    
    return {
        "recommendation": recommendation,
        "confidence": 0.6,
        "sources": ["General Guidelines", "Pattern Analysis"],
        "patterns": [],
        "category": "general"
    }


def generate_general_insights(analysis: dict, baby_profile: dict) -> dict:
    """Generate general insights when no specific question is asked"""
    insights = []
    recommendations = []
    
    # Analyze patterns and generate insights
    cry_patterns = analysis.get("cry_patterns", {})
    if cry_patterns:
        total_cries = cry_patterns.get("total_cries", 0)
        if total_cries > 0:
            insights.append(f"Analyzed {total_cries} recent crying episodes")
            
            hungry_pct = (cry_patterns.get("hungry_count", 0) / total_cries) * 100
            if hungry_pct > 50:
                recommendations.append("Consider reviewing feeding schedule - frequent hunger cues detected")
    
    age_insights = analysis.get("age_insights", {})
    if age_insights:
        development_stage = age_insights.get("development_stage", "")
        insights.append(f"Baby is in {development_stage} developmental stage")
    
    recommendation = "Based on your baby's recent patterns, " + ". ".join(recommendations) if recommendations else "Continue monitoring your baby's patterns for personalized insights."
    
    return {
        "recommendation": recommendation,
        "confidence": 0.7,
        "sources": ["Pattern Analysis", "Developmental Guidelines"],
        "patterns": insights,
        "category": "general"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
