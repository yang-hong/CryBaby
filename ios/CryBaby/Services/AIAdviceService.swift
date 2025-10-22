//
//  AIAdviceService.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import Foundation
import SwiftUI

class AIAdviceService: ObservableObject {
    private let backendURL = "http://127.0.0.1:8000" // Local backend for development
    
    // For external AI API integration
    private let openAIAPIKey = "" // Add your OpenAI API key here
    private let openAIBaseURL = "https://api.openai.com/v1/chat/completions"
    
    func generateAdvice(
        cryAnalysisData: [CryAnalysisResult]?,
        activityData: [BabyActivity]?,
        completion: @escaping (AIAdvice?) -> Void
    ) {
        
        // First try to get advice from backend if available
        getAdviceFromBackend(cryAnalysisData: cryAnalysisData, activityData: activityData) { backendAdvice in
            if let backendAdvice = backendAdvice {
                completion(backendAdvice)
            } else {
                // Fallback to mock data for now
                self.generateMockAdvice(completion: completion)
            }
        }
    }
    
    private func getAdviceFromBackend(
        cryAnalysisData: [CryAnalysisResult]?,
        activityData: [BabyActivity]?,
        completion: @escaping (AIAdvice?) -> Void
    ) {
        guard let url = URL(string: "\(backendURL)/api/v1/ai/advice") else {
            completion(nil)
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestData = AdviceRequest(
            cryAnalysisResults: cryAnalysisData ?? [],
            babyActivities: activityData ?? []
        )
        
        do {
            request.httpBody = try JSONEncoder().encode(requestData)
        } catch {
            print("Failed to encode advice request: \(error)")
            completion(nil)
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    print("Advice request error: \(error)")
                    completion(nil)
                    return
                }
                
                guard let data = data else {
                    print("No data received for advice")
                    completion(nil)
                    return
                }
                
                do {
                    // Parse the backend response structure
                    let jsonResponse = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    
                    if let success = jsonResponse?["success"] as? Bool, success,
                       let adviceData = jsonResponse?["advice"] as? [String: Any] {
                        
                        let advice = AIAdvice(
                            patterns: adviceData["patterns"] as? [String] ?? [],
                            recommendation: adviceData["recommendation"] as? String ?? "No specific advice available.",
                            dataPoints: adviceData["dataPoints"] as? Int ?? 0,
                            confidence: adviceData["confidence"] as? Double,
                            timestamp: Date() // Use current date if not available in response
                        )
                        completion(advice)
                    } else {
                        print("Invalid response format from backend")
                        completion(nil)
                    }
                } catch {
                    print("Failed to decode advice response: \(error)")
                    completion(nil)
                }
            }
        }.resume()
    }
    
    // MARK: - External AI Integration (OpenAI/ChatGPT)
    
    func generateAdviceWithOpenAI(
        cryAnalysisData: [CryAnalysisResult]?,
        activityData: [BabyActivity]?,
        completion: @escaping (AIAdvice?) -> Void
    ) {
        
        guard !openAIAPIKey.isEmpty else {
            print("OpenAI API key not configured")
            completion(nil)
            return
        }
        
        // Prepare the context for AI
        let context = buildContextString(cryAnalysisData: cryAnalysisData, activityData: activityData)
        
        let requestBody = OpenAIRequest(
            model: "gpt-3.5-turbo",
            messages: [
                OpenAIMessage(
                    role: "system",
                    content: "You are a helpful parenting assistant specializing in baby care. Analyze the provided baby activity and cry analysis data to give personalized, evidence-based advice."
                ),
                OpenAIMessage(
                    role: "user", 
                    content: context
                )
            ],
            max_tokens: 300
        )
        
        guard let url = URL(string: openAIBaseURL) else {
            completion(nil)
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.setValue("Bearer \(openAIAPIKey)", forHTTPHeaderField: "Authorization")
        
        do {
            request.httpBody = try JSONEncoder().encode(requestBody)
        } catch {
            print("Failed to encode OpenAI request: \(error)")
            completion(nil)
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                if let error = error {
                    print("OpenAI request error: \(error)")
                    completion(nil)
                    return
                }
                
                guard let data = data else {
                    print("No data received from OpenAI")
                    completion(nil)
                    return
                }
                
                do {
                    let openAIResponse = try JSONDecoder().decode(OpenAIResponse.self, from: data)
                    if let choice = openAIResponse.choices.first {
                        let advice = self.parseAIResponse(choice.message.content)
                        completion(advice)
                    } else {
                        completion(nil)
                    }
                } catch {
                    print("Failed to decode OpenAI response: \(error)")
                    completion(nil)
                }
            }
        }.resume()
    }
    
    private func buildContextString(
        cryAnalysisData: [CryAnalysisResult]?,
        activityData: [BabyActivity]?
    ) -> String {
        var context = "Please analyze this baby's data and provide personalized parenting advice:\n\n"
        
        if let cryData = cryAnalysisData, !cryData.isEmpty {
            context += "CRY ANALYSIS:\n"
            for (index, result) in cryData.enumerated() {
                context += "\(index + 1). Type: \(result.cryType.rawValue), Confidence: \(Int(result.confidence * 100))%\n"
            }
            context += "\n"
        }
        
        if let activities = activityData, !activities.isEmpty {
            context += "DAILY ACTIVITIES:\n"
            for activity in activities.prefix(10) { // Limit to recent 10 activities
                let feedingCount = activity.feedingTimes.count
                let diaperCount = activity.diaperChanges.count
                let sleepCount = activity.sleepSessions.count
                let cryCount = activity.cryingEpisodes.count
                context += "- Date: \(activity.date) - Feedings: \(feedingCount), Diapers: \(diaperCount), Sleep: \(sleepCount), Cries: \(cryCount)\n"
            }
            context += "\n"
        }
        
        context += "Please provide: 1-2 specific, actionable recommendations based on this data."
        
        return context
    }
    
    private func parseAIResponse(_ content: String) -> AIAdvice {
        return AIAdvice(
            patterns: [content], // Simple parsing - could be improved
            recommendation: content,
            dataPoints: 0, // Could be calculated from actual data
            confidence: 0.8, // Default confidence
            timestamp: Date()
        )
    }
    
    private func generateMockAdvice(completion: @escaping (AIAdvice?) -> Void) {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            let mockAdvice = AIAdvice(
                patterns: ["Your baby typically cries for food every 2-3 hours"],
                recommendation: "Based on your baby's patterns, try offering a feeding before the typical evening fussy period.",
                dataPoints: 5,
                confidence: 0.78,
                timestamp: Date()
            )
            completion(mockAdvice)
        }
    }
}

// MARK: - Data Models for AI Advice

struct AdviceRequest: Codable {
    let cryAnalysisResults: [CryAnalysisResult]
    let babyActivities: [BabyActivity]
}

struct OpenAIRequest: Codable {
    let model: String
    let messages: [OpenAIMessage]
    let max_tokens: Int
}

struct OpenAIMessage: Codable {
    let role: String
    let content: String
}

struct OpenAIResponse: Codable {
    let choices: [OpenAIChoice]
}

struct OpenAIChoice: Codable {
    let message: OpenAIMessage
}
