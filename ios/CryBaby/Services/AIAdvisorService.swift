//
//  AIAdvisorService.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import Foundation
import SwiftUI

// MARK: - AI Advisor Models
struct AdvisorSession: Identifiable, Codable {
    let id: UUID
    let startTime: Date
    var context: AdvisorContext
    var messages: [AdvisorMessage]
    var recommendations: [AdvisorRecommendation]
    
    init(context: AdvisorContext, messages: [AdvisorMessage] = [], recommendations: [AdvisorRecommendation] = []) {
        self.id = UUID()
        self.startTime = Date()
        self.context = context
        self.messages = messages
        self.recommendations = recommendations
    }
    
    mutating func addMessage(_ message: AdvisorMessage) {
        messages.append(message)
    }
    
    mutating func addRecommendation(_ recommendation: AdvisorRecommendation) {
        recommendations.append(recommendation)
    }
    
    mutating func updateContext(_ newContext: AdvisorContext) {
        context = newContext
    }
}

struct AdvisorContext: Codable {
    let babyProfile: BabyProfile?
    let recentCryAnalyses: [CryAnalysisResult]
    let recentActivities: [BabyActivity]
    let timeRange: TimeRange
    let parentQuestion: String?
    
    enum TimeRange: String, Codable {
        case today = "today"
        case week = "week"
        case month = "month"
        case all = "all"
    }
}

struct AdvisorMessage: Identifiable, Codable {
    let id: UUID
    let timestamp: Date
    let role: MessageRole
    let content: String
    let confidence: Double?
    let sources: [String]?
    
    init(role: MessageRole, content: String, confidence: Double? = nil, sources: [String]? = nil) {
        self.id = UUID()
        self.timestamp = Date()
        self.role = role
        self.content = content
        self.confidence = confidence
        self.sources = sources
    }
}

enum MessageRole: String, Codable {
    case user = "user"
    case advisor = "advisor"
    case system = "system"
}

struct AdvisorRecommendation: Identifiable, Codable {
    let id: UUID
    let timestamp: Date
    let category: RecommendationCategory
    let title: String
    let description: String
    let priority: Priority
    let actionable: Bool
    let estimatedImpact: Double // 0-1
    let sources: [String]
    
    init(category: RecommendationCategory, title: String, description: String, priority: Priority = .medium, actionable: Bool = true, estimatedImpact: Double = 0.5, sources: [String] = []) {
        self.id = UUID()
        self.timestamp = Date()
        self.category = category
        self.title = title
        self.description = description
        self.priority = priority
        self.actionable = actionable
        self.estimatedImpact = estimatedImpact
        self.sources = sources
    }
}

enum RecommendationCategory: String, CaseIterable, Codable {
    case feeding = "feeding"
    case sleep = "sleep"
    case comfort = "comfort"
    case development = "development"
    case health = "health"
    case safety = "safety"
    case routine = "routine"
}

enum Priority: String, CaseIterable, Codable {
    case low = "low"
    case medium = "medium"
    case high = "high"
    case urgent = "urgent"
}

// MARK: - AI Advisor Service
@MainActor
class AIAdvisorService: ObservableObject {
    @Published var currentSession: AdvisorSession?
    @Published var isLoading = false
    @Published var errorMessage: String?
    
    private let backendURL = "http://127.0.0.1:8000"
    private let openAIAPIKey = "" // Add your API key here
    private let knowledgeBase = PediatricKnowledgeBase()
    private let conversationManager = ConversationManager()
    
    // MARK: - Public Methods
    
    func startSession(context: AdvisorContext) {
        conversationManager.startNewSession(context: context)
        currentSession = conversationManager.currentSession
        
        // Generate initial recommendations based on context
        generateInitialRecommendations(for: context)
    }
    
    func askQuestion(_ question: String, completion: @escaping (AdvisorMessage?) -> Void) {
        guard var session = currentSession else {
            completion(nil)
            return
        }
        
        isLoading = true
        errorMessage = nil
        
        let userMessage = AdvisorMessage(role: .user, content: question)
        session.addMessage(userMessage)
        currentSession = session
        
        // Save user message to conversation manager
        conversationManager.addMessage(userMessage)
        
        // Generate AI response with conversation context
        let conversationMemory = conversationManager.getContextualMemory()
        generateAIResponse(for: question, context: session.context, memory: conversationMemory) { [weak self] response in
            Task { @MainActor in
                self?.isLoading = false
                if let response = response {
                    var updatedSession = self?.currentSession
                    updatedSession?.addMessage(response)
                    self?.currentSession = updatedSession
                    self?.conversationManager.addMessage(response)
                    self?.conversationManager.saveCurrentSession()
                    completion(response)
                } else {
                    self?.errorMessage = "Failed to generate response"
                    completion(nil)
                }
            }
        }
    }
    
    func generateRecommendations(for timeRange: AdvisorContext.TimeRange = .week, completion: @escaping ([AdvisorRecommendation]) -> Void) {
        guard let context = currentSession?.context else {
            completion([])
            return
        }
        
        // Analyze patterns and generate recommendations
        let recommendations = analyzePatternsAndGenerateRecommendations(context: context, timeRange: timeRange)
        completion(recommendations)
    }
    
    // MARK: - Private Methods
    
    private func generateInitialRecommendations(for context: AdvisorContext) {
        Task {
            let recommendations = analyzePatternsAndGenerateRecommendations(context: context, timeRange: .week)
            await MainActor.run {
                currentSession?.recommendations = recommendations
            }
        }
    }
    
    private func generateAIResponse(for question: String, context: AdvisorContext, memory: [String] = [], completion: @escaping (AdvisorMessage?) -> Void) {
        // Try multiple AI providers in order of preference
        generateResponseFromBackend(question: question, context: context, memory: memory) { [weak self] response in
            if let response = response {
                completion(response)
            } else {
                // Fallback to OpenAI if backend fails and API key is configured
                self?.generateResponseFromOpenAI(question: question, context: context, memory: memory, completion: completion)
            }
        }
    }
    
    private func generateResponseFromBackend(question: String, context: AdvisorContext, memory: [String] = [], completion: @escaping (AdvisorMessage?) -> Void) {
        guard let url = URL(string: "\(backendURL)/api/v1/ai/advisor") else {
            completion(nil)
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody: [String: Any] = [
            "question": question,
            "context": [
                "babyProfile": context.babyProfile?.dictionaryRepresentation ?? [:],
                "recentCryAnalyses": context.recentCryAnalyses.map { $0.dictionaryRepresentation },
                "recentActivities": context.recentActivities.map { $0.dictionaryRepresentation },
                "timeRange": context.timeRange.rawValue
            ],
            "conversationMemory": memory
        ]
        
        do {
            request.httpBody = try JSONSerialization.data(withJSONObject: requestBody, options: [])
        } catch {
            completion(nil)
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                guard let data = data,
                      let jsonResponse = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                      let success = jsonResponse["success"] as? Bool,
                      success,
                      let advice = jsonResponse["advice"] as? [String: Any],
                      let content = advice["recommendation"] as? String else {
                    completion(nil)
                    return
                }
                
                let advisorMessage = AdvisorMessage(
                    role: .advisor,
                    content: content,
                    confidence: advice["confidence"] as? Double,
                    sources: advice["sources"] as? [String]
                )
                completion(advisorMessage)
            }
        }.resume()
    }
    
    private func generateResponseFromOpenAI(question: String, context: AdvisorContext, memory: [String] = [], completion: @escaping (AdvisorMessage?) -> Void) {
        guard !openAIAPIKey.isEmpty && openAIAPIKey != "" else {
            // Fallback to knowledge base
            generateResponseFromKnowledgeBase(question: question, context: context, completion: completion)
            return
        }
        
        // Implementation for OpenAI API call
        // This would be similar to the existing OpenAI implementation in AIAdviceService
        // but with more sophisticated prompt engineering for conversational AI
        
        let prompt = buildPromptForOpenAI(question: question, context: context, memory: memory)
        
        // TODO: Implement OpenAI API call with conversation history
        // For now, fallback to knowledge base
        generateResponseFromKnowledgeBase(question: question, context: context, completion: completion)
    }
    
    private func generateResponseFromKnowledgeBase(question: String, context: AdvisorContext, completion: @escaping (AdvisorMessage?) -> Void) {
        // Use local knowledge base to generate response with safety guardrails
        let response = knowledgeBase.generateResponse(for: question, context: context)
        let safeResponse = addSafetyGuardrails(response, question: question, context: context)
        
        let advisorMessage = AdvisorMessage(
            role: .advisor,
            content: safeResponse,
            confidence: 0.7,
            sources: ["Pediatric Knowledge Base", "Safety Guidelines"]
        )
        completion(advisorMessage)
    }
    
    private func buildPromptForOpenAI(question: String, context: AdvisorContext, memory: [String] = []) -> String {
        var prompt = """
        You are an expert pediatric AI advisor. Analyze the following context and answer the parent's question.
        
        IMPORTANT SAFETY NOTICE: 
        - Always recommend consulting a pediatrician for medical concerns
        - Do not provide specific medical diagnoses
        - Focus on general guidance and pattern recognition
        
        Baby Information:
        - Age: \(context.babyProfile?.ageInDays ?? 0) days old
        - Feeding Method: \(context.babyProfile?.feedingMethod.rawValue ?? "unknown")
        
        Recent Cry Analysis Results: \(context.recentCryAnalyses.count) entries
        Recent Activities: \(context.recentActivities.count) entries
        
        Conversation Context: \(memory.joined(separator: "; "))
        
        Parent Question: \(question)
        
        Provide helpful, evidence-based advice. Always include sources when possible and remind user to consult pediatrician if needed.
        """
        
        return prompt
    }
    
    // MARK: - Safety Guardrails
    
    private func addSafetyGuardrails(_ response: String, question: String, context: AdvisorContext) -> String {
        let questionLower = question.lowercased()
        
        // Check for medical concerns
        let medicalKeywords = ["sick", "fever", "illness", "medicine", "medication", "pain", "hurt", "injury", "emergency"]
        let isMedicalQuery = medicalKeywords.contains { questionLower.contains($0) }
        
        // Check for urgent concerns
        let urgentKeywords = ["emergency", "urgent", "immediately", "call doctor", "hospital"]
        let isUrgent = urgentKeywords.contains { questionLower.contains($0) }
        
        var finalResponse = response
        
        if isUrgent {
            finalResponse = "⚠️ URGENT: This sounds like it may require immediate medical attention. Please contact your pediatrician or go to the emergency room if your baby is in distress."
        } else if isMedicalQuery {
            finalResponse = response + "\n\n⚠️ IMPORTANT: For medical concerns, please consult with your pediatrician. This advice is for general guidance only and should not replace professional medical care."
        } else {
            // Add general disclaimer for all responses
            finalResponse = response + "\n\n💡 Remember: Always consult your pediatrician with specific concerns about your baby's health and development."
        }
        
        return finalResponse
    }
    
    private func analyzePatternsAndGenerateRecommendations(context: AdvisorContext, timeRange: AdvisorContext.TimeRange) -> [AdvisorRecommendation] {
        var recommendations: [AdvisorRecommendation] = []
        
        // Analyze cry patterns
        if let cryRecommendation = analyzeCryPatterns(context: context) {
            recommendations.append(cryRecommendation)
        }
        
        // Analyze feeding patterns
        if let feedingRecommendation = analyzeFeedingPatterns(context: context) {
            recommendations.append(feedingRecommendation)
        }
        
        // Analyze sleep patterns
        if let sleepRecommendation = analyzeSleepPatterns(context: context) {
            recommendations.append(sleepRecommendation)
        }
        
        // Generate age-appropriate recommendations
        if let babyProfile = context.babyProfile {
            recommendations.append(contentsOf: generateAgeAppropriateRecommendations(for: babyProfile))
        }
        
        return recommendations.sorted { $0.priority.rawValue > $1.priority.rawValue }
    }
    
    private func analyzeCryPatterns(context: AdvisorContext) -> AdvisorRecommendation? {
        let cryAnalyses = context.recentCryAnalyses
        
        if cryAnalyses.isEmpty { return nil }
        
        let hungryCount = cryAnalyses.filter { $0.cryType == .hungry }.count
        let uncomfortableCount = cryAnalyses.filter { $0.cryType == .uncomfortable }.count
        let unknownCount = cryAnalyses.filter { $0.cryType == .unknown }.count
        
        if hungryCount > uncomfortableCount + 2 {
            return AdvisorRecommendation(
                category: .feeding,
                title: "Frequent Hunger Cues Detected",
                description: "Your baby shows frequent hungry cries. Consider reviewing feeding schedule or amounts.",
                priority: .high,
                estimatedImpact: 0.8,
                sources: ["Cry Analysis Patterns"]
            )
        }
        
        if unknownCount > cryAnalyses.count / 3 {
            return AdvisorRecommendation(
                category: .comfort,
                title: "Consider Environmental Factors",
                description: "Many cries couldn't be classified. Check for temperature, diaper, or overstimulation.",
                priority: .medium,
                estimatedImpact: 0.6,
                sources: ["Cry Analysis Patterns"]
            )
        }
        
        return nil
    }
    
    private func analyzeFeedingPatterns(context: AdvisorContext) -> AdvisorRecommendation? {
        // Analyze feeding patterns from activities
        // This would analyze timing, duration, and amounts
        return nil // Placeholder
    }
    
    private func analyzeSleepPatterns(context: AdvisorContext) -> AdvisorRecommendation? {
        // Analyze sleep patterns from activities
        // This would analyze timing and duration
        return nil // Placeholder
    }
    
    private func generateAgeAppropriateRecommendations(for profile: BabyProfile) -> [AdvisorRecommendation] {
        let ageInDays = profile.ageInDays
        var recommendations: [AdvisorRecommendation] = []
        
        if ageInDays < 30 {
            recommendations.append(AdvisorRecommendation(
                category: .feeding,
                title: "Newborn Feeding",
                description: "Newborns typically need 8-12 feeds per day. Watch for hunger cues.",
                priority: .high,
                sources: ["AAP Guidelines"]
            ))
        } else if ageInDays < 180 {
            recommendations.append(AdvisorRecommendation(
                category: .development,
                title: "Tummy Time",
                description: "Aim for 15-20 minutes of supervised tummy time daily to strengthen neck and back muscles.",
                priority: .medium,
                sources: ["AAP Guidelines"]
            ))
        }
        
        return recommendations
    }
}

// MARK: - Knowledge Base
class PediatricKnowledgeBase {
    
    func generateResponse(for question: String, context: AdvisorContext) -> String {
        // Simple rule-based responses based on keywords and context
        let lowercaseQuestion = question.lowercased()
        
        if lowercaseQuestion.contains("hungry") || lowercaseQuestion.contains("feeding") {
            return generateFeedingResponse(context: context)
        } else if lowercaseQuestion.contains("sleep") || lowercaseQuestion.contains("tired") {
            return generateSleepResponse(context: context)
        } else if lowercaseQuestion.contains("cry") || lowercaseQuestion.contains("fussy") {
            return generateCryResponse(context: context)
        } else {
            return "I'd be happy to help with your question about your baby. Based on the data I have, I can provide general guidance. For specific concerns, please consult with your pediatrician."
        }
    }
    
    private func generateFeedingResponse(context: AdvisorContext) -> String {
        if let profile = context.babyProfile {
            let ageInDays = profile.ageInDays
            if ageInDays < 30 {
                return "Newborns typically need 8-12 feeds per day, about every 2-3 hours. Watch for early hunger cues like lip-smacking or bringing hands to mouth."
            } else if ageInDays < 180 {
                return "Babies at this age usually feed 6-8 times per day. Look for hunger cues and try to feed before the baby becomes overly upset."
            }
        }
        return "Establishing a regular feeding schedule can help both you and your baby. Pay attention to your baby's hunger cues and try to feed consistently."
    }
    
    private func generateSleepResponse(context: AdvisorContext) -> String {
        return "Healthy sleep is crucial for your baby's development. Try to establish a consistent bedtime routine and watch for sleep cues like yawning or rubbing eyes."
    }
    
    private func generateCryResponse(context: AdvisorContext) -> String {
        let recentCries = context.recentCryAnalyses
        if !recentCries.isEmpty {
            let lastCry = recentCries.last!
            return "Based on recent cry analysis, your baby appears to be \(lastCry.cryType.rawValue). Try addressing this need first - check if they need feeding, comfort, or a diaper change."
        }
        return "When your baby cries, try the basics first: check their diaper, offer feeding, or provide comfort. Sometimes babies cry to release stress or overstimulation."
    }
}

// MARK: - Extensions
extension Encodable {
    var dictionaryRepresentation: [String: Any]? {
        guard let data = try? JSONEncoder().encode(self) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data, options: .allowFragments)).flatMap { $0 as? [String: Any] }
    }
}
