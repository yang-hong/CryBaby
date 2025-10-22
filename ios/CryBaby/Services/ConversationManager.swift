//
//  ConversationManager.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import Foundation
import SwiftUI

class ConversationManager: ObservableObject {
    @Published var currentSession: AdvisorSession?
    @Published var sessionHistory: [AdvisorSession] = []
    
    private let userDefaults = UserDefaults.standard
    private let sessionKey = "advisor_sessions"
    private let currentSessionKey = "current_advisor_session"
    
    init() {
        loadSessionHistory()
        loadCurrentSession()
    }
    
    // MARK: - Session Management
    
    func startNewSession(context: AdvisorContext) {
        // Save previous session if exists
        if let currentSession = currentSession {
            saveSession(currentSession)
        }
        
        // Create new session
        currentSession = AdvisorSession(context: context)
    }
    
    func updateCurrentSession(messages: [AdvisorMessage], recommendations: [AdvisorRecommendation]) {
        guard var session = currentSession else { return }
        
        // Note: Since AdvisorSession uses let properties, we need to create a new instance
        let updatedSession = AdvisorSession(
            context: session.context,
            messages: messages,
            recommendations: recommendations
        )
        currentSession = updatedSession
    }
    
    func addMessage(_ message: AdvisorMessage) {
        guard var session = currentSession else { return }
        
        var updatedMessages = session.messages
        updatedMessages.append(message)
        
        updateCurrentSession(messages: updatedMessages, recommendations: session.recommendations)
    }
    
    func addRecommendation(_ recommendation: AdvisorRecommendation) {
        guard var session = currentSession else { return }
        
        var updatedRecommendations = session.recommendations
        updatedRecommendations.append(recommendation)
        
        updateCurrentSession(messages: session.messages, recommendations: updatedRecommendations)
    }
    
    // MARK: - Persistence
    
    func saveSession(_ session: AdvisorSession) {
        sessionHistory.append(session)
        saveSessionHistory()
        
        // Keep only last 10 sessions to manage storage
        if sessionHistory.count > 10 {
            sessionHistory = Array(sessionHistory.suffix(10))
        }
    }
    
    func saveCurrentSession() {
        guard let session = currentSession else { return }
        
        do {
            let data = try JSONEncoder().encode(session)
            userDefaults.set(data, forKey: currentSessionKey)
        } catch {
            print("Failed to save current session: \(error)")
        }
    }
    
    private func loadSessionHistory() {
        guard let data = userDefaults.data(forKey: sessionKey),
              let sessions = try? JSONDecoder().decode([AdvisorSession].self, from: data) else {
            return
        }
        
        sessionHistory = sessions
    }
    
    private func loadCurrentSession() {
        guard let data = userDefaults.data(forKey: currentSessionKey),
              let session = try? JSONDecoder().decode(AdvisorSession.self, from: data) else {
            return
        }
        
        currentSession = session
    }
    
    private func saveSessionHistory() {
        do {
            let data = try JSONEncoder().encode(sessionHistory)
            userDefaults.set(data, forKey: sessionKey)
        } catch {
            print("Failed to save session history: \(error)")
        }
    }
    
    // MARK: - Context Management
    
    func updateContext(_ context: AdvisorContext) {
        guard var session = currentSession else { return }
        
        let updatedSession = AdvisorSession(
            context: context,
            messages: session.messages,
            recommendations: session.recommendations
        )
        currentSession = updatedSession
    }
    
    func getContextualMemory() -> [String] {
        guard let session = currentSession else { return [] }
        
        // Extract key insights from recent conversation
        let recentMessages = session.messages.suffix(5)
        var memory: [String] = []
        
        for message in recentMessages {
            if message.role == .user {
                // Remember user's key concerns or questions
                if message.content.lowercased().contains("hungry") {
                    memory.append("User asked about hunger/feeding concerns")
                }
                if message.content.lowercased().contains("sleep") {
                    memory.append("User asked about sleep patterns")
                }
                if message.content.lowercased().contains("cry") {
                    memory.append("User asked about crying behavior")
                }
            } else if message.role == .advisor {
                // Remember advisor's key recommendations
                if message.confidence ?? 0 > 0.8 {
                    memory.append("High confidence advice provided: \(message.content.prefix(50))...")
                }
            }
        }
        
        return memory
    }
    
    func clearHistory() {
        sessionHistory.removeAll()
        currentSession = nil
        userDefaults.removeObject(forKey: sessionKey)
        userDefaults.removeObject(forKey: currentSessionKey)
    }
}

// MARK: - Enhanced AdvisorSession with Memory
extension AdvisorSession {
    var conversationSummary: String {
        let userQuestions = messages.filter { $0.role == .user }.count
        let advisorResponses = messages.filter { $0.role == .advisor }.count
        let highConfidenceResponses = messages.filter { $0.role == .advisor && ($0.confidence ?? 0) > 0.7 }.count
        
        return """
        Session started: \(DateFormatter.shortDateFormatter.string(from: startTime))
        Questions asked: \(userQuestions)
        Responses given: \(advisorResponses)
        High confidence responses: \(highConfidenceResponses)
        Active recommendations: \(recommendations.count)
        """
    }
    
    var lastActivity: Date {
        return [startTime] + messages.map { $0.timestamp } + recommendations.map { $0.timestamp }
            .max() ?? startTime
    }
    
    var isActive: Bool {
        return Date().timeIntervalSince(lastActivity) < 3600 // Active if used within last hour
    }
}

extension DateFormatter {
    static let shortDateFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .short
        formatter.timeStyle = .short
        return formatter
    }()
}
