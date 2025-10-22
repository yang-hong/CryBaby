//
//  CryAnalysisModels.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import Foundation
import SwiftUI

enum CryType: String, CaseIterable, Codable {
    case hungry = "hungry"
    case uncomfortable = "uncomfortable"
    case unknown = "unknown"
}

struct CryAnalysisResult: Codable {
    let cryType: CryType
    let confidence: Double
    let probabilities: [String: Double]
    let timestamp: Date
    let audioDuration: TimeInterval
    
    var suggestion: String? {
        switch cryType {
        case .hungry:
            return "Your baby might be hungry. Consider checking when they last fed and offering a feeding."
        case .uncomfortable:
            return "Your baby might be uncomfortable. Check their diaper, clothing, temperature, or if they need to be repositioned."
        case .unknown:
            return "The cry pattern isn't clearly recognizable. Consider checking all basic needs - hunger, comfort, sleep, or stimulation."
        }
    }
    
    var iconName: String {
        switch cryType {
        case .hungry:
            return "fork.knife"
        case .uncomfortable:
            return "exclamationmark.triangle"
        case .unknown:
            return "questionmark.circle"
        }
    }
    
    var color: Color {
        switch cryType {
        case .hungry:
            return .orange
        case .uncomfortable:
            return .red
        case .unknown:
            return .gray
        }
    }
}

struct CryAnalysisRequest: Codable {
    let audioData: Data // Base64 encoded audio
    let duration: TimeInterval
    let timestamp: Date
}

struct CryAnalysisResponse: Codable {
    let success: Bool
    let result: CryAnalysisResult?
    let error: String?
}

struct FeedbackRequest: Codable {
    let analysisResult: CryAnalysisResult
    let isCorrect: Bool
    let actualCryType: CryType?
    let notes: String?
}

struct FeedbackResponse: Codable {
    let success: Bool
    let message: String?
    let error: String?
}
