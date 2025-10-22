//
//  CryAnalyzer.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import Foundation
import UIKit
import AVFoundation

class CryAnalyzer: ObservableObject {
    private let backendURL = "http://127.0.0.1:8000" // Local backend for development
    
    func analyzeCry(audioURL: URL, completion: @escaping (CryAnalysisResult) -> Void) {
        guard (try? Data(contentsOf: audioURL)) != nil else {
            // Return error result
            let errorResult = CryAnalysisResult(
                cryType: .unknown,
                confidence: 0.0,
                probabilities: [:],
                timestamp: Date(),
                audioDuration: 0
            )
            completion(errorResult)
            return
        }
        
        // Get audio duration
        let audioDuration = getAudioDuration(from: audioURL)
        
        sendAnalysisRequest(audioURL: audioURL, duration: audioDuration) { result in
            completion(result)
        }
    }
    
    private func getAudioDuration(from url: URL) -> TimeInterval {
        do {
            let audioPlayer = try AVAudioPlayer(contentsOf: url)
            return audioPlayer.duration
        } catch {
            return 0
        }
    }
    
    private func sendAnalysisRequest(audioURL: URL, duration: TimeInterval, completion: @escaping (CryAnalysisResult) -> Void) {
        guard let url = URL(string: "\(backendURL)/api/v1/cry/predict") else {
            let errorResult = CryAnalysisResult(
                cryType: .unknown,
                confidence: 0.0,
                probabilities: [:],
                timestamp: Date(),
                audioDuration: duration
            )
            completion(errorResult)
            return
        }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        
        // Create multipart form data
        let boundary = "Boundary-\(UUID().uuidString)"
        request.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        
        var body = Data()
        
        // Add audio file
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append("Content-Disposition: form-data; name=\"audio\"; filename=\"recording.wav\"\r\n".data(using: .utf8)!)
        body.append("Content-Type: audio/wav\r\n\r\n".data(using: .utf8)!)
        
        do {
            let audioData = try Data(contentsOf: audioURL)
            body.append(audioData)
            body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
            request.httpBody = body
        } catch {
            let errorResult = CryAnalysisResult(
                cryType: .unknown,
                confidence: 0.0,
                probabilities: [:],
                timestamp: Date(),
                audioDuration: duration
            )
            completion(errorResult)
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            DispatchQueue.main.async {
                print("🔍 DEBUG: Starting network request...")
                
                if let error = error {
                    print("❌ Network error: \(error)")
                    let errorResult = CryAnalysisResult(
                        cryType: .unknown,
                        confidence: 0.0,
                        probabilities: [:],
                        timestamp: Date(),
                        audioDuration: duration
                    )
                    completion(errorResult)
                    return
                }
                
                guard let data = data else {
                    print("❌ No data received from backend")
                    let errorResult = CryAnalysisResult(
                        cryType: .unknown,
                        confidence: 0.0,
                        probabilities: [:],
                        timestamp: Date(),
                        audioDuration: duration
                    )
                    completion(errorResult)
                    return
                }
                
                print("✅ Received data from backend: \(data.count) bytes")
                
                // Print raw response for debugging
                if let rawResponse = String(data: data, encoding: .utf8) {
                    print("📄 Raw backend response: \(rawResponse)")
                }
                
                do {
                    let response = try JSONSerialization.jsonObject(with: data) as? [String: Any]
                    print("✅ Parsed JSON response: \(response ?? [:])")
                    
                    guard let responseDict = response,
                          let predictedLabel = responseDict["predicted_label"] as? String,
                          let probabilities = responseDict["probabilities"] as? [String: Double],
                          let maxProbability = responseDict["max_probability"] as? Double else {
                        let errorResult = CryAnalysisResult(
                            cryType: .unknown,
                            confidence: 0.0,
                            probabilities: [:],
                            timestamp: Date(),
                            audioDuration: duration
                        )
                        completion(errorResult)
                        return
                    }
                    
                    let cryType = CryType(rawValue: predictedLabel) ?? .unknown
                    let result = CryAnalysisResult(
                        cryType: cryType,
                        confidence: maxProbability,
                        probabilities: probabilities,
                        timestamp: Date(),
                        audioDuration: duration
                    )
                    
                    print("🎯 Final result: \(cryType.rawValue) with confidence: \(maxProbability) (probabilities: \(probabilities))")
                    completion(result)
                    
                } catch {
                    print("JSON decode error: \(error)")
                    let errorResult = CryAnalysisResult(
                        cryType: .unknown,
                        confidence: 0.0,
                        probabilities: [:],
                        timestamp: Date(),
                        audioDuration: duration
                    )
                    completion(errorResult)
                }
            }
        }.resume()
    }
    
    func submitFeedback(result: CryAnalysisResult, feedback: FeedbackRequest) {
        guard let url = URL(string: "\(backendURL)/api/v1/cry/feedback") else { return }
        
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        do {
            request.httpBody = try JSONEncoder().encode(feedback)
        } catch {
            print("Failed to encode feedback: \(error)")
            return
        }
        
        URLSession.shared.dataTask(with: request) { data, response, error in
            if let error = error {
                print("Feedback submission error: \(error)")
            }
            // Handle response if needed
        }.resume()
    }
}
