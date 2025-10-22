//
//  AIAdviceView.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import SwiftUI

// MARK: - Models

struct AIAdvice {
    let patterns: [String]
    let recommendation: String
    let dataPoints: Int
    let confidence: Double?
    let timestamp: Date
}

struct AIAdviceView: View {
    @State private var advice: AIAdvice?
    @State private var isLoading = false
    @StateObject private var adviceService: AIAdviceService = AIAdviceService()
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header
                VStack(spacing: 10) {
                    Image(systemName: "brain.head.profile")
                        .font(.system(size: 50))
                        .foregroundColor(.purple)
                    
                    Text("AI Parenting Advice")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Get personalized insights based on your baby's patterns")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 20)
                
                Spacer()
                
                if isLoading {
                    VStack {
                        ProgressView()
                            .scaleEffect(1.5)
                        Text("Analyzing your data...")
                            .font(.subheadline)
                            .foregroundColor(.secondary)
                            .padding(.top)
                    }
                } else if let advice = advice {
                    AdviceCardView(advice: advice)
                        .padding(.horizontal)
                } else {
                    VStack(spacing: 15) {
                        Image(systemName: "lightbulb")
                            .font(.system(size: 40))
                            .foregroundColor(.yellow)
                        
                        Text("Tap below to get personalized advice based on your baby's activity patterns and cry analysis.")
                            .font(.body)
                            .foregroundColor(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal)
                    }
                }
                
                Spacer()
                
                // Get Advice Button
                Button(action: getAdvice) {
                    HStack {
                        Image(systemName: "sparkles")
                        Text("Get AI Advice")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.purple)
                    .cornerRadius(10)
                }
                .disabled(isLoading)
                .padding(.bottom, 30)
            }
            .navigationBarHidden(true)
        }
    }
    
    private func getAdvice() {
        isLoading = true
        
        // Get actual data from ActivityManager and CryAnalyzer
        // For now, we'll use nil data and let the service handle fallback
        adviceService.generateAdvice(
            cryAnalysisData: nil as [CryAnalysisResult]?, // TODO: Get from CryAnalyzer results
            activityData: nil as [BabyActivity]?     // TODO: Get from ActivityManager
        ) { result in
            DispatchQueue.main.async {
                self.isLoading = false
                self.advice = result
            }
        }
    }
}


struct AdviceCardView: View {
    let advice: AIAdvice
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: "brain.head.profile")
                    .font(.title2)
                    .foregroundColor(.purple)
                
                VStack(alignment: .leading) {
                    Text("Personalized Advice")
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text("Based on \(advice.dataPoints) data points")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            
            Text(advice.recommendation)
                .font(.body)
                .lineLimit(nil)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

#Preview {
    AIAdviceView()
}
