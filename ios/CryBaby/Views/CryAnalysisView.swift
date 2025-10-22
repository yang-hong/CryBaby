//
//  CryAnalysisView.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import SwiftUI
import AVFoundation

struct CryAnalysisView: View {
    @StateObject private var audioRecorder = AudioRecorder()
    @StateObject private var cryAnalyzer = CryAnalyzer()
    @State private var isRecording = false
    @State private var analysisResult: CryAnalysisResult?
    @State private var showingFeedback = false
    
    var body: some View {
        NavigationView {
            VStack(spacing: 30) {
                // Header
                VStack(spacing: 10) {
                    Image(systemName: "waveform.circle.fill")
                        .font(.system(size: 60))
                        .foregroundColor(.blue)
                    
                    Text("Cry Analysis")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Record your baby's cry to get insights about their needs")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(.top, 20)
                
                Spacer()
                
                // Recording Section
                VStack(spacing: 20) {
                    // Record Button
                    Button(action: {
                        if isRecording {
                            stopRecording()
                        } else {
                            startRecording()
                        }
                    }) {
                        ZStack {
                            Circle()
                                .fill(isRecording ? Color.red : Color.blue)
                                .frame(width: 120, height: 120)
                            
                            Image(systemName: isRecording ? "stop.fill" : "mic.fill")
                                .font(.system(size: 40))
                                .foregroundColor(.white)
                        }
                    }
                    .disabled(audioRecorder.isProcessing)
                    
                    // Status Text
                    Text(audioRecorder.statusText)
                        .font(.headline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
                
                // Analysis Result
                if let result = analysisResult {
                    AnalysisResultCard(result: result) {
                        showingFeedback = true
                    }
                    .padding(.horizontal)
                }
                
                Spacer()
            }
            .padding()
            .navigationBarHidden(true)
            .sheet(isPresented: $showingFeedback) {
                FeedbackView(result: analysisResult) { feedback in
                    cryAnalyzer.submitFeedback(result: analysisResult!, feedback: feedback)
                }
            }
        }
    }
    
    private func startRecording() {
        audioRecorder.startRecording()
        isRecording = true
        analysisResult = nil
    }
    
    private func stopRecording() {
        audioRecorder.stopRecording { url in
            isRecording = false
            cryAnalyzer.analyzeCry(audioURL: url) { result in
                DispatchQueue.main.async {
                    self.analysisResult = result
                }
            }
        }
    }
}

struct AnalysisResultCard: View {
    let result: CryAnalysisResult
    let onFeedbackRequested: () -> Void
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            HStack {
                Image(systemName: result.iconName)
                    .font(.title2)
                    .foregroundColor(result.color)
                
                VStack(alignment: .leading) {
                    Text(result.cryType.rawValue.capitalized)
                        .font(.headline)
                        .fontWeight(.semibold)
                    
                    Text("Confidence: \(Int(result.confidence * 100))%")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                
                Spacer()
            }
            
            if let suggestion = result.suggestion {
                Text(suggestion)
                    .font(.body)
                    .foregroundColor(.secondary)
                    .padding(.top, 5)
            }
            
            Button("Was this correct?") {
                onFeedbackRequested()
            }
            .font(.subheadline)
            .foregroundColor(.blue)
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

#Preview {
    CryAnalysisView()
}
