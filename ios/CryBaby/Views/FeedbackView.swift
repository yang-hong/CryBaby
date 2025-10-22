//
//  FeedbackView.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import SwiftUI

struct FeedbackView: View {
    let result: CryAnalysisResult?
    let onSubmit: (FeedbackRequest) -> Void
    
    @Environment(\.dismiss) private var dismiss
    @State private var isCorrect = true
    @State private var actualCryType: CryType = .hungry
    @State private var notes = ""
    
    var body: some View {
        NavigationView {
            VStack(spacing: 25) {
                // Header
                VStack(spacing: 15) {
                    if let result = result {
                        HStack {
                            Image(systemName: result.iconName)
                                .font(.title2)
                                .foregroundColor(result.color)
                            
                            VStack(alignment: .leading) {
                                Text("Analysis Result")
                                    .font(.headline)
                                
                                Text(result.cryType.rawValue.capitalized)
                                    .font(.subheadline)
                                    .foregroundColor(.secondary)
                            }
                            
                            Spacer()
                            
                            Text("\(Int(result.confidence * 100))%")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        .padding()
                        .background(Color(.systemGray6))
                        .cornerRadius(10)
                    }
                    
                    Text("Was this prediction correct?")
                        .font(.title2)
                        .fontWeight(.semibold)
                }
                .padding(.top)
                
                // Correctness Toggle
                VStack(spacing: 15) {
                    HStack {
                        Button(action: { isCorrect = true }) {
                            HStack {
                                Image(systemName: isCorrect ? "checkmark.circle.fill" : "circle")
                                    .foregroundColor(.green)
                                Text("Yes, this was correct")
                            }
                        }
                        .foregroundColor(.primary)
                        
                        Spacer()
                        
                        Button(action: { isCorrect = false }) {
                            HStack {
                                Image(systemName: !isCorrect ? "checkmark.circle.fill" : "circle")
                                    .foregroundColor(.red)
                                Text("No, this was wrong")
                            }
                        }
                        .foregroundColor(.primary)
                    }
                    .font(.body)
                }
                .padding(.horizontal)
                
                Spacer()
                
                // Submit Button
                Button(action: submitFeedback) {
                    Text("Submit Feedback")
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(10)
                }
                .padding(.horizontal)
                .padding(.bottom, 30)
            }
            .navigationTitle("Feedback")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Cancel") {
                        dismiss()
                    }
                }
            }
        }
    }
    
    private func submitFeedback() {
        guard let result = result else {
            dismiss()
            return
        }
        
        let feedback = FeedbackRequest(
            analysisResult: result,
            isCorrect: isCorrect,
            actualCryType: isCorrect ? nil : actualCryType,
            notes: notes.isEmpty ? nil : notes
        )
        
        onSubmit(feedback)
        dismiss()
    }
}

#Preview {
    FeedbackView(result: CryAnalysisResult(
        cryType: .hungry,
        confidence: 0.85,
        probabilities: ["hungry": 0.85, "uncomfortable": 0.15],
        timestamp: Date(),
        audioDuration: 5.2
    )) { _ in }
}
