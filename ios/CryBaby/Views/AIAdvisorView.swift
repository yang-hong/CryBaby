//
//  AIAdvisorView.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import SwiftUI

struct AIAdvisorView: View {
    @StateObject private var advisorService = AIAdvisorService()
    @State private var currentQuestion = ""
    @State private var showingQuestionInput = false
    @State private var babyProfile: BabyProfile?
    
    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Header
                VStack(spacing: 15) {
                    Image(systemName: "person.crop.circle.badge.questionmark")
                        .font(.system(size: 50))
                        .foregroundColor(.blue)
                    
                    Text("AI Baby Advisor")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Ask questions about your baby's patterns, feeding, sleep, and more")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }
                .padding(.top, 20)
                .padding(.bottom, 30)
                
                // Chat Interface
                if let session = advisorService.currentSession {
                    ChatView(session: session, advisorService: advisorService)
                } else {
                    // Initial setup
                    InitialSetupView(advisorService: advisorService, babyProfile: $babyProfile)
                }
                
                // Question Input Area
                VStack(spacing: 10) {
                    if showingQuestionInput {
                        HStack {
                            TextField("Ask a question about your baby...", text: $currentQuestion, axis: .vertical)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .lineLimit(2...4)
                            
                            Button(action: askQuestion) {
                                Image(systemName: "paperplane.fill")
                                    .foregroundColor(.blue)
                            }
                            .disabled(currentQuestion.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || advisorService.isLoading)
                        }
                        .padding(.horizontal)
                    }
                    
                    // Toggle Question Input Button
                    Button(action: { 
                        showingQuestionInput.toggle()
                        if showingQuestionInput {
                            // Focus on text field
                        }
                    }) {
                        HStack {
                            Image(systemName: showingQuestionInput ? "keyboard" : "plus.circle.fill")
                            Text(showingQuestionInput ? "Hide Question" : "Ask a Question")
                        }
                        .font(.headline)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(10)
                    }
                    .padding(.horizontal)
                    .padding(.bottom, 20)
                }
            }
            .navigationBarHidden(true)
            .onAppear {
                setupInitialSession()
            }
        }
    }
    
    private func setupInitialSession() {
        // Get baby profile and recent data
        // For now, create a basic context
        let context = AdvisorContext(
            babyProfile: babyProfile,
            recentCryAnalyses: [], // TODO: Get from CryAnalyzer
            recentActivities: [], // TODO: Get from ActivityManager
            timeRange: .week,
            parentQuestion: nil
        )
        
        advisorService.startSession(context: context)
        showingQuestionInput = true
    }
    
    private func askQuestion() {
        let question = currentQuestion.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !question.isEmpty else { return }
        
        advisorService.askQuestion(question) { response in
            if response != nil {
                currentQuestion = ""
                showingQuestionInput = false
            }
        }
    }
}

struct InitialSetupView: View {
    @ObservedObject var advisorService: AIAdvisorService
    @Binding var babyProfile: BabyProfile?
    
    var body: some View {
        VStack(spacing: 20) {
            Image(systemName: "person.crop.circle.badge.plus")
                .font(.system(size: 60))
                .foregroundColor(.gray)
            
            Text("Set up your baby's profile for personalized advice")
                .font(.headline)
                .multilineTextAlignment(.center)
            
            Button("Set Up Baby Profile") {
                // TODO: Navigate to profile setup
                // For now, create a sample profile
                babyProfile = BabyProfile(
                    name: "Baby",
                    birthDate: Calendar.current.date(byAdding: .day, value: -30, to: Date()) ?? Date(),
                    gender: .other,
                    feedingMethod: .mixed
                )
            }
            .font(.headline)
            .foregroundColor(.white)
            .padding()
            .background(Color.blue)
            .cornerRadius(10)
        }
        .padding()
    }
}

struct ChatView: View {
    let session: AdvisorSession
    @ObservedObject var advisorService: AIAdvisorService
    
    var body: some View {
        ScrollView {
            LazyVStack(spacing: 15) {
                // Recommendations Section
                if !session.recommendations.isEmpty {
                    RecommendationsSection(recommendations: session.recommendations)
                }
                
                // Chat Messages
                ForEach(session.messages) { message in
                    MessageBubble(message: message)
                }
                
                // Loading indicator
                if advisorService.isLoading {
                    HStack {
                        ProgressView()
                            .scaleEffect(0.8)
                        Text("Thinking...")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                }
                
                // Error message
                if let error = advisorService.errorMessage {
                    Text("Error: \(error)")
                        .font(.caption)
                        .foregroundColor(.red)
                        .padding()
                }
            }
            .padding()
        }
    }
}

struct RecommendationsSection: View {
    let recommendations: [AdvisorRecommendation]
    
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Image(systemName: "lightbulb.fill")
                    .foregroundColor(.yellow)
                Text("Recommendations")
                    .font(.headline)
                    .fontWeight(.semibold)
                Spacer()
            }
            
            ForEach(recommendations.prefix(3)) { recommendation in
                RecommendationCard(recommendation: recommendation)
            }
        }
        .padding()
        .background(Color.yellow.opacity(0.1))
        .cornerRadius(12)
    }
}

struct RecommendationCard: View {
    let recommendation: AdvisorRecommendation
    
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(recommendation.title)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                
                Spacer()
                
                PriorityBadge(priority: recommendation.priority)
            }
            
            Text(recommendation.description)
                .font(.caption)
                .foregroundColor(.secondary)
            
            if !recommendation.sources.isEmpty {
                Text("Sources: \(recommendation.sources.joined(separator: ", "))")
                    .font(.caption2)
                    .foregroundColor(.blue)
            }
        }
        .padding(.vertical, 8)
    }
}

struct PriorityBadge: View {
    let priority: Priority
    
    var body: some View {
        Text(priority.rawValue.capitalized)
            .font(.caption2)
            .fontWeight(.semibold)
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(priorityColor.opacity(0.2))
            .foregroundColor(priorityColor)
            .cornerRadius(8)
    }
    
    private var priorityColor: Color {
        switch priority {
        case .urgent: return .red
        case .high: return .orange
        case .medium: return .blue
        case .low: return .green
        }
    }
}

struct MessageBubble: View {
    let message: AdvisorMessage
    
    var body: some View {
        HStack {
            if message.role == .user {
                Spacer()
                
                VStack(alignment: .trailing, spacing: 4) {
                    Text(message.content)
                        .font(.body)
                        .foregroundColor(.white)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(16, corners: [.topLeft, .topRight, .bottomLeft])
                    
                    if let confidence = message.confidence {
                        Text("Confidence: \(Int(confidence * 100))%")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                }
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(alignment: .top) {
                        Image(systemName: "brain.head.profile")
                            .foregroundColor(.purple)
                            .font(.title3)
                        
                        VStack(alignment: .leading, spacing: 8) {
                            Text(message.content)
                                .font(.body)
                                .foregroundColor(.primary)
                            
                            if let sources = message.sources, !sources.isEmpty {
                                Text("Sources: \(sources.joined(separator: ", "))")
                                    .font(.caption)
                                    .foregroundColor(.blue)
                            }
                        }
                        
                        if let confidence = message.confidence {
                            VStack {
                                Text("\(Int(confidence * 100))%")
                                    .font(.caption2)
                                    .fontWeight(.semibold)
                                Text("confident")
                                    .font(.caption2)
                            }
                            .foregroundColor(.secondary)
                        }
                        
                        Spacer()
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(16, corners: [.topLeft, .topRight, .bottomRight])
                
                Spacer()
            }
        }
    }
}

// Extension for custom corner radius
extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCorner(radius: radius, corners: corners))
    }
}

struct RoundedCorner: Shape {
    var radius: CGFloat = .infinity
    var corners: UIRectCorner = .allCorners

    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: corners,
            cornerRadii: CGSize(width: radius, height: radius)
        )
        return Path(path.cgPath)
    }
}

#Preview {
    AIAdvisorView()
}
