//
//  SettingsView.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("confidenceThreshold") private var confidenceThreshold: Double = 0.6
    @AppStorage("backendURL") private var backendURL: String = "http://127.0.0.1:8000"
    
    var body: some View {
        NavigationView {
            List {
                Section("Analysis Settings") {
                    VStack(alignment: .leading) {
                        Text("Confidence Threshold")
                        Text("Minimum confidence level for predictions (currently: \(Int(confidenceThreshold * 100))%)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        
                        Slider(value: $confidenceThreshold, in: 0.3...0.9, step: 0.1)
                    }
                    .padding(.vertical, 5)
                }
                
                Section("Backend Configuration") {
                    VStack(alignment: .leading, spacing: 5) {
                        Text("Backend URL")
                        TextField("http://127.0.0.1:8000", text: $backendURL)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                        Text("URL of your CryBaby backend service")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                
                Section("About") {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }
                }
            }
            .navigationTitle("Settings")
        }
    }
}

#Preview {
    SettingsView()
}
