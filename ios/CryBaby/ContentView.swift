//
//  ContentView.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import SwiftUI

struct ContentView: View {
    @State private var selectedTab = 0
    
    var body: some View {
        TabView(selection: $selectedTab) {
            CryAnalysisView()
                .tabItem {
                    Image(systemName: "waveform")
                    Text("Cry Analysis")
                }
                .tag(0)
            
            ActivityInputView()
                .tabItem {
                    Image(systemName: "calendar.badge.plus")
                    Text("Daily Activities")
                }
                .tag(1)
            
            AIAdviceView()
                .tabItem {
                    Image(systemName: "brain.head.profile")
                    Text("AI Advice")
                }
                .tag(2)
            
            SettingsView()
                .tabItem {
                    Image(systemName: "gearshape")
                    Text("Settings")
                }
                .tag(3)
        }
        .accentColor(.blue)
    }
}

#Preview {
    ContentView()
}
