//
//  ActivityInputView.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import SwiftUI

struct ActivityInputView: View {
    @StateObject private var activityManager = ActivityManager()
    @State private var selectedDate = Date()
    @State private var showingAddActivity = false
    
    var body: some View {
        NavigationView {
            VStack {
                // Date Picker
                DatePicker("Select Date", selection: $selectedDate, displayedComponents: .date)
                    .datePickerStyle(CompactDatePickerStyle())
                    .padding()
                
                // Activity Summary
                if let todaysActivity = activityManager.getActivity(for: selectedDate) {
                    ActivitySummaryView(activity: todaysActivity)
                        .padding(.horizontal)
                } else {
                    Text("No activities recorded for this date")
                        .foregroundColor(.secondary)
                        .padding()
                }
                
                Spacer()
                
                // Add Activity Button
                Button(action: {
                    showingAddActivity = true
                }) {
                    HStack {
                        Image(systemName: "plus.circle.fill")
                        Text("Add Activity")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .padding()
                    .background(Color.blue)
                    .cornerRadius(10)
                }
                .padding(.bottom, 30)
            }
            .navigationTitle("Daily Activities")
        }
    }
}

struct ActivitySummaryView: View {
    let activity: BabyActivity
    
    var body: some View {
        VStack(alignment: .leading, spacing: 15) {
            Text("Today's Activities")
                .font(.headline)
                .fontWeight(.semibold)
            
            HStack {
                ActivityStatView(
                    icon: "fork.knife",
                    count: activity.feedingTimes.count,
                    label: "Feedings",
                    color: .orange
                )
                
                Spacer()
                
                ActivityStatView(
                    icon: "diaper",
                    count: activity.diaperChanges.count,
                    label: "Diaper Changes",
                    color: .green
                )
                
                Spacer()
                
                ActivityStatView(
                    icon: "bed.double.fill",
                    count: activity.sleepSessions.count,
                    label: "Sleep Sessions",
                    color: .purple
                )
            }
        }
        .padding()
        .background(Color(.systemGray6))
        .cornerRadius(12)
    }
}

struct ActivityStatView: View {
    let icon: String
    let count: Int
    let label: String
    let color: Color
    
    var body: some View {
        VStack(spacing: 5) {
            Image(systemName: icon)
                .font(.title2)
                .foregroundColor(color)
            
            Text("\(count)")
                .font(.headline)
                .fontWeight(.bold)
            
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
        }
    }
}

#Preview {
    ActivityInputView()
}
