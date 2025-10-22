//
//  ActivityManager.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import Foundation

class ActivityManager: ObservableObject {
    private var activities: [Date: BabyActivity] = [:]
    
    func getActivity(for date: Date) -> BabyActivity? {
        let dateKey = Calendar.current.startOfDay(for: date)
        return activities[dateKey]
    }
    
    func addActivity<T>(_ activity: T, for date: Date) {
        let dateKey = Calendar.current.startOfDay(for: date)
        
        var currentActivity = activities[dateKey] ?? BabyActivity(date: dateKey)
        
        // This is a simplified version - in a real app, you'd handle different activity types properly
        if let feeding = activity as? FeedingActivity {
            currentActivity = BabyActivity(
                date: dateKey,
                feedingTimes: (currentActivity.feedingTimes + [feeding]),
                diaperChanges: currentActivity.diaperChanges,
                sleepSessions: currentActivity.sleepSessions,
                cryingEpisodes: currentActivity.cryingEpisodes
            )
        }
        
        activities[dateKey] = currentActivity
    }
}
