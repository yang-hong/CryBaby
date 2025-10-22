//
//  ActivityModels.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import Foundation

struct BabyActivity: Identifiable, Codable {
    let id: UUID
    let date: Date
    let feedingTimes: [FeedingActivity]
    let diaperChanges: [DiaperActivity]
    let sleepSessions: [SleepActivity]
    let cryingEpisodes: [CryingEpisode]
    
    init(date: Date = Date(), feedingTimes: [FeedingActivity] = [], diaperChanges: [DiaperActivity] = [], sleepSessions: [SleepActivity] = [], cryingEpisodes: [CryingEpisode] = []) {
        self.id = UUID()
        self.date = date
        self.feedingTimes = feedingTimes
        self.diaperChanges = diaperChanges
        self.sleepSessions = sleepSessions
        self.cryingEpisodes = cryingEpisodes
    }
}

struct FeedingActivity: Identifiable, Codable {
    let id: UUID
    let time: Date
    let type: FeedingType
    let amount: Double? // in ml or oz
    let duration: TimeInterval? // in minutes
    let notes: String?
    
    init(time: Date, type: FeedingType, amount: Double? = nil, duration: TimeInterval? = nil, notes: String? = nil) {
        self.id = UUID()
        self.time = time
        self.type = type
        self.amount = amount
        self.duration = duration
        self.notes = notes
    }
}

enum FeedingType: String, CaseIterable, Codable {
    case breast = "breast"
    case bottle = "bottle"
    case solid = "solid"
    case snack = "snack"
}

struct DiaperActivity: Identifiable, Codable {
    let id: UUID
    let time: Date
    let type: DiaperType
    let notes: String?
    
    init(time: Date, type: DiaperType, notes: String? = nil) {
        self.id = UUID()
        self.time = time
        self.type = type
        self.notes = notes
    }
}

enum DiaperType: String, CaseIterable, Codable {
    case wet = "wet"
    case dirty = "dirty"
    case both = "both"
    case dry = "dry"
}

struct SleepActivity: Identifiable, Codable {
    let id: UUID
    let startTime: Date
    let endTime: Date?
    let duration: TimeInterval? // calculated if endTime is set
    let sleepType: SleepType
    let notes: String?
    
    init(startTime: Date, sleepType: SleepType, notes: String? = nil) {
        self.id = UUID()
        self.startTime = startTime
        self.endTime = nil
        self.duration = nil
        self.sleepType = sleepType
        self.notes = notes
    }
}

enum SleepType: String, CaseIterable, Codable {
    case nap = "nap"
    case night = "night"
    case bedtime = "bedtime"
}

struct CryingEpisode: Identifiable, Codable {
    let id: UUID
    let startTime: Date
    let endTime: Date?
    let duration: TimeInterval? // calculated if endTime is set
    let cryAnalysis: CryAnalysisResult?
    let notes: String?
    
    init(startTime: Date, cryAnalysis: CryAnalysisResult? = nil, notes: String? = nil) {
        self.id = UUID()
        self.startTime = startTime
        self.endTime = nil
        self.duration = nil
        self.cryAnalysis = cryAnalysis
        self.notes = notes
    }
}

// MARK: - Baby Profile for AI Advisor
struct BabyProfile: Identifiable, Codable {
    let id: UUID
    let name: String?
    let birthDate: Date
    var ageInDays: Int {
        Calendar.current.dateComponents([.day], from: birthDate, to: Date()).day ?? 0
    }
    let gender: BabyGender?
    let birthWeight: Double? // in grams
    let currentWeight: Double? // in grams (updated regularly)
    let feedingMethod: PrimaryFeedingMethod
    let sleepSchedule: SleepSchedule?
    let growthMilestones: [GrowthMilestone]
    let allergies: [String]
    let medicalConditions: [String]
    let specialNotes: String?
    
    init(name: String? = nil, birthDate: Date, gender: BabyGender? = nil, birthWeight: Double? = nil, currentWeight: Double? = nil, feedingMethod: PrimaryFeedingMethod = .unknown, sleepSchedule: SleepSchedule? = nil, growthMilestones: [GrowthMilestone] = [], allergies: [String] = [], medicalConditions: [String] = [], specialNotes: String? = nil) {
        self.id = UUID()
        self.name = name
        self.birthDate = birthDate
        self.gender = gender
        self.birthWeight = birthWeight
        self.currentWeight = currentWeight
        self.feedingMethod = feedingMethod
        self.sleepSchedule = sleepSchedule
        self.growthMilestones = growthMilestones
        self.allergies = allergies
        self.medicalConditions = medicalConditions
        self.specialNotes = specialNotes
    }
}

enum BabyGender: String, CaseIterable, Codable {
    case male = "male"
    case female = "female"
    case other = "other"
}

enum PrimaryFeedingMethod: String, CaseIterable, Codable {
    case breastOnly = "breast_only"
    case bottleOnly = "bottle_only"
    case mixed = "mixed"
    case solid = "solid"
    case unknown = "unknown"
}

struct SleepSchedule: Codable {
    let bedtime: String // "19:30"
    let wakeTime: String // "07:00"
    let napTimes: [String] // ["09:00", "13:00", "16:00"]
    let totalSleepHours: Double
}

struct GrowthMilestone: Identifiable, Codable {
    let id: UUID
    let date: Date
    let type: MilestoneType
    let description: String
    let notes: String?
    
    init(date: Date, type: MilestoneType, description: String, notes: String? = nil) {
        self.id = UUID()
        self.date = date
        self.type = type
        self.description = description
        self.notes = notes
    }
}

enum MilestoneType: String, CaseIterable, Codable {
    case feeding = "feeding"
    case sleep = "sleep"
    case motor = "motor"
    case social = "social"
    case cognitive = "cognitive"
    case weight = "weight"
    case height = "height"
    case other = "other"
}
