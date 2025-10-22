//
//  AudioRecorder.swift
//  CryBaby
//
//  Created by AI Assistant on 2025.
//

import AVFoundation
import SwiftUI

class AudioRecorder: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var isProcessing = false
    @Published var statusText = "Tap to record your baby's cry"
    
    private var audioRecorder: AVAudioRecorder?
    private var audioSession: AVAudioSession = AVAudioSession.sharedInstance()
    private var recordingURL: URL?
    
    override init() {
        super.init()
        setupAudioSession()
    }
    
    private func setupAudioSession() {
        do {
            try audioSession.setCategory(.playAndRecord, mode: .default, options: [.defaultToSpeaker])
            try audioSession.setActive(true)
            
            // Request microphone permission
            audioSession.requestRecordPermission { [weak self] allowed in
                DispatchQueue.main.async {
                    if !allowed {
                        self?.statusText = "Microphone permission required"
                    }
                }
            }
        } catch {
            statusText = "Audio setup failed"
        }
    }
    
    func startRecording() {
        guard !isRecording else { return }
        guard audioSession.recordPermission == .granted else {
            statusText = "Microphone permission required"
            return
        }
        
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let audioFilename = documentsPath.appendingPathComponent("cry_recording_\(Date().timeIntervalSince1970).wav")
        
        let settings: [String: Any] = [
            AVFormatIDKey: Int(kAudioFormatLinearPCM),
            AVSampleRateKey: 16000, // Match backend expectation
            AVNumberOfChannelsKey: 1, // Mono
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false
        ]
        
        do {
            audioRecorder = try AVAudioRecorder(url: audioFilename, settings: settings)
            audioRecorder?.delegate = self
            audioRecorder?.isMeteringEnabled = true
            audioRecorder?.record()
            
            isRecording = true
            recordingURL = audioFilename
            statusText = "Recording... Tap to stop"
        } catch {
            statusText = "Failed to start recording"
        }
    }
    
    func stopRecording(completion: @escaping (URL) -> Void) {
        guard isRecording else { return }
        
        audioRecorder?.stop()
        isRecording = false
        isProcessing = true
        statusText = "Processing recording..."
        
        // Small delay to ensure recording is complete
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) {
            if let url = self.recordingURL {
                completion(url)
            }
            self.isProcessing = false
            self.statusText = "Tap to record again"
        }
    }
}

extension AudioRecorder: AVAudioRecorderDelegate {
    func audioRecorderDidFinishRecording(_ recorder: AVAudioRecorder, successfully flag: Bool) {
        if !flag {
            DispatchQueue.main.async {
                self.statusText = "Recording failed"
                self.isProcessing = false
            }
        }
    }
    
    func audioRecorderEncodeErrorDidOccur(_ recorder: AVAudioRecorder, error: Error?) {
        DispatchQueue.main.async {
            self.statusText = "Recording error occurred"
            self.isRecording = false
            self.isProcessing = false
        }
    }
}
