import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# 1. Load a .wav file (replace with your file path)
path = "data/raw/hungry/hung_sugianto_minta_susu_11.wav"

# Load at original sample rate
wav_orig, sr_orig = librosa.load(path, sr=None)  # sr=None = preserve native rate
print("Original sample rate:", sr_orig, "Hz")
print("Original length in samples:", len(wav_orig))

# 2. Resample to 16 kHz
wav_16k = librosa.resample(wav_orig, orig_sr=sr_orig, target_sr=16000)
print("Resampled to 16k length in samples:", len(wav_16k))

# 3. Plot the first 0.05 seconds (zoom in to see waveform shape)
time_orig = librosa.times_like(wav_orig, sr=sr_orig)
time_16k = librosa.times_like(wav_16k, sr=16000)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(time_orig[:int(sr_orig*0.05)], wav_orig[:int(sr_orig*0.05)])
plt.title(f"Original waveform ({sr_orig} Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")

plt.subplot(1, 2, 2)
plt.plot(time_16k[:int(16000*0.05)], wav_16k[:int(16000*0.05)])
plt.title("Resampled waveform (16k Hz)")
plt.xlabel("Time (s)")

plt.tight_layout()
plt.show()

