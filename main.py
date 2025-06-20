import librosa
import soundfile as sf
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

def load_model():
    print("Loading pre-trained Wav2Vec2 model...")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")
    return processor, model

def load_audio(file_path, sampling_rate=16000):
    print(f"Loading audio file from {file_path}...")
    audio, sr = librosa.load(file_path, sr=sampling_rate)
    return audio, sr

def transcribe_audio(audio, processor, model, sampling_rate=16000):
    print("Preparing input for the model...")
    inputs = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    print("Running inference...")
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])
    return transcription

if __name__ == "__main__":
    audio_path = "D:\FREELANCE_PROJECTS\deep_learning_speech_reco\Data\harvard.wav"

    processor, model = load_model()

    audio, sampling_rate = load_audio(audio_path)

    transcription = transcribe_audio(audio, processor, model)
    print("\nTranscription:")
    print(transcription)

    with open(r"D:\FREELANCE_PROJECTS\deep_learning_speech_reco\Audio-Files\transcription.txt", "w") as f:
        f.write(transcription)
    print("Transcription saved to 'transcription.txt'")
