import wave
import os
import json
from alg.text_comparison import calculate_similarity

import sys

from vosk import Model, KaldiRecognizer, SetLogLevel

audio_file_path = os.path.join(".", "test.wav")

model_path = os.path.join(".", "vosk-model-small-en-us-0.15")

print(model_path)

audio = wave.open(audio_file_path, "rb")

model = Model(model_path=model_path)

rec = KaldiRecognizer(model, audio.getframerate())
rec.SetWords(True)
rec.SetPartialWords(True)


while True:
    data = audio.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        continue
    else:
        continue


result = rec.FinalResult()
to_json = json.loads(result)
transcription = to_json["text"]


audio_description = "one zero zero zero one nine oh two i no zero one eight zero three"

similarity = calculate_similarity(audio_description, transcription)

to_percent = int(similarity * 100)

print(to_percent)
