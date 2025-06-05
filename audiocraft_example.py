import sys

sys.path.append("./audiocraft")

from audiocraft.models import AudioGen
from audiocraft.data.audio import audio_write

model = AudioGen.get_pretrained("facebook/audiogen_medium")
model.set_generation_params(duration=8)  # generate 8 seconds.

descriptions = ["happy rock", "energetic EDM"]

wav = model.generate(descriptions)  # generates 2 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f"{idx}", one_wav.cpu(), model.sample_rate, strategy="loudness")
