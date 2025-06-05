import sys

sys.path.append("./audiocraft")

from audiocraft.audiocraft.models import AudioGen
from audiocraft.audiocraft.data.audio import audio_write

model = AudioGen.get_pretrained("facebook/audiogen-medium")
model.set_generation_params(duration=8)  # generate 8 seconds.
model.set_generation_params(temperature=1.0)

descriptions = ["Water trickle puddle"]

wav = model.generate(descriptions)  # generates 2 samples.

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f"{idx}", one_wav.cpu(), model.sample_rate, strategy="loudness")
