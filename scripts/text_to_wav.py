from espnet2.bin.tts_inference import Text2Speech
import scipy.io.wavfile
import argparse
import os
import librosa
import soundfile
from distutils.dir_util import copy_tree

parser = argparse.ArgumentParser(description='example: python text_to_wav.py bragel')
parser.add_argument('--keywords', nargs='+', default=[],
                    help='audio to be generated from list of keywords')
parser.add_argument('--datasets_dir', default='data/raw/datasets/tts-generated/queries/', help = "directory for raw datasets and labels files")

args = parser.parse_args()

os.makedirs(args.datasets_dir, exist_ok=True)
print("Copying Gronings reference files...")
copy_tree("data/raw/datasets/gos-kdl/references", "data/raw/datasets/tts-generated/references/")

tts = Text2Speech.from_pretrained(
  model_tag="https://huggingface.co/ahnafsamin/FastSpeech2-gronings/resolve/main/tts_train_fastspeech2_raw_char_tacotron_train.loss.ave.zip",
  vocoder_tag="parallel_wavegan/ljspeech_parallel_wavegan.v3",
  speed_control_alpha=1.0
)

for keyword in args.keywords:
    print(f"Synthesizing {keyword}...")

    # synthesis
    speech = tts(keyword)["wav"]
    scipy.io.wavfile.write(args.datasets_dir + keyword + ".wav", tts.fs, speech.view(-1).cpu().numpy())

    # resample
    y, sr = librosa.load(args.datasets_dir + keyword + ".wav", sr=16_000)
    soundfile.write(args.datasets_dir + keyword + ".wav", y, sr, "PCM_16")
