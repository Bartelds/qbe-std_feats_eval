#!/bin/bash

pip3 install espnet parallel_wavegan espnet_model_zoo --user

echo -n "Type the keywords to search: "
read keywords
python3 scripts/text_to_wav.py --keywords $keywords
echo -n "Specify the w2v2 model to use: "
read model
echo -n "Specify the output layer: "
read layer
python3 scripts/wav_to_w2v2-feats.py --dataset tts-generated --stage transformer --layer $layer --model $model
echo -n "Specify the model certainty (default = 0.9): "
read certainty
python3 scripts/feats_to_dtw_return.py _all_ tts-generated --top_n 10 --certainty $certainty
rm -rf data/raw/datasets/tts-generated/queries/
