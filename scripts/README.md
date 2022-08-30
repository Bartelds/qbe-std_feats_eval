# Replication/usage instructions

In this document, we walk through running our analyses on the Gronings dataset (gos-kdl), which has been released on Zenodo ([https://zenodo.org/record/4634878](https://zenodo.org/record/4634878)).
The datasets expected by our scripts have the following form:

```
gos-kdl/
|-- labels.csv
|-- queries/
|--|-- ED_aapmoal.wav
|--|-- ED_achter.wav
...
|-- references/
|--|-- OV-aapmoal-verschillend-mor-aapmoal-prachteg-van-kleur.wav
|--|-- RB-de-gruinteboer-staait-mit-n-blaauw-schoet-veur-achter-de-teunbaank.wav
...
```

## System requirements

The instructions here are working as of 2021-03-24, tested on a virtual instance with 24 CPU cores and 64 GB of RAM running Ubuntu 20.04 LTS, Docker Engine 20.10.5, and Docker Compose 1.28.4. See [Docker Desktop for Mac](https://docs.docker.com/docker-for-mac/install/) or [Docker Desktop for Windows](https://docs.docker.com/docker-for-windows/install/) if you're on one of these platforms. The Docker images have all required interpreters and package dependencies pre-installed (Python, NumPy; R, dplyr; Perl, etc.), so if you choose not to use them, you may need to spend some time configuring your local environment to work with these scripts (e.g. STDEval requires Perl no newer than 5.18.4, and the Python scripts require at least Pandas 1.2.1; see `requirements.txt` and `scripts/Docker/Dockerfile`).

### Docker installation script

The script was used to install docker and docker-compose on a fresh instance of Ubuntu 20.04 LTS, based on [DigitalOcean instructions](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04).

```bash
sudo apt update && \
sudo apt-get -y install apt-transport-https ca-certificates curl software-properties-common && \
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - && \
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu focal stable" && \
sudo apt update && \
apt-cache policy docker-ce && \
sudo apt-get -y install docker-ce && \
sudo curl -L "https://github.com/docker/compose/releases/download/1.28.4/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose && \
sudo chmod +x /usr/local/bin/docker-compose
```

## 1. Setup

### 1.1 Clone qbe-std_feats_eval repository

```bash
git clone https://github.com/fauxneticien/qbe-std_feats_eval.git
cd qbe-std_feats_eval
```

### 1.2 Set up `gos-kdl` dataset locally

```bash
# Download gos-kdl.zip into qbe-std_feats_eval/tmp directory
wget https://zenodo.org/record/4634878/files/gos-kdl.zip -P tmp/

## Install unzip if necessary
# apt-get install unzip

# Create directory data/raw/datasets/gos-kdl
mkdir -p data/raw/datasets/gos-kdl

# Unzip into directory
unzip tmp/gos-kdl.zip -d data/raw/datasets/gos-kdl
``` 

#### 1.3 Pull docker image(s)

```bash
# For extracting wav2vec 2.0 features and running evaluation scripts
docker pull fauxneticien/qbe-std_feats_eval

# For extracting MFCC and BNF features (optional)
docker pull fauxneticien/shennong
```

## 2. Feature extraction

All extraction routines create a `queries.pickle` and `references.pickle`, which are Pandas data frames with two columns. As illustrated below, the first column is the name of the .wav file from which features were extracted using some feature extraction routine (e.g. MFCC) and the second column is a NumPy array of the features for that wav file created by the routine.
By default, these files are placed in `data/interim/features/{DATASET}/{FEATURE}/`, for example, `data/interim/features/gos-kdl/mfcc/queries.pickle`.


| filename | features |
|----------|----------|
|   ED_aapmoal    | [[16.485865, -11.592721, -14.900574, 20.032818...
|   ED_achter  | [[11.749482, -9.294043, -6.118123, -7.8093295,...

### 2.1 MFCC and BNF features

We couldn't get the Shennong dependencies to play nice with the ones needed for the wav2vec 2.0 image, so if you want to extract MFCC and BNF features (as we did for our baselines), you'll have to do it in the `fauxneticien/shennong` image.

```bash
# Start docker container according to 'shennong' config
# specified in the docker-compose.yml file
docker-compose run --rm shennong

# Activate conda environment inside the container 
conda activate shennong

# Extract MFCC and BNF features using wav_to_shennong-feats.py
#
# For help, run: python scripts/wav_to_shennong-feats.py -h

python scripts/wav_to_shennong-feats.py \
    _all_ \
    gos-kdl

# Exit the shennong container
exit
```

### 2.2 wav2vec 2.0 features

We use Hugging Face to help fetch the wav2vec 2.0 models to use for feature extraction. The model repo paths (e.g. `facebook/wav2vec2-base`) can be found in the `wav_to_w2v2-feats.py` script (note: for reproducibility of the analyses, the `wav2vec2-large` and `wav2vec2-large-xlsr-53` have specific model versions):

```python
KNOWN_MODELS = {
    # Pre-trained
    'wav2vec2-base': 'facebook/wav2vec2-base',
    'wav2vec2-large': {'name' : 'facebook/wav2vec2-large', 'revision' : '85c73b1a7c1ee154fd7b06634ca7f42321db94db' },
    # March 11, 2021 version: https://huggingface.co/facebook/wav2vec2-large/commit/85c73b1a7c1ee154fd7b06634ca7f42321db94db
    'wav2vec2-large-lv60': 'facebook/wav2vec2-large-lv60',
    'wav2vec2-large-xlsr-53': {'name' : 'facebook/wav2vec2-large-xlsr-53', 'revision' : '8e86806e53a4df405405f5c854682c785ae271da' },
    # May 6, 2021 version: https://huggingface.co/facebook/wav2vec2-large-xlsr-53/commit/8e86806e53a4df405405f5c854682c785ae271da
    
    # Fine-tuned
    'wav2vec2-base-960h': 'facebook/wav2vec2-base-960h',
    'wav2vec2-large-960h': 'facebook/wav2vec2-large-960h',
    'wav2vec2-large-960h-lv60': 'facebook/wav2vec2-large-960h-lv60',
    'wav2vec2-large-960h-lv60-self': 'facebook/wav2vec2-large-960h-lv60-self',
    'wav2vec2-large-xlsr-53-english': 'jonatasgrosman/wav2vec2-large-xlsr-53-english',
    'wav2vec2-large-xlsr-53-tamil': 'manandey/wav2vec2-large-xlsr-tamil'
}
```

To extract features using one of these models run:

```bash
# Start docker container according to 'dev' config
# specified in the docker-compose.yml file
docker-compose run --rm dev

# Extract features from all stages/layers (encoder, quantizer, transformer 1-24)
# of wav2vec 2.0 model (wav2vec2-large, revision: 85c73b)
#
# For help, run: python scripts/wav_to_w2v2-feats.py -h

python scripts/wav_to_w2v2-xlsr-feats.py \
   --dataset gos-kdl \
   --stage _all_ \
   --layer _all_ \
   --model wav2vec2-large
```



### 2.3 Fetch features from Zenodo (optional)

Extracted features from all 10 datasets have been uploaded to Zenodo (see [https://zenodo.org/record/4635493](https://zenodo.org/record/4635493) and [https://zenodo.org/record/4635438](https://zenodo.org/record/4635438)). Features for any of the datasets can be downloaded and extracted using (for example):

```bash
# Get link from 'Download' button on https://zenodo.org/record/4635438
wget https://zenodo.org/record/4635438/files/wbp-jk.zip -P tmp/

# Make data/interim/features directory if necessary
# mkdir -p data/interim/features

unzip tmp/wbp-jk.zip -d data/interim/features
```

## 3. DTW search

Using features based on a given feature extraction method, we add a corresponding prediction for how likely the query occurs in the reference using an iterative Dynamic Time Warping search, where a window the size of the query is moved along the length of the reference and a DTW-based distance is calculated at each iteration. The final score is `1 - min(dists)`, and is appended to the labels table (example from `data/processed/dtw/mfcc_gos-kdl.csv`). Based on the specified model certainty, likely matching query-reference pairs are returned to the user.

| query |        reference      | label | prediction |
|-------|-----------------------|-------|------------|
| ED_aapmoal | OV-aapmoal-verschillend-mor-aapmoal-prachteg-van-kleur |   1   |    0.9179029429624552    |
| ED_aapmoal | RB-de-gruinteboer-staait-mit-n-blaauw-schoet-veur-achter-de-teunbaank        |   0   |    0.8537919659746750    |
|  ED_achter  | OV-aapmoal-verschillend-mor-aapmoal-prachteg-van-kleur |   0   |    0.8851410753164508    |
|  ED_achter  | RB-de-gruinteboer-staait-mit-n-blaauw-schoet-veur-achter-de-teunbaank        |   1   |    0.8891430558060820     |

```
# If you're not already inside the 'dev' container:
# docker-compose run --rm dev

# Run DTW search for each feature extraction method on gos-kdl
#
# For help, run: python scripts/feats_to_dtw_return.py -h

python scripts/feats_to_dtw_return.py \
    _all_ \
    gos-kdl \
    --top_n 10 \
    --certainty 0.9
```

## 4. DTW search with TTS queries (Gronings (gos-kdl) only)

Make sure to set up the gos-kdl dataset locally, as we use those references with the TTS-generated queries.

```
# If you're not already inside the 'dev' container:
# docker-compose run --rm dev

bash run-tts-qbe-std.sh
```
