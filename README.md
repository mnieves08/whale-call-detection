# Whale Call Detection
**Michael Angelo Nieves | Princeton AI4ALL project**

A machine learning project exploring how underwater audio can be used to identify North Atlantic right whale calls. The workflow compares a neural network trained on audio waveforms with a convolutional neural network trained on mel spectrograms.

## Project overview
The code loads labeled audio clips, visualizes their waveforms and spectrograms, trains binary classifiers, and examines classification errors. This repository presents the code from my Whale Detection v3 Colab notebook, with small execution fixes made for the portfolio export.

**Tools:** Python, PyTorch, torchaudio, librosa, NumPy, pandas, scikit-learn, Matplotlib, and Seaborn.

## Workflow
1. Load audio clips and their whale / non-whale labels.
2. Explore examples through audio playback, waveforms, and spectrograms.
3. Train a fully connected baseline on raw waveforms.
4. Convert audio into mel spectrograms and split the data into 60% training, 20% validation, and 20% testing.
5. Augment training spectrograms with time shifts, frequency shifts, and noise.
6. Train a CNN and evaluate predictions using accuracy and a confusion matrix.

The exported CNN has **three convolutional layers**, followed by a fully connected classifier. The final cells also define a modified ResNet-18 and an ensemble as further experiments; those cells do not train or evaluate the experimental models.

## Files
| File | Purpose |
| --- | --- |
| `whale_detection_v3.py` | Notebook code cells in their original order, with minor execution fixes |
| `requirements.txt` | Python dependencies used by the project |
| `.gitignore` | Excludes downloaded data, model weights, and local environment files |

## Setup and use
This is an exploratory notebook export with plots, audio playback, and an interactive prompt. An IPython/Jupyter environment is recommended for viewing its outputs.

```bash
git clone https://github.com/mnieves08/whale-call-detection.git
cd whale-call-detection
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ipython
```

Then run this inside IPython:
```python
%run whale_detection_v3.py
```

The script automatically downloads the project dataset archive using `gdown` and extracts it into the working directory. Dataset access requires the original Google Drive file to remain available. Audio and dataset archives are not included in this repository.

Expected layout:
```text
whale-detection-challenge/
  data/
    audio/
    labels.csv
```
The CSV uses `clip_name` and `label` columns. Labels are 0 for non-whale and 1 for whale. The baseline assumes 4,000-value audio inputs; the spectrogram workflow expects consistent clip dimensions.

To use a dataset already on your computer, skip notebook code cell 4 (the download/extraction block) and set `WHALE_DATA_DIR` to the folder containing `audio/` and `labels.csv`. Setting the variable alone does not disable the download.

## Evaluation and limitations
The script computes test accuracy and confusion matrices when run. This repository does not include saved training outputs or trained weights, so no numerical performance claim is made here. End-to-end training has not been rerun for this upload.

An experimental prompt offers to replace labels with confident model predictions. Answer **no** to retain the original labels. Model confidence alone does not establish that a label is incorrect; any relabeling should be independently reviewed.

The later ResNet and ensemble cells instantiate new models and overwrite the `model` variable. They are experimental definitions, not trained final models. Augmentation uses unseeded randomness, so results can vary across runs.

## Export notes
Notebook cell markers are preserved. Small fixes handle single-example error indexing, rewind an in-memory audio buffer before playback, retain the batch dimension for CNN loss calculation, and initialize the ensemble CNNs with spectrogram shapes.

Python syntax compilation passed. Dependency installation, dataset access, and full training were not validated for this upload; dependency versions are not locked.

## Context
This project was developed during Princeton AI4ALL. It is shared here as an educational portfolio project documenting work with audio preprocessing, neural networks, and model evaluation.
