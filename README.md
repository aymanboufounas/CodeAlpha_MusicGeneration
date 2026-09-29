# CodeAlpha AI Music Generation

An **LSTM-based AI music generator** created for **CodeAlpha Artificial Intelligence Internship — Task 3**.

The project reads MIDI files, extracts musical notes and chords using `music21`, turns them into numerical sequences, trains a TensorFlow/Keras **LSTM neural network**, generates new musical sequences, and exports the result as a standard MIDI file.

## Task 3 Requirements Covered

| CodeAlpha requirement | Implementation |
|---|---|
| Collect MIDI music data | `data/midi/` |
| Preprocess music into note sequences | `music21` + `src/preprocess.py` |
| Build a deep-learning model | TensorFlow/Keras LSTM |
| Train the AI on music patterns | `train.py` |
| Generate new sequences | `generate.py` |
| Save generated music as MIDI | `tokens_to_midi()` |

## AI Pipeline

```text
MIDI Dataset
     ↓
music21 Parser
     ↓
Notes + Chords
     ↓
Token Vocabulary
     ↓
Training Sequences
     ↓
LSTM Neural Network
     ↓
Next-Token Prediction
     ↓
Generated Sequence
     ↓
generated_music.mid
```

## Project Structure

```text
CodeAlpha_MusicGeneration/
├── train.py
├── generate.py
├── prepare_demo_data.py
├── requirements.txt
├── README.md
├── PROJECT_INFO.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── model.py
│   └── preprocess.py
├── data/
│   └── midi/
│       └── README.md
├── models/
│   └── README.md
├── output/
│   └── README.md
└── tests/
    └── test_preprocess.py
```

## Installation

Python 3.10 or 3.11 is recommended.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Demo Dataset

To test the entire pipeline without finding MIDI files first:

```bash
python prepare_demo_data.py
```

This generates several small synthetic MIDI files in `data/midi/`.

For your final internship demonstration, add a larger real MIDI collection that you are allowed to use. A larger and stylistically consistent dataset usually produces better generated music.

## Train the LSTM

```bash
python train.py
```

Default configuration:

```text
Sequence length: 50
Epochs: 50
Batch size: 64
Model: models/music_lstm.keras
Metadata: models/metadata.json
```

For a quick test:

```bash
python train.py --epochs 5 --batch-size 32
```

For longer training:

```bash
python train.py --epochs 100 --batch-size 64
```

The neural network contains:

- LSTM layer
- Dropout
- Batch normalization
- Second LSTM layer
- Dense layer
- Softmax output layer
- Adam optimizer
- Early stopping
- Model checkpointing

## Generate New Music

After training:

```bash
python generate.py
```

The result is saved to:

```text
output/generated_music.mid
```

Generate more notes:

```bash
python generate.py --length 500
```

## Control Creativity

Lower temperature makes the model more conservative:

```bash
python generate.py --temperature 0.6
```

Balanced:

```bash
python generate.py --temperature 0.9
```

More experimental:

```bash
python generate.py --temperature 1.2
```

## Use Your Own MIDI Dataset

Put MIDI files in:

```text
data/midi/
```

Then train again:

```bash
python train.py
```

## How the AI Learns

The model performs **next-token prediction**. Given a musical sequence, it learns a probability distribution for the next note or chord. During generation, each sampled prediction becomes part of the next input sequence, allowing the model to create a new composition.

## Play the Result

`generated_music.mid` can be opened in MuseScore, VLC, FL Studio, Ableton Live, GarageBand, or LMMS.

## Tests

```bash
pytest
```

## AI Contribution

This project performs actual neural-network training. It does **not** call ChatGPT or a music-generation API to create the music. The model learns musical patterns from the supplied MIDI dataset and generates new note/chord sequences from those learned patterns.

## Author

**Ayman Boufounas**

Built for the CodeAlpha Artificial Intelligence Internship.
