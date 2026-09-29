from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
from music21 import chord, converter, instrument, note, stream

SEQUENCE_LENGTH = 50

def extract_tokens_from_midi(midi_path: str | Path) -> List[str]:
    score = converter.parse(str(midi_path))
    try:
        parts = instrument.partitionByInstrument(score)
        elements = parts.parts[0].recurse() if parts and parts.parts else score.flat.notes
    except Exception:
        elements = score.flat.notes

    tokens=[]
    for element in elements:
        if isinstance(element, note.Note):
            tokens.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            tokens.append(".".join(str(n) for n in element.normalOrder))
    return tokens

def collect_tokens(midi_dir: str | Path) -> List[str]:
    midi_dir=Path(midi_dir)
    midi_files=sorted(list(midi_dir.rglob("*.mid"))+list(midi_dir.rglob("*.midi")))
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found in {midi_dir}. Run prepare_demo_data.py or add MIDI files.")
    all_tokens=[]
    for f in midi_files:
        try:
            tokens=extract_tokens_from_midi(f)
            all_tokens.extend(tokens)
            print(f"[OK] {f.name}: {len(tokens)} tokens")
        except Exception as exc:
            print(f"[SKIP] {f.name}: {exc}")
    if len(all_tokens)<=SEQUENCE_LENGTH:
        raise ValueError("Not enough musical tokens. Add more MIDI files.")
    return all_tokens

def build_vocabulary(tokens: Sequence[str]) -> Tuple[Dict[str,int],Dict[int,str]]:
    vocab=sorted(set(tokens))
    token_to_int={t:i for i,t in enumerate(vocab)}
    int_to_token={i:t for t,i in token_to_int.items()}
    return token_to_int,int_to_token

def create_sequences(tokens: Sequence[str], token_to_int: Dict[str,int], sequence_length: int=SEQUENCE_LENGTH):
    network_input=[]
    network_output=[]
    for i in range(len(tokens)-sequence_length):
        sequence_in=tokens[i:i+sequence_length]
        sequence_out=tokens[i+sequence_length]
        network_input.append([token_to_int[t] for t in sequence_in])
        network_output.append(token_to_int[sequence_out])
    if not network_input:
        raise ValueError("No training sequences could be created.")
    x=np.asarray(network_input,dtype=np.float32).reshape((-1,sequence_length,1))
    y=np.asarray(network_output,dtype=np.int32)
    x=x/float(len(token_to_int))
    return x,y

def save_metadata(path, tokens, token_to_int, sequence_length=SEQUENCE_LENGTH):
    payload={
        "tokens":list(tokens),
        "token_to_int":token_to_int,
        "sequence_length":sequence_length,
        "vocab_size":len(token_to_int),
    }
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")

def load_metadata(path):
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    token_to_int={str(k):int(v) for k,v in payload["token_to_int"].items()}
    int_to_token={v:k for k,v in token_to_int.items()}
    return payload,token_to_int,int_to_token

def tokens_to_midi(prediction_output: Sequence[str], output_path: str | Path, step: float=0.5):
    offset=0.0
    output_notes=[]
    for pattern in prediction_output:
        if "." in pattern and all(part.isdigit() for part in pattern.split(".")):
            notes=[]
            for current_note in pattern.split("."):
                n=note.Note(int(current_note)); n.storedInstrument=instrument.Piano(); notes.append(n)
            c=chord.Chord(notes); c.offset=offset; output_notes.append(c)
        else:
            n=note.Note(pattern); n.offset=offset; n.storedInstrument=instrument.Piano(); output_notes.append(n)
        offset+=step
    midi_stream=stream.Stream(output_notes)
    p=Path(output_path); p.parent.mkdir(parents=True,exist_ok=True)
    midi_stream.write("midi",fp=str(p))
    return p
