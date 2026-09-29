from pathlib import Path
from music21 import chord, instrument, note, stream, tempo

OUTPUT_DIR=Path("data/midi")

def create_demo_song(filename,root,pattern,repeats=12):
    score=stream.Stream()
    score.append(tempo.MetronomeMark(number=105))
    root_midi=note.Note(root).pitch.midi
    for repeat in range(repeats):
        for degree in pattern:
            pitch=root_midi+degree
            if repeat%4==3 and degree in {0,5,7}:
                c=chord.Chord([pitch,pitch+4,pitch+7])
                c.quarterLength=0.5
                c.storedInstrument=instrument.Piano()
                score.append(c)
            else:
                n=note.Note(pitch)
                n.quarterLength=0.5
                n.storedInstrument=instrument.Piano()
                score.append(n)
    path=OUTPUT_DIR/filename
    score.write("midi",fp=str(path))
    print(f"Created {path}")

def main():
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    demos=[
        ("demo_c_major.mid","C4",[0,2,4,5,7,9,11,12]),
        ("demo_a_minor.mid","A3",[0,2,3,5,7,8,10,12]),
        ("demo_g_major.mid","G3",[0,2,4,5,7,9,11,12]),
        ("demo_f_major.mid","F3",[0,2,4,5,7,9,10,12]),
        ("demo_d_minor.mid","D4",[0,2,3,5,7,8,10,12]),
    ]
    for filename,root,pattern in demos:
        create_demo_song(filename,root,pattern)
    print("Demo MIDI dataset created. Add real MIDI files for stronger training.")

if __name__=="__main__":
    main()
