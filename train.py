import argparse
from pathlib import Path
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.model import build_lstm_model
from src.preprocess import SEQUENCE_LENGTH, build_vocabulary, collect_tokens, create_sequences, save_metadata

def args_parser():
    p=argparse.ArgumentParser(description="Train an LSTM music-generation model.")
    p.add_argument("--data",default="data/midi")
    p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--sequence-length",type=int,default=SEQUENCE_LENGTH)
    p.add_argument("--model-out",default="models/music_lstm.keras")
    p.add_argument("--metadata-out",default="models/metadata.json")
    return p.parse_args()

def main():
    a=args_parser()
    if a.epochs<=0 or a.batch_size<=0: raise SystemExit("Epochs and batch size must be positive.")
    tokens=collect_tokens(a.data)
    token_to_int,_=build_vocabulary(tokens)
    x,y=create_sequences(tokens,token_to_int,a.sequence_length)
    model_path=Path(a.model_out); model_path.parent.mkdir(parents=True,exist_ok=True)
    save_metadata(a.metadata_out,tokens,token_to_int,a.sequence_length)
    model=build_lstm_model(a.sequence_length,len(token_to_int))
    model.summary()
    callbacks=[
        ModelCheckpoint(str(model_path),monitor="loss",save_best_only=True,verbose=1),
        EarlyStopping(monitor="loss",patience=8,restore_best_weights=True,verbose=1),
    ]
    model.fit(x,y,epochs=a.epochs,batch_size=a.batch_size,callbacks=callbacks,shuffle=True)
    model.save(model_path)
    print(f"Saved model to {model_path}")
    print(f"Saved metadata to {a.metadata_out}")

if __name__=="__main__":
    main()
