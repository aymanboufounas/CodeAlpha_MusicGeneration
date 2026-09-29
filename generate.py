import argparse, random
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocess import load_metadata, tokens_to_midi

def sample_with_temperature(probabilities,temperature=1.0):
    probabilities=np.asarray(probabilities,dtype=np.float64)
    probabilities=np.clip(probabilities,1e-9,1.0)
    logits=np.log(probabilities)/temperature
    exp=np.exp(logits-np.max(logits))
    distribution=exp/np.sum(exp)
    return int(np.random.choice(len(distribution),p=distribution))

def generate_tokens(model,seed_pattern,int_to_token,vocab_size,length=200,temperature=0.9):
    pattern=list(seed_pattern); generated=[]
    for _ in range(length):
        x=np.asarray(pattern,dtype=np.float32).reshape((1,len(pattern),1))/float(vocab_size)
        prediction=model.predict(x,verbose=0)[0]
        index=sample_with_temperature(prediction,temperature)
        generated.append(int_to_token[index])
        pattern.append(index); pattern=pattern[1:]
    return generated

def main():
    p=argparse.ArgumentParser(description="Generate a new MIDI composition.")
    p.add_argument("--model",default="models/music_lstm.keras")
    p.add_argument("--metadata",default="models/metadata.json")
    p.add_argument("--length",type=int,default=200)
    p.add_argument("--temperature",type=float,default=0.9)
    p.add_argument("--output",default="output/generated_music.mid")
    p.add_argument("--seed",type=int,default=None)
    a=p.parse_args()
    if a.length<=0 or a.temperature<=0: raise SystemExit("Length and temperature must be positive.")
    if a.seed is not None:
        random.seed(a.seed); np.random.seed(a.seed)
    payload,token_to_int,int_to_token=load_metadata(a.metadata)
    tokens=payload["tokens"]; seq=int(payload["sequence_length"]); vocab=int(payload["vocab_size"])
    encoded=[token_to_int[t] for t in tokens]
    start=random.randint(0,len(encoded)-seq-1)
    seed_pattern=encoded[start:start+seq]
    model=load_model(a.model)
    generated=generate_tokens(model,seed_pattern,int_to_token,vocab,a.length,a.temperature)
    out=tokens_to_midi(generated,a.output)
    print(f"Generated {len(generated)} tokens -> {out}")

if __name__=="__main__":
    main()
