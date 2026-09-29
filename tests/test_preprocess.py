from src.preprocess import build_vocabulary, create_sequences

def test_build_vocabulary():
    tokens=["C4","E4","G4","C4"]
    token_to_int,int_to_token=build_vocabulary(tokens)
    assert len(token_to_int)==3
    assert int_to_token[token_to_int["C4"]]=="C4"

def test_create_sequences():
    tokens=["C4","D4","E4","F4","G4","A4"]*3
    token_to_int,_=build_vocabulary(tokens)
    x,y=create_sequences(tokens,token_to_int,sequence_length=5)
    assert x.shape[1:]==(5,1)
    assert len(x)==len(y)
    assert len(x)>0
