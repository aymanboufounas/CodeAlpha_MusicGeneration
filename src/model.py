from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_lstm_model(sequence_length:int,vocab_size:int,learning_rate:float=0.001):
    model=Sequential([
        LSTM(256,input_shape=(sequence_length,1),return_sequences=True),
        Dropout(0.30),
        BatchNormalization(),
        LSTM(256,return_sequences=False),
        Dropout(0.30),
        BatchNormalization(),
        Dense(256,activation="relu"),
        Dropout(0.20),
        Dense(vocab_size,activation="softmax"),
    ])
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=["accuracy"],
    )
    return model
