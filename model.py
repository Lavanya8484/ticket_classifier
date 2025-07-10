# import pandas as pd
# import numpy as np
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder
# import pickle
#
#
# def train_model():
#     print("📥 Loading dataset...")
#     df = pd.read_csv("data.csv")
#
#     if df.empty:
#         print("❌ Dataset is empty. Check data.csv!")
#         return
#
#     print("✅ Dataset loaded. First few rows:")
#     print(df.head())
#
#     # Prepare data
#     texts = df["ticket"].astype(str).tolist()
#     labels = df["module"].astype(str).tolist()
#
#     # Encode labels
#     label_encoder = LabelEncoder()
#     labels_encoded = label_encoder.fit_transform(labels)
#
#     # Save label encoder
#     with open("label_encoder.pkl", "wb") as f:
#         pickle.dump(label_encoder, f)
#     print("✅ Saved label_encoder.pkl")
#
#     # Tokenize text
#     tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
#     tokenizer.fit_on_texts(texts)
#     sequences = tokenizer.texts_to_sequences(texts)
#     padded = pad_sequences(sequences, maxlen=20, padding='post')
#
#     # Save tokenizer
#     with open("tokenizer.pkl", "wb") as f:
#         pickle.dump(tokenizer, f)
#     print("✅ Saved tokenizer.pkl")
#
#     # Build model
#     model = Sequential()
#     model.add(Embedding(input_dim=5000, output_dim=64, input_length=20))
#     model.add(LSTM(64))
#     model.add(Dense(32, activation="relu"))
#     model.add(Dense(len(set(labels_encoded)), activation="softmax"))
#
#     model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
#
#     print("🚀 Training model...")
#     model.fit(padded, np.array(labels_encoded), epochs=5, verbose=1)
#
#     # Save model
#     model.save("ticket_model.h5")
#     print("✅ Model saved as ticket_model.h5")
#
#
# if __name__ == "__main__":
#     train_model()


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

def train_model():
    print("📥 Loading dataset...")
    df = pd.read_csv("data.csv")
    df.dropna(inplace=True)

    texts = df["ticket"].astype(str).tolist()
    labels = df["module"].astype(str).tolist()

    print("🔠 Tokenizing...")
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, padding='post', maxlen=20)

    print("🏷️ Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(padded, encoded_labels, test_size=0.2, random_state=42)

    print("🧠 Building model...")
    model = Sequential([
        Embedding(1000, 16, input_length=20),
        GlobalAveragePooling1D(),
        Dense(24, activation='relu'),
        Dense(len(set(encoded_labels)), activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test), verbose=1)

    print("✅ Model saved as ticket_model.keras")
    model.save("ticket_model.keras")

    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("✅ Done training!")

if __name__ == "__main__":
    train_model()
