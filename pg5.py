import tkinter as tk
from tkinter import messagebox
import speech_recognition as sr
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout

# Read data
data = pd.read_csv("train1.txt", sep=';')
data.columns = ["Text", "Emotions"]

texts = data["Text"].tolist()
labels = data["Emotions"].tolist()

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
max_length = max([len(seq) for seq in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Encode the string labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# One-hot encode the labels
num_classes = len(set(labels))
one_hot_labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(padded_sequences,
                                                one_hot_labels,
                                                test_size=0.2,
                                                random_state=42)

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1,
                    output_dim=128, input_length=max_length))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(units=num_classes, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(xtrain, ytrain, epochs=10 , batch_size=32, validation_data=(xtest, ytest))

# Function to detect emotion from text
def detect_text_emotion():
    input_text = text_entry.get()
    if input_text.strip() == "":
        messagebox.showwarning("Warning", "Please enter some text.")
        return
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    prediction = model.predict(padded_input_sequence)
    predicted_label = np.argmax(prediction[0])
    predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]
    messagebox.showinfo("Emotion Detection Result", f"The detected emotion is: {predicted_emotion}")

# Function to detect emotion from speech
def detect_speech_emotion():
    captured_text = capture_speech()
    if captured_text.strip() == "":
        messagebox.showwarning("Warning", "No speech detected.")
        return
    input_sequence = tokenizer.texts_to_sequences([captured_text])
    padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
    prediction = model.predict(padded_input_sequence)
    predicted_label = np.argmax(prediction[0])
    predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]
    messagebox.showinfo("Emotion Detection Result", f"The detected emotion is: {predicted_emotion}")

# Function to capture speech input
def capture_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("\nListening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
    try:
        print("Recognizing...")
        # Convert speech to text
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as e:
        print("Error fetching results; {0}".format(e))

# Main Tkinter window
root = tk.Tk()
root.title("Emotion Analysis")

# Styling
root.geometry("600x400")
root.configure(bg="#f0f0f0")

# Colors
bg_color = "#f0f0f0"
button_bg_color = "red"  # Light blue
button_bg_color1 = "green" 
button_fg_color = "white"
label_bg_color = "#f0f0f0"
label_fg_color = "black"
entry_bg_color = "white"
entry_fg_color = "black"

# Heading
heading_label = tk.Label(root, text="Understanding Emotions via Text and Speech Analysis", bg=label_bg_color, fg=label_fg_color, font=("Arial", 20))
heading_label.pack(pady=10)

# Text Entry
text_label = tk.Label(root, text="Enter Text:", bg=label_bg_color, fg=label_fg_color, font=("Arial", 12))
text_label.pack(pady=5)
text_entry = tk.Entry(root, width=60, bg=entry_bg_color, fg=entry_fg_color, font=("Arial", 10))
text_entry.pack(pady=5)

# Buttons
text_button = tk.Button(root, text="Detect Text Emotion", command=detect_text_emotion, bg=button_bg_color, fg=button_fg_color, font=("Arial", 12))
text_button.pack(pady=10)

speech_button = tk.Button(root, text="Detect Speech Emotion", command=detect_speech_emotion, bg=button_bg_color1, fg=button_fg_color, font=("Arial", 12))
speech_button.pack(pady=10)



root.mainloop()

y_pred_prob = model.predict(xtest)

# Convert predicted probabilities to labels
y_pred = np.argmax(y_pred_prob, axis=1)

# Convert one-hot encoded true labels to integer labels
y_true = np.argmax(ytest, axis=1)

# Calculate accuracy
accuracy = np.mean(y_pred == y_true)
print("Accuracy:", accuracy)


#He's feeling sad because his pet died.-sadness
#She had a very happy childhood. - joy
#He got angry when he found out about their plans.- angry
#"I was really surprised that she won!" - suprise
#The rabbit looked scared. - fear
#i wonder how it feels to be loved by someone you love -love