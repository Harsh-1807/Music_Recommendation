import os
import numpy as np
from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model
import librosa
import librosa.display
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# --- Flask Setup ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Genre Mapping (Update as per your model) ---
GENRE_LABELS = [
    'rock', 'pop', 'classical', 'jazz', 'blues', 
    'hiphop', 'country', 'reggae', 'metal', 'disco'
]

# --- Load Trained Model ---
MODEL_PATH = 'models/killer_combo_model.h5'
model = load_model(MODEL_PATH)
INPUT_SHAPE = (224, 224)  # Update this if different

# --- Convert Audio to Spectrogram ---
def audio_to_spectrogram(audio_path, target_size=(224, 224)):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)

    fig = plt.figure(figsize=(2.24, 2.24), dpi=100)
    plt.axis('off')
    librosa.display.specshow(S_DB, sr=sr, cmap='magma')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    img = Image.open(buf).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Predict Genre ---
def predict_genre(audio_path):
    img_input = audio_to_spectrogram(audio_path)
    prediction = model.predict(img_input)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return GENRE_LABELS[predicted_class]

# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            audio_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(audio_path)
            genre = predict_genre(audio_path)
            return render_template_string(PAGE_HTML, result=genre)
    return render_template_string(PAGE_HTML, result=None)

# --- Simple HTML Template ---
PAGE_HTML = '''
<!doctype html>
<title>Music Genre Classifier</title>
<h2>Upload an audio file to predict the music genre</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file accept=".wav,.mp3,.ogg">
  <input type=submit value=Upload>
</form>
{% if result %}
  <h3>Predicted Genre: <span style="color:green">{{ result }}</span></h3>
{% endif %}
'''

# --- Run Server ---
if __name__ == '__main__':
    app.run(debug=True)
