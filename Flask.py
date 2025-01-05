from flask import Flask, render_template, request
import tensorflow as tf #bib pour charger les modeles
from PIL import Image #pour manipuler les images
import numpy as np #traitement des données numeriques

app = Flask(__name__) #creation d'une instance
model = tf.keras.models.load_model('model/savemodelVet.h5')

#classes a prédire
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route('/') #route principale
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    # Process the image
    image = Image.open(file).convert('L')  # ouvrir et convertir l'image en niveaux gris
    image = image.resize((28, 28))  # Resize to 28x28 , c la taille attendue par le modele
    image_array = np.array(image) / 255.0  #convertir l'image en tableau numpy ,Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict with the model
    logits = model.predict(image_array)  # Raw outputs (logits)
    probabilities = tf.nn.softmax(logits[0]).numpy()  # Apply Softmax

    # Get the predicted class and confidence
    predicted_class = class_names[np.argmax(probabilities)]
    confidence = 100 * np.max(probabilities)

    return f"Predicted: {predicted_class} with {confidence:.2f}% confidence"

if __name__ == '__main__':
    app.run(debug=True)

#pour executer : py Flask.py