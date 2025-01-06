from flask import Flask, render_template, request
import tensorflow as tf #bib pour charger les modeles
from PIL import Image #pour manipuler les images
import numpy as np #traitement des données numeriques

app = Flask(__name__) #creation d'une instance
model = tf.keras.models.load_model('../model/savemodelVet.h5')

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
    image_array = np.array(image) / 255.0  #convertir l'image en tableau numpy et Normaliser
    image_array = np.expand_dims(image_array, axis=0)  #  Ajoute une dimension supplémentaire à un tableau d'image, souvent utilisée pour représenter un lot d'images.

    # Predict with the model
    logits = model.predict(image_array)  # utiliser le modele pour prédire les résultats a partir de l'image
    probabilities = tf.nn.softmax(logits[0]).numpy()  # convertir en probabilités entre 0 et 1, et softmax est une fonction

    # Get the predicted class and confidence
    predicted_class = class_names[np.argmax(probabilities)] #pour determiner la classe prédite
    confidence = 100 * np.max(probabilities) #calcule le pourcentage de confiance pour la classe predite

    return f"Predicted: {predicted_class} with {confidence:.2f}% confidence"

#Executer l'application en mode serveur
if __name__ == '__main__':
    app.run(debug=True)

#pour executer : py Flask.py