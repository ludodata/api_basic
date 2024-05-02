# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)

# Charge le model
model = pickle.load(open('boostv4_df.pkl','rb')) # dans le meme repertoire

#On crée un fonction predict pour faire la prediction
def predict(data):
    
    
    prediction = model.predict_proba(data)# recupere la prediction (array 2d)

    prediction_proba = prediction[:, 1] # on recupere que la proba 2eme colonne pour la classe 1 ( ne rembourse PAS)

    
    prediction_class = (prediction[:, 1] >= 0.14).astype(int) # retourne un booleen si >= seuil , convertit en 1 ou 0
                                                        
    # Crée un dictionnaire contenant les prédictions de classe et de probabilité
    prediction_dict = {'prediction_class': prediction_class.tolist(),
                       'prediction_proba': prediction_proba.tolist()}
   
    
    return (prediction_dict)


@app.route('/', methods=['POST']) #decorateur applique la fonction
def predict_api():
    # Obtient les données à partir de la requete post
    data = request.get_json()
    # On applique la fonction predict à nos data et stocke dans var resultat
    resultat = predict(data)
    # Retourne la var resultat sous forme de JSON
    return jsonify(resultat)

if __name__ == '__main__':
    #app.run(port=5000, debug=True)
    app.run()