import pytest
import json
import numpy as np
from app import predict

# importer les données via un JSON dans le meme dossier

chemin_fichier_json = "data_test1.json"

# Charger le fichier JSON
with open(chemin_fichier_json, "r") as fichier_json:
    data_json = json.load(fichier_json)

data_reshaped = np.array(data_json)
data_reshaped = data_reshaped.reshape(1, -1)


# Test la sortie de la fonction predict ( dictionnaire avec 2 clefs )
def test_predict_format_output():
    # Créez des données de test
    test_data = data_reshaped

    # Appelez la fonction predict avec les données de test
    result = predict(test_data)

  
    # Assurez-vous que le résultat est un dictionnaire avec les clés 'prediction_class' et 'prediction_proba'
    assert isinstance(result, dict)
    assert 'prediction_class' in result
    assert 'prediction_proba' in result

# Test de la fonction predict ( la prediction est bien entre 0 et 1)
def test_predict_class_output():
    # Créez des données de test
    test_data = data_reshaped
    expected_class = [1] # on sait que notre individu apartient a la classe 1
    # Appelez la fonction predict avec les données de test
    result = predict(test_data)


    # Vérifiez si la classe prédite est correcte
    assert result["prediction_class"] == expected_class