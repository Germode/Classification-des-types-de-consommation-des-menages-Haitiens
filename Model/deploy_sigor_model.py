
import os
import sys
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

class SigorModelDeployer:
    """Classe pour déployer et tester le modèle en production"""
    
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.encoder = None
        self.features = ['avg_amperage_per_day', 'avg_depense_per_day', 
                        'nombre_personnes', 'jours_observed', 'ratio_depense_amperage']
        
    def load_artifacts(self):
        """Charge tous les artefacts du modèle"""
        try:
            # Trouve le dernier modèle
            model_files = [f for f in os.listdir(self.model_dir) 
                         if f.startswith('best_model') and f.endswith('.joblib')]
            if not model_files:
                raise FileNotFoundError("Aucun modèle trouvé")
                
            latest_model = sorted(model_files)[-1]
            model_path = os.path.join(self.model_dir, latest_model)
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.joblib'))
            self.encoder = joblib.load(os.path.join(self.model_dir, 'label_encoder.joblib'))
            
            print("✅ Artefacts chargés avec succès")
            return True
            
        except Exception as e:
            print(f"❌ Erreur lors du chargement: {e}")
            return False
    
    def predict_single(self, input_dict):
        """Prédiction pour un seul foyer"""
        try:
            # Construction du vecteur d'entrée
            input_vector = [input_dict.get(feature, 0) for feature in self.features]
            input_scaled = self.scaler.transform([input_vector])
            
            # Prédiction
            prediction_encoded = self.model.predict(input_scaled)[0]
            probabilities = self.model.predict_proba(input_scaled)[0]
            prediction = self.encoder.inverse_transform([prediction_encoded])[0]
            
            return {
                'prediction': prediction,
                'probabilities': {
                    'petit': float(probabilities[0]),
                    'moyen': float(probabilities[1]),
                    'grand': float(probabilities[2])
                },
                'confidence': float(np.max(probabilities))
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def batch_predict(self, data_df):
        """Prédiction par lot"""
        try:
            # Vérification des colonnes
            missing_cols = set(self.features) - set(data_df.columns)
            if missing_cols:
                return {'error': f'Colonnes manquantes: {missing_cols}'}
            
            # Préparation des données
            X = data_df[self.features].fillna(0)
            X_scaled = self.scaler.transform(X)
            
            # Prédictions
            predictions_encoded = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)
            predictions = self.encoder.inverse_transform(predictions_encoded)
            
            # Résultats
            results = data_df.copy()
            results['niveau_conso_pred'] = predictions
            results['confiance'] = np.max(probabilities, axis=1)
            
            for i, classe in enumerate(self.encoder.classes_):
                results[f'prob_{classe}'] = probabilities[:, i]
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def health_check(self):
        """Vérifie l'état du modèle"""
        checks = {
            'model_loaded': self.model is not None,
            'scaler_loaded': self.scaler is not None,
            'encoder_loaded': self.encoder is not None,
            'features_match': len(self.features) == 5
        }
        
        # Test de prédiction
        if all(checks.values()):
            test_input = {feature: 1.0 for feature in self.features}
            test_pred = self.predict_single(test_input)
            checks['prediction_works'] = 'error' not in test_pred
        
        return checks

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialisation
    deployer = SigorModelDeployer('/content/drive/MyDrive/sigor_model_artifacts')
    
    # Chargement
    if deployer.load_artifacts():
        # Health check
        print("🔍 HEALTH CHECK:")
        health = deployer.health_check()
        for check, status in health.items():
            print(f"  {check}: {'✅' if status else '❌'}")
        
        # Test de prédiction
        print("\n🎯 TEST DE PRÉDICTION:")
        test_data = {
            'avg_amperage_per_day': 15.5,
            'avg_depense_per_day': 28.0,
            'nombre_personnes': 4,
            'jours_observed': 30,
            'ratio_depense_amperage': 1.8
        }
        
        result = deployer.predict_single(test_data)
        print(f"Input: {test_data}")
        print(f"Résultat: {result}")
        
        print("\n🚀 MODÈLE PRÊT POUR LA PRODUCTION!")

print("\n📁 FICHIERS CRÉÉS:")
print("1. nested_cross_validation.py - Validation croisée imbriquée")
print("2. pipeline_tests.py - Tests unitaires complets") 
print("3. sigor_dashboard.py - Dashboard interactif Streamlit")
print("4. deploy_sigor_model.py - Script de déploiement production")
print("\n🎯 PROCHAINES ÉTAPES:")
print("1. Exécutez les tests unitaires pour vérifier la robustesse")
print("2. Lancez le dashboard: streamlit run sigor_dashboard.py")
print("3. Utilisez deploy_sigor_model.py pour l'intégration production")
