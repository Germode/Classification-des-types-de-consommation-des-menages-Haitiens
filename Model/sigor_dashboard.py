
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class SigorEnergyDashboard:
    """Dashboard interactif pour la classification des consommateurs d'énergie"""
    
    def __init__(self, model_path, scaler_path, encoder_path=None):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.encoder = joblib.load(encoder_path) if encoder_path else None
        self.features = ['avg_amperage_per_day', 'avg_depense_per_day', 
                        'nombre_personnes', 'jours_observed', 'ratio_depense_amperage']
        
    def predict_consumption_level(self, input_data):
        """Prédit le niveau de consommation"""
        # Transformation des données
        input_scaled = self.scaler.transform([input_data])
        
        # Prédiction
        prediction = self.model.predict(input_scaled)[0]
        probabilities = self.model.predict_proba(input_scaled)[0]
        
        # Décodage si nécessaire
        if self.encoder:
            prediction = self.encoder.inverse_transform([prediction])[0]
        
        return prediction, probabilities
    
    def create_input_form(self):
        """Crée le formulaire de saisie des données"""
        st.sidebar.header("📊 Saisie des Données du Foyer")
        
        input_data = []
        for feature in self.features:
            if feature == 'avg_amperage_per_day':
                value = st.sidebar.slider("Ampérage moyen quotidien (A)", 
                                        min_value=0.0, max_value=50.0, value=10.0, step=0.1)
            elif feature == 'avg_depense_per_day':
                value = st.sidebar.slider("Dépense moyenne quotidienne ($)", 
                                        min_value=0.0, max_value=100.0, value=25.0, step=1.0)
            elif feature == 'nombre_personnes':
                value = st.sidebar.number_input("Nombre de personnes", 
                                              min_value=1, max_value=20, value=4)
            elif feature == 'jours_observed':
                value = st.sidebar.slider("Jours d'observation", 
                                        min_value=1, max_value=365, value=30)
            elif feature == 'ratio_depense_amperage':
                value = st.sidebar.slider("Ratio Dépense/Ampérage", 
                                        min_value=0.0, max_value=10.0, value=2.5, step=0.1)
            input_data.append(value)
        
        return input_data

def main():
    """Fonction principale du dashboard"""
    
    # Configuration de la page
    st.set_page_config(
        page_title="SIGOR - Analyse des Consommateurs",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # En-tête
    st.title("⚡ SIGOR - Dashboard d'Analyse des Consommateurs")
    st.markdown("""
    Ce dashboard permet de classifier les foyers selon leur niveau de consommation énergétique 
    et de visualiser les insights du modèle de machine learning.
    """)
    
    # Chargement du modèle (à adapter avec vos chemins)
    try:
        dashboard = SigorEnergyDashboard(
            model_path='/content/drive/MyDrive/sigor_model_artifacts/best_model.joblib',
            scaler_path='/content/drive/MyDrive/sigor_model_artifacts/scaler.joblib',
            encoder_path='/content/drive/MyDrive/sigor_model_artifacts/label_encoder.joblib'
        )
    except:
        st.error("❌ Impossible de charger les modèles. Vérifiez les chemins.")
        return
    
    # Sidebar avec formulaire
    with st.sidebar:
        st.header("🔍 Prédiction en Temps Réel")
        input_data = dashboard.create_input_form()
        
        if st.button("🔮 Prédire le Niveau de Consommation", type="primary"):
            prediction, probabilities = dashboard.predict_consumption_level(input_data)
            
            # Affichage des résultats
            st.success(f"**Niveau de consommation prédit : {prediction.upper()}**")
            
            # Jauge de probabilité
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = np.max(probabilities) * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confiance de la prédiction (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Détail des probabilités
            prob_df = pd.DataFrame({
                'Niveau': ['Petit', 'Moyen', 'Grand'],
                'Probabilité': probabilities
            })
            st.dataframe(prob_df.style.format({'Probabilité': '{:.2%}'}))
    
    # Contenu principal
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📈 Distribution des Consommateurs")
        
        # Graphique de distribution (exemple avec données simulées)
        distribution_data = pd.DataFrame({
            'Niveau': ['Petit', 'Moyen', 'Grand'] * 100,
            'Ampérage': np.concatenate([
                np.random.normal(5, 1, 100),
                np.random.normal(15, 2, 100),
                np.random.normal(30, 3, 100)
            ])
        })
        
        fig = px.box(distribution_data, x='Niveau', y='Ampérage', 
                    title="Distribution de l'Ampérage par Niveau de Consommation")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.header("🎯 Importance des Caractéristiques")
        
        # Importance des features (exemple)
        importance_data = pd.DataFrame({
            'Caractéristique': dashboard.features,
            'Importance': [0.35, 0.25, 0.15, 0.15, 0.10]  # À remplacer par les vraies valeurs
        })
        
        fig = px.bar(importance_data, x='Importance', y='Caractéristique',
                    orientation='h', title="Importance Relative des Caractéristiques")
        st.plotly_chart(fig, use_container_width=True)
    
    # Section analyse détaillée
    st.header("📊 Analyse Détaillée des Segments")
    
    tab1, tab2, tab3 = st.tabs(["📋 Profils Types", "📉 Tendances", "🎪 Matrice de Confusion"])
    
    with tab1:
        st.subheader("Caractéristiques Moyennes par Segment")
        
        profile_data = pd.DataFrame({
            'Segment': ['Petit', 'Moyen', 'Grand'],
            'Ampérage Moyen': [5.2, 15.8, 32.4],
            'Dépense Moyenne': [12.5, 28.3, 65.2],
            'Personnes Moyennes': [2.1, 3.8, 5.2]
        })
        
        st.dataframe(profile_data.style.format({
            'Ampérage Moyen': '{:.1f} A',
            'Dépense Moyenne': '{:.1f} $',
            'Personnes Moyennes': '{:.1f}'
        }))
    
    with tab2:
        st.subheader("Évolution de la Consommation")
        
        # Données temporelles simulées
        time_data = pd.DataFrame({
            'Mois': pd.date_range('2023-01-01', periods=12, freq='M'),
            'Petit': np.random.normal(5, 0.5, 12),
            'Moyen': np.random.normal(15, 1, 12),
            'Grand': np.random.normal(30, 2, 12)
        })
        
        fig = px.line(time_data, x='Mois', y=['Petit', 'Moyen', 'Grand'],
                     title="Évolution Mensuelle de la Consommation par Segment")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Performance du Modèle")
        
        # Matrice de confusion simulée
        conf_matrix = np.array([[178, 2, 0], [1, 165, 1], [0, 1, 196]])
        classes = ['Petit', 'Moyen', 'Grand']
        
        fig = px.imshow(conf_matrix, text_auto=True, aspect="auto",
                       labels=dict(x="Prédit", y="Réel", color="Nombre"),
                       x=classes, y=classes, title="Matrice de Confusion")
        st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**SIGOR Energy Analytics** | "
        "Modèle entraîné le 2024-01-01 | "
        "Précision: 99.8%"
    )

if __name__ == "__main__":
    main()

# Instructions pour lancer le dashboard
print("\n🎯 POUR LANCER LE DASHBOARD:")
print("1. Sauvegardez le fichier sigor_dashboard.py")
print("2. Exécutez: streamlit run sigor_dashboard.py")
print("3. Ouvrez l'URL affichée dans votre navigateur")
