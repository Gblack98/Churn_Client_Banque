# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from streamlit_extras.stylable_container import stylable_container

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Gabar Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.gabar-analytics.com',
        'Report a bug': 'https://www.gabar-analytics.com/support',
        'About': "Plateforme d'Intelligence Client pour Institutions Financières"
    }
)

# Style CSS avancé pour une interface moderne et luxueuse
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');
    
    * {{
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
    }}
    
    .main {{
        background: #0f172a;
        color: #f8fafc;
    }}
    
    .banking-header {{
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        padding: 2rem;
        border-radius: 0 0 25px 25px;
        margin-bottom: 2rem;
    }}
    
    .stPlotlyChart {{
        border: 1px solid #334155;
        border-radius: 15px;
        padding: 15px;
        background: #1e293b;
    }}
    
    .dataframe {{
        border-radius: 10px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }}
    
    .metric-box {{
        background: rgba(34, 139, 230, 0.15);
        border: 1px solid #228BE5;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        transition: transform 0.3s ease;
    }}
    
    .metric-box:hover {{
        transform: translateY(-5px);
    }}
</style>
""", unsafe_allow_html=True)

# Fonction pour charger les données avec gestion des erreurs
@st.cache_data
def charger_donnees(uploaded_file):
    """
    Charge les données à partir d'un fichier CSV.
    Gère les erreurs de chargement et effectue un prétraitement des données.
    """
    try:
        df = pd.read_csv(uploaded_file)
        # Validation des colonnes nécessaires
        colonnes_requises = ['Attrition_Flag', 'Customer_Age', 'Credit_Limit', 'Total_Trans_Amt']
        if not all(col in df.columns for col in colonnes_requises):
            st.error(f"Colonnes manquantes: {', '.join([col for col in colonnes_requises if col not in df.columns])}")
            return None
        
        # Conversion des valeurs de la colonne Attrition_Flag
        df['Attrition_Flag'] = df['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
        
        # Calcul de la valeur client (CLV)
        df['CLV'] = (df['Total_Trans_Amt'] * df['Total_Relationship_Count']) / df['Months_on_book'].replace(0, 1)
        
        # Calcul du score de risque
        df['Score_Risque'] = np.where(df['Avg_Utilization_Ratio'] > 0.75, 3, 
                                    np.where(df['Avg_Utilization_Ratio'] > 0.5, 2, 1))
        
        return df.dropna(subset=colonnes_requises)
    except Exception as e:
        st.error(f"Erreur de chargement : {str(e)}")
        return None

# Modèle prédictif pour le churn
@st.cache_resource
def entrainer_modele_churn(df):
    """
    Entraîne un modèle de prédiction de churn et retourne les probabilités de désabonnement.
    """
    try:
        # Sélection des caractéristiques pour le modèle
        X = df[['Customer_Age', 'Credit_Limit', 'Total_Revolving_Bal', 
               'Avg_Utilization_Ratio', 'Total_Trans_Ct', 'CLV']]
        y = df['Attrition_Flag']
        
        # Division des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraînement du modèle RandomForest
        model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        
        # Ajout des probabilités de churn au DataFrame
        df['Probabilite_Churn'] = model.predict_proba(X)[:, 1]
        return df, model.feature_importances_
    except Exception as e:
        st.error(f"Erreur d'entraînement : {str(e)}")
        return df, None

# Interface utilisateur
def main():
    # Header bancaire
    with stylable_container(key="header", css_styles=".banking-header { color: white !important; }"):
        st.markdown("""
            <div class='banking-header'>
                <h1 style='margin:0;padding:0;'>Gabar Analytics </h1>
                <p style='opacity:0.8;'>Intelligence Client 360°</p>
            </div>
        """, unsafe_allow_html=True)

    # Upload de données
    with st.sidebar:
        uploaded_file = st.file_uploader("📊 Importer les données clients", type=["csv"],
                                       help="Format requis : Données transactionnelles clients")
        if uploaded_file:
            if st.button("🔄 Actualiser l'analyse"):
                st.cache_data.clear()
                st.rerun()

    if not uploaded_file:
        col1, col2 = st.columns([1, 3])
        with col2:
            st.image("banking_dashboard.png", use_container_width=True)
        return

    df = charger_donnees(uploaded_file)
    if df is None:
        return

    # Entraînement automatique du modèle de churn
    df, importance_caracteristiques = entrainer_modele_churn(df)
    
    # Section de filtres avancés
    with st.sidebar:
        st.header("🔎 Filtres Stratégiques")
        
        # Segmentation clientèle
        type_segmentation = st.selectbox("Segmentation Clientèle", 
                                       ["Comportementale", "Valeur Client", "Risque Crédit"],
                                       index=0)
        
        # Paramètres de risque
        niveau_risque = st.slider("Niveau de Risque Acceptable", 1, 3, 2,
                                help="Filtre basé sur le score de risque calculé")
        
        # Filtre de valeur client
        plage_clv = st.slider("Plage de Valeur Client (CLV)", 
                            float(df['CLV'].min()), 
                            float(df['CLV'].max()), 
                            (float(df['CLV'].quantile(0.25)), 
                             float(df['CLV'].quantile(0.75))))

    # KPI Stratégiques
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        with stylable_container(key="kpi1", css_styles=[".metric {font-size: 1.5rem !important;}"]):
            st.metric("Portefeuille Client", f"{len(df):,}", "Clients Actifs")
    with col2:
        taux_churn = df['Attrition_Flag'].mean() * 100
        couleur_delta = "inverse" if taux_churn > 15 else "normal"
        st.metric("Taux de Désabonnement", f"{taux_churn:.1f}%", 
                delta_color=couleur_delta, delta="+2.1% vs trim. précédent")
    with col3:
        st.metric("Valeur Client Moyenne", f"${df['CLV'].median():,.0f}", 
                "CLV médian")
    with col4:
        st.metric("Exposition au Risque", 
                f"{len(df[df['Score_Risque'] >= 3]):,}", 
                "Clients à haut risque")

    # Visualisations Principales
    tab1, tab2, tab3 = st.tabs(["📈 Analyse Prédictive", "👥 Segmentation Clientèle", "📋 Profils à Risque"])

    with tab1:
        # Analyse prédictive
        st.subheader("Analyse Prédictive du Churn")
        
        # Graphique de distribution des probabilités de churn
        fig = px.histogram(df, x='Probabilite_Churn', 
                         nbins=50, title='Distribution des Probabilités de Désabonnement',
                         color_discrete_sequence=['#2563eb'],
                         labels={'Probabilite_Churn': 'Probabilité de Churn'})
        fig.update_layout(bargap=0.1, xaxis_title="Probabilité de Churn", yaxis_title="Nombre de Clients")
        st.plotly_chart(fig, use_container_width=True)
        
        # Graphique des facteurs influents
        st.subheader("Facteurs Influents sur le Churn")
        if importance_caracteristiques is not None:
            caracteristiques = ['Âge', 'Limite Crédit', 'Solde Récurrent', 
                              'Utilisation', 'Transactions', 'CLV']
            df_importance = pd.DataFrame({
                'Facteur': caracteristiques,
                'Importance': importance_caracteristiques
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(df_importance, x='Importance', y='Facteur', 
                       orientation='h', color='Importance',
                       color_continuous_scale='Blues',
                       labels={'Importance': 'Importance Relative', 'Facteur': 'Facteur'})
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Segmentation clientèle
        st.subheader("Segmentation Clientèle")
        
        # Choix du nombre de segments
        n_segments = st.slider("Nombre de Segments", 2, 5, 3)
        
        # Sélection des caractéristiques en fonction du type de segmentation
        if type_segmentation == "Comportementale":
            caracteristiques = ['Total_Trans_Ct', 'Total_Trans_Amt', 'Total_Revolving_Bal']
        elif type_segmentation == "Valeur Client":
            caracteristiques = ['CLV', 'Customer_Age', 'Months_on_book']
        else:
            caracteristiques = ['Credit_Limit', 'Avg_Utilization_Ratio', 'Score_Risque']
        
        # Application du clustering
        kmeans = KMeans(n_clusters=n_segments, random_state=42)
        df['Segment'] = kmeans.fit_predict(df[caracteristiques])
        
        # Visualisation des segments
        fig = px.scatter_matrix(df, dimensions=caracteristiques, 
                              color='Segment', hover_name='Education_Level',
                              title="Matrice de Segmentation Clientèle",
                              labels={col: col.replace('_', ' ') for col in caracteristiques})
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Liste des clients à risque
        clients_risque = df[(df['Probabilite_Churn'] > 0.7) & 
                          (df['Score_Risque'] >= niveau_risque) &
                          (df['CLV'].between(*plage_clv))]
        
        st.subheader(f"Clients Prioritaires ({len(clients_risque)})")
        st.dataframe(
            clients_risque[['Customer_Age', 'Gender', 'Education_Level', 
                          'CLV', 'Probabilite_Churn', 'Score_Risque']]
            .sort_values('Probabilite_Churn', ascending=False)
            .style.format({'CLV': '${:.0f}', 'Probabilite_Churn': '{:.1%}'}),
            height=500,
            use_container_width=True
        )

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; opacity: 0.7;">
            <p>Solution développée par Ibrahima Gabar Diop</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()