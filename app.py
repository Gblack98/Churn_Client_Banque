# Importation des bibliothèques nécessaires
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

# Style CSS personnalisé
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');
    :root {
        --primary-blue: #1e3a8a;
        --dark-bg: #0f172a;
        --metric-card: #1e293b;
    }
    * {
        font-family: 'Inter', sans-serif;
        box-sizing: border-box;
    }
    .main {
        background-color: var(--dark-bg);
        color: #f8fafc;
    }
    .banking-header {
        background: linear-gradient(135deg, var(--primary-blue) 0%, #1e40af 100%);
        padding: 2rem;
        border-radius: 0 0 25px 25px;
        margin-bottom: 2rem;
        color: white;
    }
    .metric-card {
        background: var(--metric-card);
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: var(--metric-card);
        color: #f8fafc;
        text-align: center;
        padding: 10px 0;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='banking-header' style='text-align: center;'><h1>Tableau de Bord Client</h1><p>Intelligence Client 360° – Analyse, Crédit, Risques & Finance</p></div>", unsafe_allow_html=True)

# Chargement des données
@st.cache_data
def charger_donnees():
    try:
        df = pd.read_csv("BankChurners.csv")
        colonnes_requises = [
            'Attrition_Flag', 'Customer_Age', 'Credit_Limit', 'Total_Trans_Amt',
            'Total_Relationship_Count', 'Months_on_book', 'Avg_Utilization_Ratio',
            'Total_Revolving_Bal', 'Total_Trans_Ct', 'Education_Level'
        ]
        colonnes_manquantes = [col for col in colonnes_requises if col not in df.columns]
        if colonnes_manquantes:
            st.error(f"Colonnes manquantes : {', '.join(colonnes_manquantes)}")
            return None
        df['Attrition_Flag'] = df['Attrition_Flag'].map({'Attrited Customer': 1, 'Existing Customer': 0})
        df['CLV'] = (df['Total_Trans_Amt'] * df['Total_Relationship_Count']) / df['Months_on_book'].replace(0, 1)
        df['Score_Risque'] = np.where(df['Avg_Utilization_Ratio'] > 0.75, 3,
                                    np.where(df['Avg_Utilization_Ratio'] > 0.5, 2, 1))
        return df.dropna(subset=colonnes_requises)
    except Exception as e:
        st.error(f"Erreur de chargement : {str(e)}")
        return None

# Modèle prédictif
@st.cache_resource
def entrainer_modele_churn(df):
    try:
        features = ['Customer_Age', 'Credit_Limit', 'Total_Revolving_Bal',
                   'Avg_Utilization_Ratio', 'Total_Trans_Ct', 'CLV']
        X = df[features]
        y = df['Attrition_Flag']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        df['Probabilite_Churn'] = model.predict_proba(X)[:, 1]
        return df, model.feature_importances_
    except Exception as e:
        st.error(f"Erreur d'entraînement : {str(e)}")
        return df, None

# PAGE : Dashboard Global
def page_dashboard(df, importance_caracteristiques, type_segmentation):
    # Section KPI - Cartes améliorées avec emojis et couleurs sur le texte
    cols = st.columns(4)
    kpis = [
        ("💰 Portefeuille Client", "10,127", "Clients Actifs", "portefeuille"),
        ("📉 Taux de Désabonnement", "16.1%", "vs trim précédent", "taux"),
        ("💵 CLV Médian", "$366", "Valeur Client", "clv"),
        ("⚠️ Exposition au Risque", "747", "Clients à Haut Risque", "risque")
    ]

    # Personnalisation avec couleurs sur le texte via inline CSS
    for col, (titre, valeur, sous_titre, classe) in zip(cols, kpis):
        with col:
            st.markdown(f"""
                <div class='metric-card {classe}' style="border: 1px solid #333; padding: 1.5rem; border-radius: 10px; background: var(--metric-card);">
                    <div class='metric-title' style="color: {'#1e40af' if classe=='portefeuille' else '#16a34a' if classe=='taux' else '#d97706' if classe=='clv' else '#dc2626'}; font-weight: bold;">{titre}</div>
                    <div class='metric-value' style="font-size: 1.75rem; color: #f8fafc; margin: 0.5rem 0;">{valeur}</div>
                    <div class='metric-subtitle' style="font-size: 0.9rem; color: #94a3b8;">{sous_titre}</div>
                </div>
            """, unsafe_allow_html=True)

    # Visualisations principales
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Distribution du Risque de Churn")
        fig = px.histogram(df, x='Probabilite_Churn', nbins=50, 
                          color_discrete_sequence=['#2563eb'])
        fig.update_layout(template="plotly_dark", bargap=0.1,
                         xaxis_title="Probabilité de Churn",
                         yaxis_title="Nombre de Clients")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### Facteurs d'Influence")
        if importance_caracteristiques is not None:
            features = ['Âge', 'Limite Crédit', 'Solde Récurrent', 
                        'Utilisation', 'Transactions', 'CLV']
            df_imp = pd.DataFrame({
                'Facteur': features,
                'Importance': importance_caracteristiques
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(df_imp, x='Importance', y='Facteur', orientation='h',
                        color='Importance', color_continuous_scale='Blues',
                        template="plotly_dark")
            fig.update_layout(xaxis_title="Importance Relative",
                            yaxis_title="Facteur Prédictif")
            st.plotly_chart(fig, use_container_width=True)
    
    # Segmentation
    st.markdown("---")
    st.markdown("#### Segmentation Clientèle")
    page_segments(df, type_segmentation)

# PAGE : Segmentation
def page_segments(df, type_segmentation):
    segmentation_map = {
        "Comportementale": ['Total_Trans_Ct', 'Total_Trans_Amt', 'Total_Revolving_Bal'],
        "Valeur Client": ['CLV', 'Customer_Age', 'Months_on_book'],
        "Risque Crédit": ['Credit_Limit', 'Avg_Utilization_Ratio', 'Score_Risque']
    }
    
    cols_seg = segmentation_map[type_segmentation]
    n_clusters = st.slider("Nombre de Segments", 2, 5, 3, key="n_segments")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Segment'] = kmeans.fit_predict(df[cols_seg])
    
    fig = px.scatter(df, x=cols_seg[0], y=cols_seg[1], color='Segment',
                    template="plotly_dark", title="Visualisation des Segments",
                    labels={cols_seg[0]: cols_seg[0].replace('_', ' '),
                            cols_seg[1]: cols_seg[1].replace('_', ' ')})
    st.plotly_chart(fig, use_container_width=True)
# PAGE : Prédiction Churn
def page_prediction(df):
    st.markdown("## 🔮 Prédiction de Churn")
    
    # Vérification des données
    if 'Probabilite_Churn' not in df.columns:
        st.error("Problème de chargement des prédictions")
        return

    # Configuration de la mise en page
    st.markdown("""
    <style>
        div[data-testid="stHorizontalBlock"] {
            align-items: stretch;
            gap: 2rem;
        }
        .dataframe-overlay, .plotly-container {
            border: 1px solid #2d3748;
            border-radius: 10px;
            padding: 1rem;
            background: #1e293b;
        }
    </style>
    """, unsafe_allow_html=True)

    # Layout en deux colonnes avec ratio ajusté
    col_data, col_viz = st.columns([5, 7], gap="large")

    with col_data:
        st.markdown("### 📋 Liste des Clients à Risque")
        df_display = df[['Customer_Age', 'Credit_Limit', 'Total_Trans_Ct', 
                        'CLV', 'Probabilite_Churn']].sort_values('Probabilite_Churn', ascending=False)
        
        # Affichage du dataframe avec style
        st.markdown('<div class="dataframe-overlay">', unsafe_allow_html=True)
        st.dataframe(
            df_display.style.format({
                'CLV': '${:,.0f}',
                'Probabilite_Churn': '{:.1%}',
                'Credit_Limit': '${:,.0f}'
            }).applymap(
                lambda x: 'color: #ff4b4b' if isinstance(x, float) and x > 0.7 else '',
                subset=['Probabilite_Churn']
            ),
            height=700,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_viz:
        st.markdown("### 📊 Analyse Multidimensionnelle")
        st.markdown('<div class="plotly-container">', unsafe_allow_html=True)
        
        # Configuration du graphique
        fig = px.scatter(
            df,
            x='Customer_Age',
            y='CLV',
            size='Credit_Limit',
            color='Probabilite_Churn',
            hover_data={
                'Total_Trans_Amt': ':.2f',
                'Avg_Utilization_Ratio': ':.0%',
                'Customer_Age': True,
                'CLV': '$.2f'
            },
            color_continuous_scale='RdYlGn_r',
            template='plotly_dark'
        )
        
        # Personnalisation du layout
        fig.update_layout(
            xaxis_title="Âge du Client (années)",
            yaxis_title="Customer Lifetime Value (USD)",
            coloraxis_colorbar_title='Probabilité<br>de Churn',
            hovermode='closest',
            height=700,  # Hauteur fixe identique au dataframe
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Section des filtres avancés
    with st.expander("🔎 Filtres Avancés", expanded=False):
        cols = st.columns(3)
        with cols[0]:
            min_prob = st.slider("Probabilité minimale", 0.0, 1.0, 0.0, 0.01)
        with cols[1]:
            max_prob = st.slider("Probabilité maximale", 0.0, 1.0, 1.0, 0.01)
        with cols[2]:
            st.metric("Clients filtrés", 
                     df['Probabilite_Churn'].between(min_prob, max_prob).sum(),
                     help="Nombre de clients dans la plage sélectionnée")
# Fonction principale
def main():
    df = charger_donnees()
    if df is None:
        return
    
    df, feature_importances = entrainer_modele_churn(df)
    
    # Configuration sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.selectbox("Menu", ["Dashboard Global", "Prédiction Churn"])
        
        if page == "Dashboard Global":
            type_seg = st.selectbox("Type de Segmentation", 
                                   ["Comportementale", "Valeur Client", "Risque Crédit"])
        else:
            seuil_churn = st.slider("Seuil d'alerte", 0.0, 1.0, 0.7, step=0.05)
            df = df[df['Probabilite_Churn'] >= seuil_churn]
        
        st.markdown("---")
        plage_clv = st.slider("Filtre CLV", 
                             float(df['CLV'].min()), 
                             float(df['CLV'].max()), 
                             (float(df['CLV'].quantile(0.25)), float(df['CLV'].quantile(0.75))))
        df = df[df['CLV'].between(plage_clv[0], plage_clv[1])]
    
    # Gestion des pages
    if page == "Dashboard Global":
        page_dashboard(df, feature_importances, type_seg)
    else:
        page_prediction(df)
    
 

if __name__ == "__main__":
    main()