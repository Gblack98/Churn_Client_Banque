import numpy as np
import pandas as pd
from faker import Faker
from scipy.stats import skewnorm, beta, poisson

# Configuration initiale
np.random.seed(42)
fake = Faker()
num_rows = 10000  # Nombre de lignes à générer

# Génération des données de base
data = {
    'CLIENTNUM': [fake.unique.random_number(digits=9) for _ in range(num_rows)],
    'Attrition_Flag': np.random.choice(['Existing Customer', 'Attrited Customer'], 
                                      size=num_rows, p=[0.84, 0.16]),
    'Customer_Age': np.clip(np.round(skewnorm.rvs(5, loc=45, scale=10, size=num_rows)), 18, 100),
    'Gender': np.random.choice(['M', 'F'], size=num_rows, p=[0.55, 0.45]),
    'Dependent_count': np.clip(poisson.rvs(2.5, size=num_rows), 0, 5),
    'Education_Level': np.random.choice(
        ['Unknown', 'Uneducated', 'High School', 'College', 'Graduate', 'Post-Graduate', 'Doctorate'],
        size=num_rows,
        p=[0.05, 0.15, 0.25, 0.2, 0.2, 0.1, 0.05]
    ),
    'Marital_Status': np.random.choice(
        ['Married', 'Single', 'Divorced', 'Unknown'],
        size=num_rows,
        p=[0.5, 0.3, 0.15, 0.05]
    ),
    'Income_Category': pd.cut(
        np.exp(np.random.normal(10.5, 0.4, num_rows)),
        bins=[0, 40000, 60000, 80000, 120000, np.inf],
        labels=[
            'Less than $40K',
            '$40K - $60K',
            '$60K - $80K',
            '$80K - $120K',
            '$120K +'
        ]
    ),
    'Card_Category': np.random.choice(
        ['Blue', 'Silver', 'Gold', 'Platinum'],
        size=num_rows,
        p=[0.85, 0.1, 0.04, 0.01]
    ),
    'Months_on_book': np.random.randint(12, 72, size=num_rows),
    'Total_Relationship_Count': np.random.poisson(4, size=num_rows) + 1,
    'Months_Inactive_12_mon': np.random.randint(0, 6, size=num_rows),
    'Contacts_Count_12_mon': np.random.poisson(2, size=num_rows),
    'Credit_Limit': np.round(np.exp(np.random.normal(9.5, 0.8, num_rows))),
    'Total_Amt_Chng_Q4_Q1': beta.rvs(2, 5, size=num_rows) * 4,
    'Total_Trans_Amt': np.round(np.exp(np.random.normal(7, 0.5, num_rows))),
    'Total_Trans_Ct': np.random.poisson(30, size=num_rows),
    'Total_Ct_Chng_Q4_Q1': beta.rvs(2, 5, size=num_rows) * 3,
}

# Création du DataFrame initial
df = pd.DataFrame(data)

# Génération des colonnes dépendantes
df['Total_Revolving_Bal'] = np.round(df['Credit_Limit'] * beta.rvs(2, 5, size=num_rows))
df['Avg_Open_To_Buy'] = df['Credit_Limit'] - df['Total_Revolving_Bal']
df['Avg_Utilization_Ratio'] = np.clip(df['Total_Revolving_Bal'] / df['Credit_Limit'], 0, 1)

# Génération des caractéristiques Naive Bayes
attrition_numeric = np.where(df['Attrition_Flag'] == 'Attrited Customer', 1, 0)
df['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1'] = np.clip(
    beta.rvs(0.5, 5, size=num_rows) * attrition_numeric * 0.9 + 
    beta.rvs(5, 5, size=num_rows) * 0.1, 0, 1
)
df['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'] = 1 - df.iloc[:, -1]

# Ajout de valeurs manquantes
columns_with_nulls = ['Education_Level', 'Marital_Status', 'Income_Category']
for col in columns_with_nulls:
    df.loc[df.sample(frac=0.05).index, col] = 'Unknown'

# Formatage final
df = df.round(4)
df = df.convert_dtypes()

# Sauvegarde
df.to_csv('synthetic_banking_data.csv', index=False)
print("Dataset généré avec succès!")