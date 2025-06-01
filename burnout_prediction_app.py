# Bibliotecas
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from matplotlib.ticker import FormatStrFormatter

# Configurações iniciais
sns.set_theme(style="whitegrid")

# Função para plotar importância no XGBoost
def plot_xgb_importance(model, features, importance_type='weight'):
    importance = model.get_booster().get_score(importance_type=importance_type)
    importance = {f: importance.get(f, 0) for f in features}
    features_sorted, values = zip(*sorted(importance.items(), key=lambda x: -x[1]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(features_sorted, values)
    
    max_value = max(values) if values else 1
    for bar in bars:
        ax.text(bar.get_width() + 0.01*max_value, 
                bar.get_y() + bar.get_height()/2, 
                f"{bar.get_width():.2f}", 
                va='center', ha='left')
    
    ax.set(xlim=(0, max_value*1.1), 
           title=f'Importância das Variáveis - XGBoost ({importance_type})')
    ax.xaxis.set_major_formatter('{x:.2f}')
    ax.grid(True, axis='x', alpha=0.3)
    st.pyplot(fig)

# Função para treinar modelo
def train_model(X_train, y_train, model_type='XGBoost'):
    if model_type == 'XGBoost':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

# Função para pré-processamento
def preprocess_data(df):
    df['burnout'] = df['mental_health'].apply(lambda x: 1 if x in ['Yes', 'Possibly'] else 0)
    cols_categoricas = ['tech_company', 'benefits', 'workplace_resources', 
                       'mh_employer_discussion', 'mh_coworker_discussion', 'medical_coverage', 'gender']
    le = LabelEncoder()
    for col in cols_categoricas:
        df[col] = le.fit_transform(df[col])
    return df.drop(columns=['country', 'mental_health'])

# Interface principal
def main():
    st.title("Análise de Burnout em Profissionais de Tecnologia")
    
    # Carregar e processar dados
    with st.spinner('Carregando e processando dados...'):
        df = pd.read_csv("C:/Users/amand/OneDrive/Desktop/burnout prediction/burnout_prediction.ipynb/data_mh_in_tech.csv")
        df_processed = preprocess_data(df)
    
    # Definir features e target
    features = ['age', 'gender', 'benefits', 'workplace_resources',
               'mh_employer_discussion', 'mh_coworker_discussion']
    target = 'burnout'
    
    # Dividir dados 
    X = df_processed[features]
    y = df_processed[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Seção exploratória
    if st.checkbox('Mostrar análise exploratória'):
        st.subheader("Insights dos Dados")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Profissionais em tecnologia", "72.4%")
            st.metric("Empresas que não discutem saúde mental", "67%")
        
        with col2:
            st.metric("Acesso a recursos de saúde mental", "34.7%")
            st.metric("Profissionais com burnout ou risco", "65.2%")
        
        st.subheader("Distribuição de Variáveis")
        selected_var = st.selectbox("Selecione a variável:", 
                                  ['benefits', 'workplace_resources', 
                                   'mh_employer_discussion', 'mh_coworker_discussion'])
        
        fig, ax = plt.subplots()
        df_processed[selected_var].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)
    
    # Modelagem
    st.header("Modelagem Preditiva")
    model_type = st.radio("Selecione o modelo:", ['XGBoost', 'Random Forest'])
    
    with st.spinner(f'Treinando modelo {model_type}...'):
        model = train_model(X_train, y_train, model_type)
        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    # Resultados
    st.subheader("Desempenho do Modelo")
    
    # Matriz de Confusão
    st.markdown(f"**Matriz de Confusão - {model_type}**")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), 
                annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    st.pyplot(fig)
    
    # Métricas
    st.markdown("**Relatório de Classificação**")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().style.highlight_max(axis=0, color='#90ee90'))
    
    # Curva ROC
    st.markdown(f"**Curva ROC - {model_type}**")
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    st.pyplot(fig)
    
    # Feature Importance
    st.subheader("Importância das Variáveis")
    
    if model_type == 'Random Forest':
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(10,5))
        sns.barplot(x='Importance', y='Feature', data=importance, ax=ax)
        ax.set_title('Importância das Variáveis - Random Forest')
        st.pyplot(fig)
    else:
        tab1, tab2 = st.tabs(["Gain", "Weight"])
        with tab1:
            plot_xgb_importance(model, features, importance_type='gain')
        with tab2:
            plot_xgb_importance(model, features, importance_type='weight')
    
    # Conclusões
    st.header("Conclusões")
    st.markdown("""
    1. Diálogo aberto sobre saúde mental é o fator mais crítico para redução de burnout
    2. Benefícios tradicionais têm impacto limitado sem cultura de apoio
    3. Modelos mostraram poder preditivo moderado (AUC 0.58-0.60)
    4. Necessidade de dados complementares (horas trabalhadas, qualidade de vida fora do trabalho, etc.)
    """)

if __name__ == "__main__":
    main()