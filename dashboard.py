#manipulation des données
import pandas as pd
import numpy as np

#requêtes à l'API
import requests

#plots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns

#récupérer modèles & explainer
import joblib

#interprétabilité
import shap
shap.initjs()

#dashboard
import streamlit as st
import streamlit.components.v1 as components
st.set_option('deprecation.showPyplotGlobalUse', False)

#définition de fonctions
#fonction requête API
def get_data(url):
    resp = requests.get(url)
    return resp.json()

#fonction pour tracer des distplots
def distplots(data,var,height=600):
    x1 = data.loc[data['PRED'] == 0, var]
    x2 = data.loc[data['PRED'] == 1, var]
    x=data.loc[id_client][var]
    plot = ff.create_distplot([x1,x2], [0,1], show_hist=False, colors=['green','red'])
    plot.add_vline(x,line_width=2,line_dash="dash",line_color="orange",annotation_text="Client",annotation_font_color='orange',annotation_font_size=18)
    plot.update_layout(height=height)
    titre = "Distribution de la variable: "+var+" & Positionnement du client"
    plot.update_layout(title_text=titre)
    return plot

#chargement des données
#chargement de l'explainer SHAP
explainer = joblib.load('explainer.sav')
#chargement des fichiers de travail
#chargement de l'échantillon
clients = pd.read_csv('sample.csv')
#colonne SK_ID_CURR en index
clients.set_index('SK_ID_CURR', inplace = True)
#chargement des résultats de la prédiction (pour les graphs)
clients_pred = pd.read_csv('sample_pred.csv')
#colonne SK_ID_CURR en index
clients_pred.set_index('SK_ID_CURR', inplace = True)

#Titre
st.title(" Customer Dashboard : Loans")

#liste pour sélectionner un client
id_client = st.selectbox('Please select a Client ID :',clients.index )
id_client = int(id_client)

#url de requetage en fonction de l'ID client
url = "https://apipred.herokuapp.com/predict/"
identif = str(id_client) 
url_req = url + identif

#résultat de la requête
predict = get_data(url_req)
proba_pred = predict['predictions']

#Affichage Crédit accepté/refusé
texte = "Loan for client ID : "+identif
if proba_pred < 0.52:
    texte = texte + "  ---> <span style='color:green;font-size:20px;'> APPROVED </span>"
    st.write(texte,unsafe_allow_html=True)
else:
    texte = texte + "  ---> <span style='color:red;font-size:20px;'> REFUSED </span>"
    st.write(texte,unsafe_allow_html=True)

#jauge de score de risque
fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = proba_pred,
    mode = "gauge+number+delta",
    title = {'text': "Risk of Failure"},
    delta = {'reference': 0.52, 
             'increasing':{'color':'red'},
             'decreasing':{'color':'green'}},
    gauge = {'axis': {'range': [None, 1]},
             'bar':{'color': "black"},
             'steps' : [
                 {'range': [0, 0.52], 'color': "green"},
                 {'range': [0.52, 1], 'color': "red"}],
             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 0.52}}))

st.plotly_chart(fig, use_container_width=True)

#affichage des informations détaillée du client sélectionné
with st.expander("Detailed customer information :"):
    st.write("Here you can see the detailed information of the customer :")
    st.write(clients.loc[id_client])

#récupération des shap_values de notre échantillon
shap_values = explainer(clients)
shap_base = shap_values.base_values.mean()

#index de l'ID client renseigné
idx = clients.index.get_loc(id_client)


#feature importance locale
waterfall = shap.plots.waterfall(shap_values[idx])

with st.expander("Details of the decision"):
    st.write("This graph shows the criteria that most influenced the algorithm's decision")
    st.pyplot(waterfall)
    st.write("<span style='color:Crimson;'>Criteria that put the client at risk of defaulting on the loan </span>", unsafe_allow_html=True)
    st.write("<span style='color:DodgerBlue;'>Criteria that make the client more likely to repay the loan </span>", unsafe_allow_html=True)


#feature importance globale
summary_plot = shap.summary_plot(shap_values, max_display=10)

with st.expander("Decision criteria of the algorithm"):

    st.pyplot(summary_plot)
    st.write('This graph shows the 10 features that have the most weight in all decisions of the algorithm')
    st.write('The horizontal axis shows the impact on the model decision (positive influence, on the right, or negative, on the left).')
    st.write('The color shows the value of the variable.')
    st.write('For exemple, for EXT_SOURCE_2, when the variable takes high values (red), it impacts negatively the model.')


#On récupère le 10 features les plus importantes 
feature_names = shap_values.feature_names
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
vals = np.abs(shap_df.values).mean(0)
shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
top_ten = shap_importance['col_name'].head(10).reset_index(drop=True)
top_ten = pd.DataFrame(top_ten)

#Plus d'informations
with st.expander("More details"):
    #liste pour séléctionner la 1ere feature 
    st.write('You can select 2 variables to see where the client stands in relation to others :')
    var_1 = st.selectbox('1st Feature :',top_ten)
    list_2=top_ten.drop(top_ten[top_ten['col_name']==var_1].index)
    var_2 = st.selectbox('2nd Feature :',list_2)

    st.write("Univariate Analysis & Client Positioning:")
    #var 1
    st.plotly_chart(distplots(clients_pred,var_1), use_container_width=True)
    #var 2
    st.plotly_chart(distplots(clients_pred,var_2), use_container_width=True)

    st.write("Bivariate analysis:")
    titre="Crossing of variables : "+var_1+" & "+var_2
    scat_plot = px.scatter(clients_pred, x=var_1, y=var_2, color="SCORE",
    title=titre, color_continuous_scale='rdylgn_r')
    st.plotly_chart(scat_plot, use_container_width=True)