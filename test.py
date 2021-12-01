import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

import requests

import numpy as np
import seaborn as sns
import joblib

import streamlit as st
import streamlit.components.v1 as components
st.set_option('deprecation.showPyplotGlobalUse', False)

import shap
shap.initjs()

#fonction requête API
def get_data(url):
    resp = requests.get(url)
    return resp.json()

#fonction shap plots
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

#chargement du modèle

test_path = os.path.dirname(os.path.abspath(__file__))
#st.write(test_path)
#st.write(test_path+'\modele.sav')
model = joblib.load(test_path+'\modele.sav')
explainer = joblib.load(test_path+'\explainer.sav')
#model = joblib.load('C:/Users/Raphaël/Documents/1_Formation_OC/P7/modele.sav')

#chargement des fichiers de travail
clients = pd.read_csv(test_path+'\sample.csv')
#clients = pd.read_csv('C:/Users/Raphaël/Documents/1_Formation_OC/P7/Data/sample.csv')
clients.set_index('SK_ID_CURR', inplace = True)
#clients_train = pd.read_csv('C:/Users/Raphaël/Documents/1_Formation_OC/P7/Data/clients_train.csv')
clients_pred = pd.read_csv(test_path+'\sample_pred.csv')
#clients_pred = pd.read_csv('C:/Users/Raphaël/Documents/1_Formation_OC/P7/Data/sample_pred.csv')
clients_pred.set_index('SK_ID_CURR', inplace = True)

#
st.title('Customer Dashboard : Loans')

#liste pour sélectionner un client
id_client = st.selectbox('Please select a Client ID :',clients.index )
id_client = int(id_client)


#url de requetage en fonction de l'ID client
url = "https://apipred.herokuapp.com/predict/"
identif = str(id_client) #"397043"
url_req = url + identif

#résultat de la requête
#st.write('hello')
predict = get_data(url_req)
proba_pred = predict['predictions']
#st.write(predict)
#st.write(predict['predictions'])

if proba_pred < 0.52:
    st.write("Crédit <span style='color:green;'>ACCEPTÉ </span>", 
unsafe_allow_html=True)
else:
    st.write("Crédit <span style='color:red;'> REFUSÉ </span>", 
unsafe_allow_html=True)

#jauge
fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = proba_pred,
    mode = "gauge+number+delta",
    title = {'text': "Risk"},
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

#shap
#explainer = shap.explainers.Linear(model, clients_train, feature_names=clients.columns)
shap_values = explainer(clients)

idx = clients.index.get_loc(id_client)

#feature importance locale
waterfall = shap.plots.waterfall(shap_values[idx])

st.write('Feature importance locale :')
st.write('Ce graphique présente la valeur des variables qui ont pesée le plus dans la décision de l\'algorithme')

st.pyplot(waterfall)

#feature importance globale
st.write('Feature importance globale :')
st.write('Ce graphique présente les variables qui ont le plus de poids dans toutes les décisions de l\'algorithme ')
summary_plot = shap.summary_plot(shap_values, max_display=10)
st.pyplot(summary_plot)

#Variables

#top10 features
feature_names = shap_values.feature_names
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
vals = np.abs(shap_df.values).mean(0)
shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)

top_ten = shap_importance['col_name'].head(10).reset_index(drop=True)
top_ten = pd.DataFrame(top_ten)

#plots features
def feats_plot(id_client, data, var):
    x=data.loc[id_client][var]
    #axarr = f.add_subplot(1,1,1)
    sns.kdeplot(data.loc[data['PRED'] == 0, var], label = 'TARGET == 0')
    sns.kdeplot(data.loc[data['PRED'] == 1, var], label = 'TARGET == 1').axvline(x)

var_1 = st.selectbox('Please select a feature :',top_ten)
#st.write(var_1, ' Séléctionnée !')
#st.write(clients_pred.loc[id_client][var_1])

#st.pyplot(feats_plot(id_client,clients_pred,var_1))

#st.write('Plotly :')
import plotly.figure_factory as ff
x1 = clients_pred.loc[clients_pred['PRED'] == 0, var_1]
x2 = clients_pred.loc[clients_pred['PRED'] == 1, var_1]
x=clients_pred.loc[id_client][var_1]

plot_v1 = ff.create_distplot([x1,x2], [0,1], show_hist=False, colors=['green','red']).add_vline(x,line_width=2,line_dash="dash",annotation_text="Client",annotation_font_size=18)
plot_v1.update_layout(height=600)
st.plotly_chart(plot_v1, use_container_width=True)


list_2=top_ten.drop(top_ten[top_ten['col_name']==var_1].index)

var_2 = st.selectbox('Please select a feature :',list_2)
#st.write(var_2, ' Séléctionnée !')

#st.pyplot(feats_plot(id_client,clients_pred,var_2))

x1 = clients_pred.loc[clients_pred['PRED'] == 0, var_2]
x2 = clients_pred.loc[clients_pred['PRED'] == 1, var_2]
x=clients_pred.loc[id_client][var_2]

plot_v2 = ff.create_distplot([x1,x2], [0,1], show_hist=False, colors=['green','red']).add_vline(x,line_width=2,line_dash="dash",annotation_text="Client",annotation_font_size=18)
plot_v2.update_layout(height=600)
st.plotly_chart(plot_v2, use_container_width=True)

#bivarié
color_map = plt.cm.get_cmap('RdYlGn')
col_map = color_map.reversed()

def biplot(v1,v2,data,colmap,hue):
    plt.scatter(data[v1],data[v2],c=data[hue],cmap=colmap)
    #plt.colorbar

#st.pyplot(biplot(var_1,var_2,clients_pred,col_map,'SCORE'))
import plotly.express as px

scat_plot = px.scatter(clients_pred, x=var_1, y=var_2, color="SCORE",
                 title="TITRE", color_continuous_scale='rdylgn_r')

st.plotly_chart(scat_plot, use_container_width=True)

#st.write(top_ten)


#st_shap(waterfall)