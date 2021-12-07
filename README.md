# Dashboard Streamlit
This is the source code used to create [this Dashboard](https://share.streamlit.io/boudh/dash/main/dashboard.py), hosted by Streamlit. <br>
I made this Dashboard for my Data Scientist training. <br>
The aim of the dashboard is to be used by bank advisors to help them to know if they can approve or reject loan applications of their customers. <br>

__Files :__
- dashboard.py : the main file, displaying the dashboard thanks to the streamlit library 
- explainer.sav : a SHAP explainer used to make SHAP plots
- sample.csv : a sample of the bank customer data
- sample_pred.csv : the same sample of the bank customer but with the target predict by my model

## How it works ?

First the user select a ID for a customer <br>

The dashboard will request the [API Pred](https://github.com/Boudh/api_pred) and get the probability that a client will default on a loan :
- if P < 0.52 --> Approve the loan
- Else --> Reject the loan application

The Bank advisor can clearly see the risk score on a gauge. <br> <br>
In order to maintain a good relationship with its customers the advisor has access to more informations. He will be able to explain more specifically the decision of the algorithm. <br><br>
Those informations are displayed thanks to SHAP plots : A plot for the global feature importance and a plot for the local feature importance.  <br><br>
The local feature importance will help the advisor to understand the decision for his customer.  <br><br>
Then, the advisor can see the position of his client in relation to other individuals on different variables, 2 of the 10 most important in total. 
