import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
#import lime.lime_tabular import LimeTabularExplainer
model = joblib.load('ET.pkl')
X_test =pd.read_csv('X_test.csv')
feature_names = [
    "UA",
    "Age",
    "LDH",
    "GFR",
    "D-Dimer",
    "CA",
    "hs-cTnl",
    "HGB",
    "ALB"
]
st.title("Cardiovascular disease predictor for DN")
ua = st.number_input("Blood uric acid：",min_value=-10,max_value=1000,value=425)
age = st.number_input("Age：",min_value=10,max_value=90,value=75)
ldh = st.number_input("Lactate dehydrogenase：",min_value=-10,max_value=1000,value=216)
gfe = st.number_input("Glomerular filtration rate：",min_value=-10,max_value=1000,value=45)
d_dimer = st.number_input("D-Dimer：",min_value=-1.0,max_value=10.0,value=1.03)
ca = st.number_input("Blood calcium：",min_value=-1.00,max_value=10.00,value=2.53)
hs_cTnl = st.number_input("Hypersensitive troponin I：",min_value=-10.0,max_value=1000.0,value=20.7)
hgb = st.number_input("haemoglobin：",min_value=-10,max_value=1000,value=103)
alb = st.number_input("haemoglobin：",min_value=-10.0,max_value=1000.0,value=35.8)

feature_values = [ua,age,ldh,gfe,d_dimer,ca,hs_cTnl,hgb,alb]
features =np.array([feature_values])
if st.button("Predict"):
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[1]

    st.write(f"**Predicted Class:**{predicted_class}(1:yes,0:no)")
    st.write(f"- Low risk probability: {predicted_proba[0]:.1%}")
    st.write(f"- High risk probability: {predicted_proba[1]:.1%}")
    st.write(f"**Predicted Probabilities:**{predicted_proba}")
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of heart disease."
            f"The model predicts that your probability of having heart disease is {predicted_proba[1]:.1%}%"
            "It's adviced to consult with your healthcare provider for further evaluation and possible intervention"
        )
    else:
        advice = (
            f"According to our model, you have a low risk of heart disease."
            f"The model predicts that your probability of not having heart disease is {predicted_proba[0]:.1%}%"
            "However, Don't take your physical health lightly.Please continue regular check-ups with your healthcare provider"
        )
    st.write(advice)

    st.subheader("SHAP Force plot Explanation")
    explainer_shap = shap.TreeExplainer(model)
    shap_values = explainer_shap.shap_values(pd.DataFrame([feature_values],columns=feature_names))

    if predicted_class == 1:
        shap.force_plot(explainer_shap.expected_value[1],shap_values[:,:,1],pd.DataFrame([feature_values],columns=feature_names),matplotlib=True)
    else:
        shap.force_plot(explainer_shap.expected_value[0], shap_values[:, :,0],pd.DataFrame([feature_values],columns=feature_names),matplotlib=True)
    plt.savefig("shap_force_plot.png",bbox_inches ='tight',dpi=1200)
    st.iamge("shap_force_plot.png",caption='SHAP Force Plot Explanation')

    #st.subheader("LIME Explanation")
    #lime_explainer = LimeTabularExplainer(
    #    training_data = X_test.values,
     #   feature_names = X_test.columns.tolist(),
      #  class_names=['no','yes'],
       # mode='classification'

    #)
    #lime_exp = lime_explainer.explain_instance(
     #   data_row=features.flatten(),
      #  predicted_fn=model.predict_probe

    #)
   # lime_html = lime_exp.as_html(show_table=False)
    #st.components.v1.html(lime_html,height=800,scrolling=True)
