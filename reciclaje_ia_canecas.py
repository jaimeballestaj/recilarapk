# reciclaje_ia_canecas.py — versión sin joblib ni .pkl para Streamlit Cloud

import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Datos base de entrenamiento
residuos = pd.DataFrame({
    'residuo': [
        'botella plastica', 'cascara de banana', 'papel', 'lata', 'vaso icopor', 'envoltura metalizada',
        'carton', 'pañal usado', 'vidrio', 'bolsa plastica', 'restos de comida', 'revista'
    ],
    'tipo': [
        'reciclable', 'organico', 'reciclable', 'reciclable', 'no reciclable', 'no reciclable',
        'reciclable', 'no reciclable', 'reciclable', 'reciclable', 'organico', 'reciclable'
    ]
})

mapa_caneca = {
    'reciclable': 'azul',
    'organico': 'verde',
    'no reciclable': 'gris'
}
residuos['caneca'] = residuos['tipo'].map(mapa_caneca)

residuos['es_plastico'] = residuos['residuo'].str.contains("plast|icop|bolsa|envoltura").astype(int)
residuos['es_organico'] = residuos['residuo'].str.contains("banana|comida|cascara").astype(int)
residuos['es_papel'] = residuos['residuo'].str.contains("papel|revista|carton").astype(int)
residuos['es_vidrio'] = residuos['residuo'].str.contains("vidrio").astype(int)
residuos['es_metal'] = residuos['residuo'].str.contains("lata").astype(int)

X = residuos[['es_plastico', 'es_organico', 'es_papel', 'es_vidrio', 'es_metal']]
y = residuos['caneca']

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X, y)

# Streamlit App
st.set_page_config(page_title="Reciclaje Inteligente", page_icon="♻️", layout="centered")
st.title("♻️ Asistente de Reciclaje con IA")
st.markdown("Aprende en qué caneca depositar correctamente tus residuos.")

residuo_input = st.text_input("¿Qué residuo quieres clasificar?", "botella plastica")

color_map = {
    "azul": ("♻️ El residuo debe ir en la caneca AZUL (reciclaje)", "#0066CC"),
    "verde": ("🌿 El residuo debe ir en la caneca VERDE (orgánico)", "#33A532"),
    "gris": ("🗑️ El residuo debe ir en la caneca GRIS (no reciclable)", "#555555"),
    "roja": ("⚠️ ¡Este residuo es PELIGROSO y debe ir en la caneca ROJA!", "#B30000")
}

if st.button("Clasificar"):
    entrada = {
        'es_plastico': int(any(p in residuo_input.lower() for p in ['plast', 'icop', 'bolsa', 'envoltura'])),
        'es_organico': int(any(o in residuo_input.lower() for o in ['banana', 'comida', 'cascara', 'manzana'])),
        'es_papel': int(any(p in residuo_input.lower() for p in ['papel', 'revista', 'carton'])),
        'es_vidrio': int('vidrio' in residuo_input.lower()),
        'es_metal': int('lata' in residuo_input.lower())
    }
    entrada_df = pd.DataFrame([entrada])
    pred = modelo.predict(entrada_df)[0]

    if any(p in residuo_input.lower() for p in ['jeringa', 'batería', 'medicamento', 'químico', 'peligroso']):
        mensaje, color = color_map['roja']
    else:
        mensaje, color = color_map[pred]

    st.markdown(
        f"""
        <div style='background-color:{color}; padding: 1rem; border-radius: 0.5rem;'>
            <h4 style='color:white'>{mensaje}</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.image("https://cdn-icons-png.flaticon.com/512/992/992700.png", width=100)

# Ejecutar:
# python -m streamlit run reciclaje_ia_canecas.py
