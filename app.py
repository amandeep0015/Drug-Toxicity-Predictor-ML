import streamlit as st
import joblib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the saved model
model = joblib.load('tox_model.pkl')

st.title("🧪 Drug Toxicity Predictor")
st.write("Enter a SMILES string to check for NR-AR Toxicity.")

smiles_input = st.text_input("SMILES String", "CC(=O)OC1=CC=CC=C1C(=O)O") # Default is Aspirin

if st.button("Predict Toxicity"):
    mol = Chem.MolFromSmiles(smiles_input)
    if mol:
        # Convert input to fingerprint
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        features = np.array(list(fp)).reshape(1, -1)
        
        # Predict
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        if prediction == 1:
            st.error(f"Prediction: TOXIC (Confidence: {probability*100:.1f}%)")
        else:
            st.success(f"Prediction: NON-TOXIC (Confidence: {(1-probability)*100:.1f}%)")
    else:
        st.warning("Invalid SMILES string. Please check the structure.")