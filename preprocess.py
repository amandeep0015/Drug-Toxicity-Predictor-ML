import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Load the dataset
print("Loading Tox21 dataset...")
df = pd.read_csv('tox21.csv')

# Use only one label for speed (e.g., 'NR-AR') or use more if you want
# Let's take 'NR-AR' as the primary toxicity target
target_col = 'NR-AR'

# Converter function
def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        # Radius 2 (ECFP4) is the industry standard for toxicity
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return list(fp)
    return None

print("Converting molecules to fingerprints... (takes ~1 min)")
df['fp'] = df['smiles'].apply(smiles_to_fp)

# Clean up: Remove rows where conversion failed or toxicity data is missing
df = df.dropna(subset=['fp', target_col])

# Prepare X (features) and y (target)
X = np.array(df['fp'].tolist())
y = df[target_col].values

# Save for the next step
np.save('X.npy', X)
np.save('y.npy', y)
print(f"Done! Processed {len(df)} molecules.")