import pandas as pd
from sklearn.datasets import make_classification

# Generate a synthetic dataset (1000 samples, 20 features, 2 classes)
X, y = make_classification(
    n_samples=1000, 
    n_features=20, 
    n_informative=10, 
    n_redundant=5, 
    n_classes=2, 
    random_state=42
)

# Combine features (X) and target (y) into a single DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
df['target'] = y

# Save the data to the specified location
df.to_csv('data/data.csv', index=False)

print("Data saved to data/data.csv")