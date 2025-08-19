import pandas as pd

##1 Caricamento dei dati
df = pd.read_csv("c:/Users/QU344RM/OneDrive - EY/Documents/GitHub/depositoRremilli/19.08/esercizio1/AEP_hourly.csv")
print(f"Data shape: {df.shape}")

# Conversione Datetime to datetime format
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime")

print("\nFirst 5 rows:")
print(df.head())

##2 Classicazione del consumo rispetto alla media giornaliera
# Calcola la media giornaliera e merge con il dataframe originale
daily_means = df["AEP_MW"].resample("D").mean()
df["daily_mean"] = df.index.date
df["daily_mean"] = df["daily_mean"].map(daily_means.to_dict())

# target: 1 = alto consumo, 0 = basso consumo
df["target"] = (df["AEP_MW"] > df["daily_mean"]).astype(int)

##3 Feature Engineering
df["hour"] = df.index.hour
df["day_of_week"] = df.index.dayofweek 
df["month"] = df.index.month

X = df[["hour", "day_of_week", "month"]]
y = df["target"]

##4 Train-Test Split
from sklearn.model_selection import train_test_split
#prima prepariamo il test 
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

#separiamo il validation
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, stratify=y_temp, random_state=42
)

print(X_train.shape, X_val.shape, X_test.shape)

##5- Addestramento del modello
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)

# valutazione
y_val_pred = tree.predict_proba(X_val)[:,1]
auc_val = roc_auc_score(y_val, y_val_pred)
print("Validation ROC-AUC:", auc_val)

##6- Cross Validation
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Cross validation su tutto il dataset (può essere lenta)
print("Eseguendo cross validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Per test più veloce, usiamo un subset dei dati
print("Test su subset dei dati (10000 campioni):")
subset_indices = X.sample(n=10000, random_state=42).index
X_subset = X.loc[subset_indices]
y_subset = y.loc[subset_indices]

cv_auc_subset = cross_val_score(tree, X_subset, y_subset, cv=skf, scoring="roc_auc")
print(f"Decision Tree AUC (subset): {cv_auc_subset.mean():.3f} ± {cv_auc_subset.std():.3f}")

# Cross validation completa (commentata per velocità)
# cv_auc = cross_val_score(tree, X, y, cv=skf, scoring="roc_auc")
# print(f"Decision Tree AUC (completo): {cv_auc.mean():.3f} ± {cv_auc.std():.3f}")