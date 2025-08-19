# Analisi Dati Consumo Energetico AEP

## Descrizione del Progetto
Questo progetto analizza i dati storici del consumo energetico di American Electric Power (AEP) per predire se il consumo sarà superiore o inferiore alla media giornaliera utilizzando tecniche di machine learning.

## Dataset
- **File**: `AEP_hourly.csv`
- **Dimensioni**: 121,273 record con dati orari
- **Periodo**: Dati storici del consumo energetico AEP
- **Colonne**:
  - `Datetime`: Timestamp orario
  - `AEP_MW`: Consumo energetico in megawatt

## Obiettivo
Sviluppare un modello di classificazione binaria per predire se il consumo energetico orario sarà:
- **1**: Alto consumo (sopra la media giornaliera)
- **0**: Basso consumo (sotto la media giornaliera)

## Struttura del Codice

### 1. 📊 Caricamento dei Dati
```python
# Caricamento dataset e conversione datetime
df = pd.read_csv("AEP_hourly.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.set_index("Datetime")
```

### 2. 🎯 Classificazione del Consumo
```python
# Calcolo media giornaliera e creazione target
daily_means = df["AEP_MW"].resample("D").mean()
df["target"] = (df["AEP_MW"] > df["daily_mean"]).astype(int)
```

### 3. 🔧 Feature Engineering
Estrazione di 3 features temporali:
- **hour**: Ora del giorno (0-23)
- **day_of_week**: Giorno della settimana (0-6)
- **month**: Mese dell'anno (1-12)

### 4. 📈 Train-Test Split
- **Training**: 70% dei dati (84,939 campioni)
- **Validation**: 15% dei dati (18,143 campioni)
- **Test**: 15% dei dati (18,191 campioni)
- Split stratificato per mantenere la distribuzione delle classi

### 5. 🤖 Addestramento del Modello
```python
# Decision Tree Classifier
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
tree.fit(X_train, y_train)
```

### 6. ✅ Cross Validation
- 5-fold Stratified Cross Validation
- Test su subset di 10,000 campioni per ottimizzazione dei tempi

## Risultati

### 🎯 Performance del Modello
- **Validation ROC-AUC**: 0.935 (93.5%)
- **Cross Validation AUC**: 0.935 ± 0.006
- **Stabilità**: Molto alta (deviazione standard di solo 0.6%)

### 📊 Interpretazione
Il modello dimostra **eccellenti performance** con:
- Alta accuratezza nel distinguere tra periodi di alto e basso consumo
- Stabilità elevata across different data splits
- Capacità predittiva robusta basata solo su features temporali

## Dipendenze
```
pandas
scikit-learn
```

## Come Eseguire
1. Assicurarsi di avere il file `AEP_hourly.csv` nella directory
2. Installare le dipendenze: `pip install pandas scikit-learn`
3. Eseguire: `python aep_data_analysis.py`

## Output Atteso
```
Data shape: (121273, 2)
First 5 rows:
                      AEP_MW
Datetime
2004-12-31 01:00:00  13478.0
2004-12-31 02:00:00  12865.0
...

(84939, 3) (18143, 3) (18191, 3)
Validation ROC-AUC: 0.9352287698981026
Eseguendo cross validation...
Test su subset dei dati (10000 campioni):
Decision Tree AUC (subset): 0.935 ± 0.006
```

## Problemi Risolti Durante lo Sviluppo

1. **Errore con `transform("mean")`**: 
   - **Problema**: ValueError con duplicate labels nel reindex
   - **Soluzione**: Implementato approach alternativo con mapping delle medie giornaliere

2. **Import errato**: 
   - **Problema**: `DecisionTreeClassifier` importato da `sklearn.ensemble`
   - **Soluzione**: Corretto import da `sklearn.tree`

3. **Cross validation lenta**: 
   - **Problema**: CV su 121K campioni troppo lenta
   - **Soluzione**: Implementato testing su subset di 10K campioni

## Conclusioni

Il modello Decision Tree con `max_depth=5` raggiunge performance eccellenti (AUC=93.5%) nella predizione del consumo energetico. Le features temporali (ora, giorno settimana, mese) sono sufficienti per catturare i pattern di consumo energetico, suggerendo una forte componente stagionale e circadiana nei dati.

