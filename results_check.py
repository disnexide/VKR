import pandas as pd
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from preprocessing import preprocess_data  # ваш модуль предобработки

# Параметры
RAW_PATH        = 'transactions_data_preprocessed.csv'
ALL_PATH        = 'C:/Users/Катюша/Desktop/anomaly_detector/all_transactions_with_anomalies.csv'
ANOM_PATH       = 'C:/Users/Катюша/Desktop/anomaly_detector/anomalies.csv'
LABELS_JSON     = 'train_fraud_labels.json'
START_ROW       = 1
END_ROW         = 1000000

# 1. Загружаем сырые данные и берём срез
df_raw = pd.read_csv(RAW_PATH).iloc[START_ROW:END_ROW].reset_index(drop=True)

# 2. Применяем ту же предобработку, что и при обучении
df_proc, cols_to_scale = preprocess_data(df_raw)

# 3. Берём только ID тех строк, что действительно прошли preprocess_data
processed_ids = set(df_proc['id'])

# 4. Загружаем весь файл с anomaly_score и фильтруем по processed_ids
all_df = pd.read_csv(ALL_PATH, usecols=['id'])
all_df = all_df[all_df['id'].isin(processed_ids)]

# 5. Загружаем предсказанные аномалии и пересекаем с processed_ids
anom_df      = pd.read_csv(ANOM_PATH, usecols=['id'])
predicted_ids = set(anom_df['id']) & processed_ids

# 6. Загружаем истинные метки fraud и фильтруем по processed_ids
with open(LABELS_JSON, 'r', encoding='utf-8') as f:
    targets = json.load(f)['target']
labels = [(int(k), 1 if v.lower()=='yes' else 0)
          for k, v in targets.items()
          if int(k) in processed_ids]
labels_df = pd.DataFrame(labels, columns=['id','isFraud'])

# 7. Собираем единую таблицу и рассчитываем метрики
df_eval = all_df.merge(labels_df, on='id', how='inner')
df_eval['predicted'] = df_eval['id'].isin(predicted_ids).astype(int)

y_true = df_eval['isFraud']
y_pred = df_eval['predicted']

cm   = confusion_matrix(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec  = recall_score(y_true, y_pred, zero_division=0)
f1   = f1_score(y_true, y_pred, zero_division=0)

# 8. Вывод
print(f"Анализ на {len(df_eval)} записях после предобработки:")
print("Матрица ошибок:")
print(cm)
print(f"\nPrecision = {prec:.4f}")
print(f"Recall    = {rec:.4f}")
print(f"F1-score  = {f1:.4f}")