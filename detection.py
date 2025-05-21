from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def detect_anomalies(data, columns_to_scale, contamination=0.0015):
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    data['anomaly_score'] = model.fit_predict(data[columns_to_scale])
    return data, model

def plot_anomaly_distribution(data):
    plt.figure(figsize=(8, 6))
    sns.histplot(data, x='amount', kde=True, bins=50, hue=data['anomaly_score'].astype(str))
    plt.title("Распределение сумм транзакций с аномалиями")
    plt.xlabel("Сумма транзакции")
    plt.ylabel("Частота")
    plt.show()

def plot_anomaly_scatter(data):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='amount', y='transaction_count_by_client', hue='anomaly_score', palette={1: 'blue', -1: 'red'})
    plt.title("Аномалии по сумме и количеству транзакций")
    plt.xlabel("Сумма транзакции")
    plt.ylabel("Количество транзакций по клиенту")
    plt.show()

def save_anomalies(data, output_path="anomalies.csv"):
    anomalies = data[data['anomaly_score'] == -1]
    anomalies.to_csv(output_path, index=False)
    return anomalies

def detect_and_save_all(data_path,
                        columns_to_scale,
                        output_path="all_transactions_with_anomalies.csv",
                        contamination=0.0015):
    data = pd.read_csv(data_path)
    annotated_data, model = detect_anomalies(data,
                                             columns_to_scale,
                                             contamination)
    annotated_data.to_csv(output_path, index=False)
    return annotated_data, model