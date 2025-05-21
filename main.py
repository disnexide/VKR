import pandas as pd
import argparse
from preprocessing import preprocess_data
from detection import detect_anomalies, plot_anomaly_distribution, plot_anomaly_scatter, save_anomalies

def main(file_path, start_row, end_row):
    print(f"Загружаю данные из {file_path} с {start_row} по {end_row} строки...")
    data = pd.read_csv(file_path, skiprows=range(1, start_row), nrows=end_row - start_row)
    print("✅ Данные загружены.")

    data, columns_to_scale = preprocess_data(data)
    print("🔧 Предобработка завершена.")

    data, model = detect_anomalies(data, columns_to_scale)
    print("📌 Модель обучена, аномалии обнаружены.")

    print("📊 Строю графики...")
    plot_anomaly_distribution(data)
    plot_anomaly_scatter(data)

    anomalies = save_anomalies(data)
    print(f"💾 Аномалии сохранены. Обнаружено {len(anomalies)} аномалий из {len(data)} записей.")

    full_output = "all_transactions_with_anomalies.csv"
    data.to_csv(full_output, index=False)
    print(f"💾 Полный датасет с аномалиями сохранён в {full_output}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ транзакций на наличие аномалий")
    parser.add_argument("--file", type=str, required=True, help="Путь к CSV-файлу с транзакциями")
    parser.add_argument("--start", type=int, default=0, help="Начальная строка для анализа")
    parser.add_argument("--end", type=int, default=100_000, help="Конечная строка для анализа")

    args = parser.parse_args()
    main(args.file, args.start, args.end)