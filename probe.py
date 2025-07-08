import pandas as pd

# -------------------------------
# 1️⃣ Daten laden
# -------------------------------

traffic_files = [
    'sensordata_traffic_2024-09-09_2024-09-12.json',
    'sensordata_traffic_2024-09-13_2024-09-15.json',
    'sensordata_traffic_2024-09-20_2024-09-26.json'
]

df_traffic_list = []
for file in traffic_files:
    df_raw = pd.read_json(file)
    df = pd.json_normalize(df_raw["sensordata"])
    df_traffic_list.append(df)

df_traffic = pd.concat(df_traffic_list, ignore_index=True)
df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
df_traffic['date'] = df_traffic['timestamp'].dt.date

env_files = [
    'sensordata_environment_2024-09-09_2024-09-15.json',
    'sensordata_environment_2024-09-20_2024-09-26.json'
]

df_env_list = []
for file in env_files:
    df_raw = pd.read_json(file)
    df = pd.json_normalize(df_raw["sensordata"])
    df_env_list.append(df)

df_env = pd.concat(df_env_list, ignore_index=True)
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

# -------------------------------
# 2️⃣ Schadstoffe definieren
# -------------------------------
sensor_labels = {
    'dust_pm1': 'PM1 (µg/m³)',
    'dust_pm25': 'PM2.5 (µg/m³)',
    'dust_pm10': 'PM10 (µg/m³)',
    'no2': 'NO₂ (ppb)',
    'co': 'CO (ppb)',
    'temperature': 'Temperatur (°C)'
}

env_daily = []
for label, name in sensor_labels.items():
    df_label = df_env[df_env['sensorLabel'] == label]
    if not df_label.empty:
        daily_avg = df_label.groupby('date')['value'].mean().reset_index(name=name)
        env_daily.append(daily_avg)

if not env_daily:
    print("❌ Keine Umweltdaten verfügbar.")
    exit()

from functools import reduce
df_env_merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), env_daily)

# -------------------------------
# 3️⃣ Fahrzeuganzahl je Straße und Tag
# -------------------------------
if 'location' in df_traffic.columns:
    traffic_per_street = df_traffic.groupby(['location', 'date']).size().reset_index(name='Fahrzeuganzahl')
else:
    print("❌ Keine 'location' in Traffic-Daten.")
    exit()

# -------------------------------
# 4️⃣ Korrelationen je Straße berechnen
# -------------------------------
results = []

for street, group in traffic_per_street.groupby('location'):
    merged = pd.merge(group, df_env_merged, on='date', how='inner')
    if len(merged) < 3:
        continue

    row = {'Straße': street}
    for pollutant in sensor_labels.values():
        if pollutant in merged.columns:
            corr = merged['Fahrzeuganzahl'].corr(merged[pollutant])
            row[f'Korr Verkehr vs {pollutant}'] = round(corr, 3) if pd.notna(corr) else ''
        else:
            row[f'Korr Verkehr vs {pollutant}'] = ''
    results.append(row)

# -------------------------------
# 5️⃣ Tabelle ausgeben
# -------------------------------
df_results = pd.DataFrame(results)

if not df_results.empty:
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 150)
    print("\n✅ Korrelationen je Straße:")
    print(df_results)

    # CSV Export (Excel lesbar)
    df_results.to_csv("streetwise_correlations.csv", index=False)
    print("\n✅ Gespeichert als 'streetwise_correlations.csv'.")
else:
    print("⚠️ Keine ausreichenden Daten für Korrelationen vorhanden.")
