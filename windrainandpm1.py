import pandas as pd
import matplotlib.pyplot as plt

# Environment-Daten laden
df_env_raw = pd.read_json("sensordata_environment_2024-09-20_2024-09-26.json")
df_env = pd.json_normalize(df_env_raw["sensordata"])
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

# Zeitraum setzen
start_date = pd.to_datetime('2024-09-20').date()
end_date = pd.to_datetime('2024-09-26').date()
df_env_week = df_env[(df_env['date'] >= start_date) & (df_env['date'] <= end_date)]

# Tagesmittel Windgeschwindigkeit
wind = df_env_week[df_env_week['sensorLabel'] == 'windspeed']
wind_daily = wind.groupby('date')['value'].mean().reset_index(name='windspeed')

# Tagesmittel Feinstaub
pm_labels = {'dust_pm1': 'PM1', 'dust_pm10': 'PM10', 'dust_pm25': 'PM2.5'}
pm_dfs = []
for label, col_name in pm_labels.items():
    pm = df_env_week[df_env_week['sensorLabel'] == label]
    pm_daily = pm.groupby('date')['value'].mean().reset_index(name=col_name)
    pm_dfs.append(pm_daily)

# Zusammenführen
df_merged = wind_daily
for df in pm_dfs:
    df_merged = pd.merge(df_merged, df, on='date')

print("\n📊 Übersicht Windgeschwindigkeit und Feinstaub pro Tag:")
print(df_merged)

# Korrelationen berechnen
for col in ['PM1', 'PM10', 'PM2.5']:
    corr = df_merged['windspeed'].corr(df_merged[col])
    print(f"🔗 Korrelation Windgeschwindigkeit vs. {col}: {corr:.3f}")

# Scatterplots
for col in ['PM1', 'PM10', 'PM2.5']:
    plt.figure(figsize=(8,6))
    plt.scatter(df_merged['windspeed'], df_merged[col], s=120, alpha=0.7)
    for idx, row in df_merged.iterrows():
        plt.annotate(str(row['date']), 
                     (row['windspeed'], row[col]),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    corr = df_merged['windspeed'].corr(df_merged[col])
    plt.title(f"{col} vs. Windgeschwindigkeit\nKorrelation: {corr:.3f}")
    plt.xlabel("Windgeschwindigkeit (m/s)")
    plt.ylabel(f"{col} (μg/m³)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Traffic-Daten vorbereiten
# Lade beide Dateien und kombiniere
df_traffic_raw1 = pd.read_json("sensordata_traffic_2024-09-09_2024-09-12.json")
df_traffic1 = pd.json_normalize(df_traffic_raw1["sensordata"])
df_traffic_raw2 = pd.read_json("sensordata_traffic_2024-09-13_2024-09-15.json")
df_traffic2 = pd.json_normalize(df_traffic_raw2["sensordata"])
df_traffic = pd.concat([df_traffic1, df_traffic2], ignore_index=True)

df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
df_traffic['date'] = df_traffic['timestamp'].dt.date

# Zeitraum
start_date = pd.to_datetime('2024-09-09').date()
end_date = pd.to_datetime('2024-09-15').date()

# Filter
df_traffic_week = df_traffic[(df_traffic['date'] >= start_date) & (df_traffic['date'] <= end_date)]

# Fahrzeuganzahl pro Tag
autos_per_day = df_traffic_week.groupby('date').size().reset_index(name='anzahl_autos')
print("\n🚗 Fahrzeuganzahl pro Tag:")
print(autos_per_day)

# ----------------------------
# 2️⃣ Environment-Daten vorbereiten
df_env_raw = pd.read_json("sensordata_environment_2024-09-09_2024-09-15.json")
df_env = pd.json_normalize(df_env_raw["sensordata"])
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

# Filter
df_env_week = df_env[(df_env['date'] >= start_date) & (df_env['date'] <= end_date)]

# ----------------------------
# 3️⃣ Tagesdurchschnittswerte für Messgrößen
sensor_labels = {
    'dust_pm1': 'PM1 (μg/m³)',
    'dust_pm10': 'PM10 (μg/m³)',
    'dust_pm25': 'PM2.5 (μg/m³)',
    'no2': 'NO₂ (ppb)',
    'co': 'CO (ppb)',
    'temperature': 'Temperatur (°C)',
    'windspeed': 'Windgeschwindigkeit (m/s)'
}

env_summary = []

for label, description in sensor_labels.items():
    df_sensor = df_env_week[df_env_week['sensorLabel'] == label]
    if not df_sensor.empty:
        daily_avg = df_sensor.groupby('date')['value'].mean().reset_index(name=description)
        env_summary.append(daily_avg)
    else:
        print(f"⚠️ Keine Daten für {label} vorhanden.")

# Zusammenführen
df_env_merged = env_summary[0]
for df in env_summary[1:]:
    df_env_merged = pd.merge(df_env_merged, df, on='date', how='outer')

# ----------------------------
# 4️⃣ Zusammenführen mit Fahrzeugdaten
combined = pd.merge(autos_per_day, df_env_merged, on='date')
combined = combined.sort_values('date')

print("\n📊 Übersicht aller Werte vom 09.09. bis 15.09.2024:")
print(combined)

# ----------------------------
# 5️⃣ Korrelationen berechnen
print("\n🔗 Korrelationen mit Fahrzeuganzahl:")
for col in combined.columns[2:]:
    corr = combined['anzahl_autos'].corr(combined[col])
    print(f"Korrelation Fahrzeuganzahl vs. {col}: {corr:.3f}")

print("\n🔗 Korrelationen Windgeschwindigkeit vs. Feinstaub:")
for col in ['PM1 (μg/m³)', 'PM10 (μg/m³)', 'PM2.5 (μg/m³)']:
    corr = combined['Windgeschwindigkeit (m/s)'].corr(combined[col])
    print(f"Korrelation Windgeschwindigkeit vs. {col}: {corr:.3f}")

# ----------------------------
# 6️⃣ Scatterplots
colors = {
    'PM1 (μg/m³)': 'gray',
    'PM10 (μg/m³)': 'orange',
    'PM2.5 (μg/m³)': 'brown',
    'NO₂ (ppb)': 'green',
    'CO (ppb)': 'red',
    'Temperatur (°C)': 'blue',
    'Windgeschwindigkeit (m/s)': 'purple'
}

# Fahrzeuganzahl vs Messgrößen
for col in combined.columns[2:]:
    plt.figure(figsize=(8,6))
    plt.scatter(combined['anzahl_autos'], combined[col], color=colors.get(col, 'black'), s=120, alpha=0.7)
    for idx, row in combined.iterrows():
        plt.annotate(str(row['date']), 
                     (row['anzahl_autos'], row[col]),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    corr = combined['anzahl_autos'].corr(combined[col])
    plt.title(f"{col} vs. Fahrzeuganzahl\nKorrelation: {corr:.3f}")
    plt.xlabel("Fahrzeuganzahl pro Tag")
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Windgeschwindigkeit vs Feinstaub
for col in ['PM1 (μg/m³)', 'PM10 (μg/m³)', 'PM2.5 (μg/m³)']:
    plt.figure(figsize=(8,6))
    plt.scatter(combined['Windgeschwindigkeit (m/s)'], combined[col], color=colors.get(col, 'black'), s=120, alpha=0.7)
    for idx, row in combined.iterrows():
        plt.annotate(str(row['date']), 
                     (row['Windgeschwindigkeit (m/s)'], row[col]),
                     textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    corr = combined['Windgeschwindigkeit (m/s)'].corr(combined[col])
    plt.title(f"{col} vs. Windgeschwindigkeit\nKorrelation: {corr:.3f}")
    plt.xlabel("Windgeschwindigkeit (m/s)")
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Funktion zur Datenvorbereitung für einen Zeitraum
def prepare_data(traffic_files, env_file, start_date, end_date, label):
    # Traffic
    df_traffic = pd.concat(
        [pd.json_normalize(pd.read_json(file)["sensordata"]) for file in traffic_files],
        ignore_index=True
    )
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
    df_traffic['date'] = df_traffic['timestamp'].dt.date
    df_traffic_week = df_traffic[
        (df_traffic['date'] >= start_date) & (df_traffic['date'] <= end_date)
    ]
    autos_per_day = df_traffic_week.groupby('date').size().reset_index(name='anzahl_autos')

    # Environment
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date
    df_env_week = df_env[
        (df_env['date'] >= start_date) & (df_env['date'] <= end_date)
    ]

    # Feinstaub
    pm_labels = {'dust_pm1': 'PM1 (μg/m³)', 'dust_pm10': 'PM10 (μg/m³)', 'dust_pm25': 'PM2.5 (μg/m³)'}
    env_summary = []
    for label_key, label_name in pm_labels.items():
        df_sensor = df_env_week[df_env_week['sensorLabel'] == label_key]
        if not df_sensor.empty:
            daily_avg = df_sensor.groupby('date')['value'].mean().reset_index(name=label_name)
            env_summary.append(daily_avg)

    df_env_merged = env_summary[0]
    for df in env_summary[1:]:
        df_env_merged = pd.merge(df_env_merged, df, on='date', how='outer')

    # Merge
    combined = pd.merge(autos_per_day, df_env_merged, on='date')
    combined['Zeitraum'] = label

    return combined

# ---------------------------
# Daten für 09.–15.09.2024
combined_1 = prepare_data(
    ["sensordata_traffic_2024-09-09_2024-09-12.json", "sensordata_traffic_2024-09-13_2024-09-15.json"],
    "sensordata_environment_2024-09-09_2024-09-15.json",
    pd.to_datetime('2024-09-09').date(),
    pd.to_datetime('2024-09-15').date(),
    "09.–15.09."
)

# Daten für 20.–26.09.2024
combined_2 = prepare_data(
    ["sensordata_traffic_2024-09-20_2024-09-26.json"],
    "sensordata_environment_2024-09-20_2024-09-26.json",
    pd.to_datetime('2024-09-20').date(),
    pd.to_datetime('2024-09-26').date(),
    "20.–26.09."
)

# ---------------------------
# Zusammenführen
df_all = pd.concat([combined_1, combined_2], ignore_index=True)

# ---------------------------
# Plot: Fahrzeuganzahl vs. PM1, PM10, PM2.5 im Vergleich
colors = {"09.–15.09.": "blue", "20.–26.09.": "orange"}

for pollutant in ["PM1 (μg/m³)", "PM10 (μg/m³)", "PM2.5 (μg/m³)"]:
    plt.figure(figsize=(9,6))
    for label, color in colors.items():
        subset = df_all[df_all['Zeitraum'] == label]
        plt.scatter(subset['anzahl_autos'], subset[pollutant],
                    color=color, label=label, s=120, alpha=0.7)
        for idx, row in subset.iterrows():
            plt.annotate(str(row['date']), (row['anzahl_autos'], row[pollutant]),
                         textcoords="offset points", xytext=(0,8),
                         ha='center', fontsize=8)
    plt.title(f"{pollutant} vs. Fahrzeuganzahl im Vergleich")
    plt.xlabel("Fahrzeuganzahl pro Tag")
    plt.ylabel(pollutant)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt

print("Nun Regenzusammenhangsanalyse")
# ---------------------------
# Funktion zur Regen-Analyse für einen Zeitraum
def analyze_rain(traffic_files, env_file, start_date, end_date, label):
    # Environment-Daten laden
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date
    df_env_week = df_env[(df_env['date'] >= start_date) & (df_env['date'] <= end_date)]

    # Regen
    rain = df_env_week[df_env_week['sensorLabel'] == 'rainintensity']
    rain_daily = rain.groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')

    # Schadstoffe
    labels = {
        'dust_pm1': 'PM1 (μg/m³)',
        'dust_pm10': 'PM10 (μg/m³)',
        'dust_pm25': 'PM2.5 (μg/m³)',
        'no2': 'NO₂ (ppb)',
        'co': 'CO (ppb)'
    }
    env_summary = []
    for label_key, label_name in labels.items():
        df_sensor = df_env_week[df_env_week['sensorLabel'] == label_key]
        if not df_sensor.empty:
            daily_avg = df_sensor.groupby('date')['value'].mean().reset_index(name=label_name)
            env_summary.append(daily_avg)

    df_merged = rain_daily
    for df in env_summary:
        df_merged = pd.merge(df_merged, df, on='date', how='outer')

    df_merged['Zeitraum'] = label

    print(f"\n📊 Übersicht Regen und Schadstoffe vom {start_date} bis {end_date}:")
    print(df_merged)

    # Korrelationen
    print(f"\n🔗 Korrelationen mit Regenintensität ({label}):")
    for col in labels.values():
        if col in df_merged.columns:
            corr = df_merged['Regenintensität (mm/h)'].corr(df_merged[col])
            print(f"Korrelation Regen vs. {col}: {corr:.3f}")

    # Scatterplots
    for col in labels.values():
        if col in df_merged.columns:
            plt.figure(figsize=(8,6))
            plt.scatter(df_merged['Regenintensität (mm/h)'], df_merged[col], s=120, alpha=0.7)
            for idx, row in df_merged.iterrows():
                plt.annotate(str(row['date']), (row['Regenintensität (mm/h)'], row[col]),
                             textcoords="offset points", xytext=(0,8),
                             ha='center', fontsize=8)
            corr = df_merged['Regenintensität (mm/h)'].corr(df_merged[col])
            plt.title(f"{col} vs. Regenintensität ({label})\nKorrelation: {corr:.3f}")
            plt.xlabel("Regenintensität (mm/h)")
            plt.ylabel(col)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

# ---------------------------
# Analyse für 09.–15.09.2024
analyze_rain(
    ["sensordata_traffic_2024-09-09_2024-09-12.json", "sensordata_traffic_2024-09-13_2024-09-15.json"],
    "sensordata_environment_2024-09-09_2024-09-15.json",
    pd.to_datetime('2024-09-09').date(),
    pd.to_datetime('2024-09-15').date(),
    "09.–15.09."
)

# Analyse für 20.–26.09.2024
analyze_rain(
    ["sensordata_traffic_2024-09-20_2024-09-26.json"],
    "sensordata_environment_2024-09-20_2024-09-26.json",
    pd.to_datetime('2024-09-20').date(),
    pd.to_datetime('2024-09-26').date(),
    "20.–26.09."
)
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Funktion für Analyse
def analyze_rain_traffic_airquality(traffic_files, env_file, start_date, end_date, label):
    # Traffic
    df_traffic = pd.concat(
        [pd.json_normalize(pd.read_json(file)["sensordata"]) for file in traffic_files],
        ignore_index=True
    )
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
    df_traffic['date'] = df_traffic['timestamp'].dt.date
    df_traffic_week = df_traffic[
        (df_traffic['date'] >= start_date) & (df_traffic['date'] <= end_date)
    ]
    autos_per_day = df_traffic_week.groupby('date').size().reset_index(name='Fahrzeuganzahl')

    # Environment
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date
    df_env_week = df_env[
        (df_env['date'] >= start_date) & (df_env['date'] <= end_date)
    ]

    # Regen
    rain = df_env_week[df_env_week['sensorLabel'] == 'rainintensity']
    rain_daily = rain.groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')

    # Schadstoffe
    labels = {
        'dust_pm1': 'PM1 (μg/m³)',
        'dust_pm10': 'PM10 (μg/m³)',
        'dust_pm25': 'PM2.5 (μg/m³)',
        'no2': 'NO₂ (ppb)',
        'co': 'CO (ppb)'
    }
    env_summary = []
    for label_key, label_name in labels.items():
        df_sensor = df_env_week[df_env_week['sensorLabel'] == label_key]
        if not df_sensor.empty:
            daily_avg = df_sensor.groupby('date')['value'].mean().reset_index(name=label_name)
            env_summary.append(daily_avg)

    df_merged = rain_daily
    for df in env_summary:
        df_merged = pd.merge(df_merged, df, on='date', how='outer')

    # Zusammenführen
    combined = pd.merge(df_merged, autos_per_day, on='date', how='outer')
    combined['Zeitraum'] = label
    combined = combined.sort_values('date')

    print(f"\n📊 Übersicht Regen, Verkehr, Luftqualität ({label}):")
    print(combined)

    # --------------------------
    # Korrelationen prüfen
    print(f"\n🔗 Korrelationen ({label}):")

    # Regen vs. Luftqualität
    for pollutant in labels.values():
        if pollutant in combined.columns:
            corr = combined['Regenintensität (mm/h)'].corr(combined[pollutant])
            print(f"Regen vs. {pollutant}: {corr:.3f}")

    # Regen vs. Fahrzeuganzahl
    corr = combined['Regenintensität (mm/h)'].corr(combined['Fahrzeuganzahl'])
    print(f"Regen vs. Fahrzeuganzahl: {corr:.3f}")

    # Fahrzeuganzahl vs. Luftqualität
    for pollutant in labels.values():
        if pollutant in combined.columns:
            corr = combined['Fahrzeuganzahl'].corr(combined[pollutant])
            print(f"Fahrzeuganzahl vs. {pollutant}: {corr:.3f}")

# ---------------------------
# Analyse für **09.–15.09.2024**
analyze_rain_traffic_airquality(
    ["sensordata_traffic_2024-09-09_2024-09-12.json", "sensordata_traffic_2024-09-13_2024-09-15.json"],
    "sensordata_environment_2024-09-09_2024-09-15.json",
    pd.to_datetime('2024-09-09').date(),
    pd.to_datetime('2024-09-15').date(),
    "09.–15.09."
)

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Funktion für Analyse
def analyze_rain_traffic_airquality(traffic_files, env_file, start_date, end_date, label):
    # Traffic
    df_traffic = pd.concat(
        [pd.json_normalize(pd.read_json(file)["sensordata"]) for file in traffic_files],
        ignore_index=True
    )
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
    df_traffic['date'] = df_traffic['timestamp'].dt.date
    df_traffic_week = df_traffic[
        (df_traffic['date'] >= start_date) & (df_traffic['date'] <= end_date)
    ]
    autos_per_day = df_traffic_week.groupby('date').size().reset_index(name='Fahrzeuganzahl')

    # Environment
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date
    df_env_week = df_env[
        (df_env['date'] >= start_date) & (df_env['date'] <= end_date)
    ]

    # Regen
    rain = df_env_week[df_env_week['sensorLabel'] == 'rainintensity']
    rain_daily = rain.groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')

    # Schadstoffe
    labels = {
        'dust_pm1': 'PM1 (μg/m³)',
        'dust_pm10': 'PM10 (μg/m³)',
        'dust_pm25': 'PM2.5 (μg/m³)',
        'no2': 'NO₂ (ppb)',
        'co': 'CO (ppb)'
    }
    env_summary = []
    for label_key, label_name in labels.items():
        df_sensor = df_env_week[df_env_week['sensorLabel'] == label_key]
        if not df_sensor.empty:
            daily_avg = df_sensor.groupby('date')['value'].mean().reset_index(name=label_name)
            env_summary.append(daily_avg)

    df_merged = rain_daily
    for df in env_summary:
        df_merged = pd.merge(df_merged, df, on='date', how='outer')

    # Zusammenführen
    combined = pd.merge(df_merged, autos_per_day, on='date', how='outer')
    combined['Zeitraum'] = label
    combined = combined.sort_values('date')

    print(f"\n📊 Übersicht Regen, Verkehr, Luftqualität ({label}):")
    print(combined)

    # --------------------------
    # Korrelationen prüfen
    print(f"\n🔗 Korrelationen ({label}):")

    # Regen vs. Luftqualität
    for pollutant in labels.values():
        if pollutant in combined.columns:
            corr = combined['Regenintensität (mm/h)'].corr(combined[pollutant])
            print(f"Regen vs. {pollutant}: {corr:.3f}")

    # Regen vs. Fahrzeuganzahl
    corr = combined['Regenintensität (mm/h)'].corr(combined['Fahrzeuganzahl'])
    print(f"Regen vs. Fahrzeuganzahl: {corr:.3f}")

    # Fahrzeuganzahl vs. Luftqualität
    for pollutant in labels.values():
        if pollutant in combined.columns:
            corr = combined['Fahrzeuganzahl'].corr(combined[pollutant])
            print(f"Fahrzeuganzahl vs. {pollutant}: {corr:.3f}")

# ---------------------------
# Analyse 09.–15.09.2024
analyze_rain_traffic_airquality(
    ["sensordata_traffic_2024-09-09_2024-09-12.json", "sensordata_traffic_2024-09-13_2024-09-15.json"],
    "sensordata_environment_2024-09-09_2024-09-15.json",
    pd.to_datetime('2024-09-09').date(),
    pd.to_datetime('2024-09-15').date(),
    "09.–15.09."
)

# Analyse 20.–26.09.2024
analyze_rain_traffic_airquality(
    ["sensordata_traffic_2024-09-20_2024-09-26.json"],
    "sensordata_environment_2024-09-20_2024-09-26.json",
    pd.to_datetime('2024-09-20').date(),
    pd.to_datetime('2024-09-26').date(),
    "20.–26.09."
)
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# Daten vorbereiten (09.–15.09.2024)

# Environment-Daten laden
df_env = pd.json_normalize(pd.read_json("sensordata_environment_2024-09-09_2024-09-15.json")["sensordata"])
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

# Traffic-Daten laden
df_traffic_1 = pd.json_normalize(pd.read_json("sensordata_traffic_2024-09-09_2024-09-12.json")["sensordata"])
df_traffic_2 = pd.json_normalize(pd.read_json("sensordata_traffic_2024-09-13_2024-09-15.json")["sensordata"])
df_traffic = pd.concat([df_traffic_1, df_traffic_2], ignore_index=True)
df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
df_traffic['date'] = df_traffic['timestamp'].dt.date

# Fahrzeuganzahl pro Tag
autos_per_day = df_traffic.groupby('date').size().reset_index(name='Fahrzeuganzahl')

# Regen pro Tag
rain = df_env[df_env['sensorLabel'] == 'rainintensity']
rain_daily = rain.groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')

# PM1 pro Tag
pm1 = df_env[df_env['sensorLabel'] == 'dust_pm1']
pm1_daily = pm1.groupby('date')['value'].mean().reset_index(name='PM1 (μg/m³)')

# Merge
df_combined = autos_per_day.merge(rain_daily, on='date').merge(pm1_daily, on='date')
df_combined = df_combined.sort_values('date')

print(df_combined)

# -------------------------------
# 3D Scatterplot

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

x = df_combined['Fahrzeuganzahl']
y = df_combined['Regenintensität (mm/h)']
z = df_combined['PM1 (μg/m³)']

sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=80)

# Datenpunkte annotieren
for idx, row in df_combined.iterrows():
    ax.text(row['Fahrzeuganzahl'], row['Regenintensität (mm/h)'], row['PM1 (μg/m³)'],
            str(row['date']), fontsize=8)

# Achsenbeschriftungen und Titel
ax.set_xlabel('Fahrzeuganzahl pro Tag')
ax.set_ylabel('Regenintensität (mm/h)')
ax.set_zlabel('PM1 (μg/m³)')
ax.set_title('Zusammenhang: Feinstaub (PM1), Regen, Fahrzeuganzahl (09.–15.09.2024)')

# Farbleiste hinzufügen
fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5, label='PM1 (μg/m³)')

plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# Daten vorbereiten (falls noch nicht vorhanden)
# Nutze eine der bestehenden Funktionen zur Datenzusammenführung
# Beispielhaft nehmen wir hier die Daten für 20.–26.09.2024:

df_env = pd.json_normalize(pd.read_json("sensordata_environment_2024-09-20_2024-09-26.json")["sensordata"])
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

df_traffic = pd.json_normalize(pd.read_json("sensordata_traffic_2024-09-20_2024-09-26.json")["sensordata"])
df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
df_traffic['date'] = df_traffic['timestamp'].dt.date

# Fahrzeuganzahl pro Tag
autos_per_day = df_traffic.groupby('date').size().reset_index(name='Fahrzeuganzahl')

# Regen pro Tag
rain = df_env[df_env['sensorLabel'] == 'rainintensity']
rain_daily = rain.groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')

# PM1 pro Tag
pm1 = df_env[df_env['sensorLabel'] == 'dust_pm1']
pm1_daily = pm1.groupby('date')['value'].mean().reset_index(name='PM1 (μg/m³)')

# Merge
df_combined = autos_per_day.merge(rain_daily, on='date').merge(pm1_daily, on='date')

print(df_combined)

# -------------------------------
# 3D Scatterplot

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

x = df_combined['Fahrzeuganzahl']
y = df_combined['Regenintensität (mm/h)']
z = df_combined['PM1 (μg/m³)']

sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=80)

for idx, row in df_combined.iterrows():
    ax.text(row['Fahrzeuganzahl'], row['Regenintensität (mm/h)'], row['PM1 (μg/m³)'], 
            str(row['date']), fontsize=8)

ax.set_xlabel('Fahrzeuganzahl pro Tag')
ax.set_ylabel('Regenintensität (mm/h)')
ax.set_zlabel('PM1 (μg/m³)')
ax.set_title('Zusammenhang: Feinstaub (PM1), Regen und Fahrzeuganzahl')

fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5, label='PM1 (μg/m³)')

plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Environment-Daten laden
df_env_raw = pd.read_json("sensordata_environment_2024-09-09_2024-09-15.json")
df_env = pd.json_normalize(df_env_raw["sensordata"])

# Datum vorbereiten
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

# Filter für Analysezeitraum
start_date = pd.to_datetime('2024-09-09').date()
end_date = pd.to_datetime('2024-09-15').date()
df_env_week = df_env[(df_env['date'] >= start_date) & (df_env['date'] <= end_date)]

# Tagesmittel für Windgeschwindigkeit
wind = df_env_week[df_env_week['sensorLabel'] == 'windspeed']
wind_daily = wind.groupby('date')['value'].mean().reset_index(name='Windgeschwindigkeit (m/s)')

# Tagesmittel für Feinstaubwerte
pm_labels = {'dust_pm1': 'PM1 (μg/m³)', 'dust_pm10': 'PM10 (μg/m³)', 'dust_pm25': 'PM2.5 (μg/m³)'}
pm_dfs = []
for label_key, label_name in pm_labels.items():
    pm = df_env_week[df_env_week['sensorLabel'] == label_key]
    pm_daily = pm.groupby('date')['value'].mean().reset_index(name=label_name)
    pm_dfs.append(pm_daily)

# Zusammenführen in eine Tabelle
df_merged = wind_daily
for df_pm in pm_dfs:
    df_merged = pd.merge(df_merged, df_pm, on='date')

# Sortierung
df_merged = df_merged.sort_values('date')

print("\n📊 Übersicht Windgeschwindigkeit und Feinstaub pro Tag:")
print(df_merged)

# ---------------------------
# Korrelationen berechnen
print("\n🔗 Korrelationen Windgeschwindigkeit vs. Feinstaub pro Tag im Gesamtzeitraum:")
for col in ['PM1 (μg/m³)', 'PM10 (μg/m³)', 'PM2.5 (μg/m³)']:
    corr = df_merged['Windgeschwindigkeit (m/s)'].corr(df_merged[col])
    print(f"Korrelation Windgeschwindigkeit vs. {col}: {corr:.3f}")

# ---------------------------
# Visualisierung: Scatterplots
for col in ['PM1 (μg/m³)', 'PM10 (μg/m³)', 'PM2.5 (μg/m³)']:
    plt.figure(figsize=(9,6))
    plt.scatter(df_merged['Windgeschwindigkeit (m/s)'], df_merged[col], s=150, color='teal', alpha=0.7)

    for idx, row in df_merged.iterrows():
        plt.annotate(str(row['date']), 
                     (row['Windgeschwindigkeit (m/s)'], row[col]),
                     textcoords="offset points", xytext=(0,8),
                     ha='center', fontsize=9)

    corr = df_merged['Windgeschwindigkeit (m/s)'].corr(df_merged[col])
    plt.title(f"{col} vs. Windgeschwindigkeit\nKorrelation: {corr:.3f}")
    plt.xlabel("Windgeschwindigkeit (m/s)")
    plt.ylabel(col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Environment-Daten laden
df_env_raw = pd.read_json("sensordata_environment_2024-09-20_2024-09-26.json")
df_env = pd.json_normalize(df_env_raw["sensordata"])

# Datum vorbereiten
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

# Filter für Analysezeitraum
start_date = pd.to_datetime('2024-09-20').date()
end_date = pd.to_datetime('2024-09-26').date()
df_env_week = df_env[(df_env['date'] >= start_date) & (df_env['date'] <= end_date)]

# Tagesmittel für Windgeschwindigkeit
wind = df_env_week[df_env_week['sensorLabel'] == 'windspeed']
wind_daily = wind.groupby('date')['value'].mean().reset_index(name='Windgeschwindigkeit (m/s)')

# Tagesmittel für Feinstaubwerte
pm_labels = {'dust_pm1': 'PM1 (μg/m³)', 'dust_pm10': 'PM10 (μg/m³)', 'dust_pm25': 'PM2.5 (μg/m³)'}
pm_dfs = []
for label_key, label_name in pm_labels.items():
    pm = df_env_week[df_env_week['sensorLabel'] == label_key]
    pm_daily = pm.groupby('date')['value'].mean().reset_index(name=label_name)
    pm_dfs.append(pm_daily)

# Zusammenführen in eine Tabelle
df_merged = wind_daily
for df_pm in pm_dfs:
    df_merged = pd.merge(df_merged, df_pm, on='date')

# Sortierung
df_merged = df_merged.sort_values('date')

print("\n📊 Übersicht Windgeschwindigkeit und Feinstaub pro Tag (20.–26.09.2024):")
print(df_merged)

# ---------------------------
# Korrelationen berechnen
print("\n🔗 Korrelationen Windgeschwindigkeit vs. Feinstaub im Gesamtzeitraum:")
for col in ['PM1 (μg/m³)', 'PM10 (μg/m³)', 'PM2.5 (μg/m³)']:
    corr = df_merged['Windgeschwindigkeit (m/s)'].corr(df_merged[col])
    print(f"Korrelation Windgeschwindigkeit vs. {col}: {corr:.3f}")

# ---------------------------
# Visualisierung: Scatterplots
for col in ['PM1 (μg/m³)', 'PM10 (μg/m³)', 'PM2.5 (μg/m³)']:
    plt.figure(figsize=(9,6))
    plt.scatter(df_merged['Windgeschwindigkeit (m/s)'], df_merged[col], s=150, color='teal', alpha=0.7)

    for idx, row in df_merged.iterrows():
        plt.annotate(str(row['date']), 
                     (row['Windgeschwindigkeit (m/s)'], row[col]),
                     textcoords="offset points", xytext=(0,8),
                     ha='center', fontsize=9)

    corr = df_merged['Windgeschwindigkeit (m/s)'].corr(df_merged[col])
    plt.title(f"{col} vs. Windgeschwindigkeit (20.–26.09.2024)\nKorrelation: {corr:.3f}")
    plt.xlabel("Windgeschwindigkeit (m/s)")
    plt.ylabel(col)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
import pandas as pd

def compute_correlations_traffic_env(traffic_files, env_file, start_date, end_date, label):
    # -----------------------------
    # 🚗 Traffic-Daten laden
    df_traffic = pd.concat(
        [pd.json_normalize(pd.read_json(file)["sensordata"]) for file in traffic_files],
        ignore_index=True
    )
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
    df_traffic['date'] = df_traffic['timestamp'].dt.date

    # Fahrzeuganzahl pro Tag
    autos_per_day = df_traffic[
        (df_traffic['date'] >= start_date) & (df_traffic['date'] <= end_date)
    ].groupby('date').size().reset_index(name='Fahrzeuganzahl')

    # -----------------------------
    # 🌧️💨📈 Environment-Daten laden
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date

    df_env_week = df_env[
        (df_env['date'] >= start_date) & (df_env['date'] <= end_date)
    ]

    # Regen
    rain = df_env_week[df_env_week['sensorLabel'] == 'rainintensity']
    rain_daily = rain.groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')

    # Wind
    wind = df_env_week[df_env_week['sensorLabel'] == 'windspeed']
    wind_daily = wind.groupby('date')['value'].mean().reset_index(name='Windgeschwindigkeit (m/s)')

    # Feinstaub
    pm_labels = {
        'dust_pm1': 'PM1 (μg/m³)',
        'dust_pm10': 'PM10 (μg/m³)',
        'dust_pm25': 'PM2.5 (μg/m³)'
    }
    pm_dfs = []
    for label_key, label_name in pm_labels.items():
        pm = df_env_week[df_env_week['sensorLabel'] == label_key]
        pm_daily = pm.groupby('date')['value'].mean().reset_index(name=label_name)
        pm_dfs.append(pm_daily)

    # -----------------------------
    # Alles zusammenführen
    df_merged = autos_per_day.merge(rain_daily, on='date', how='inner') \
                             .merge(wind_daily, on='date', how='inner')
    for df_pm in pm_dfs:
        df_merged = df_merged.merge(df_pm, on='date', how='inner')

    df_merged = df_merged.sort_values('date')

    print(f"\n📊 Übersicht Daten ({label}):")
    print(df_merged)

    # -----------------------------
    # 📈 Korrelationen berechnen
    print(f"\n🔗 Korrelationen ({label}):")
    cols_to_check = [
        'Fahrzeuganzahl',
        'Regenintensität (mm/h)',
        'Windgeschwindigkeit (m/s)',
        'PM1 (μg/m³)',
        'PM10 (μg/m³)',
        'PM2.5 (μg/m³)'
    ]
    corr_matrix = df_merged[cols_to_check].corr()

    print(corr_matrix.round(3))

    # Optional: Als CSV für Thesis sichern
    corr_matrix.to_csv(f'korrelationen_{label.replace(" ","_")}.csv', sep=';', decimal=',')
    print(f"\n✅ Korrelationen als CSV gespeichert: korrelationen_{label.replace(' ','_')}.csv")

    return corr_matrix

# --------------------------------------------------
# Analyse für 09.–15.09.2024
compute_correlations_traffic_env(
    traffic_files=[
        "sensordata_traffic_2024-09-09_2024-09-12.json",
        "sensordata_traffic_2024-09-13_2024-09-15.json"
    ],
    env_file="sensordata_environment_2024-09-09_2024-09-15.json",
    start_date=pd.to_datetime('2024-09-09').date(),
    end_date=pd.to_datetime('2024-09-15').date(),
    label="09.–15.09.2024"
)

# --------------------------------------------------
# Analyse für 20.–26.09.2024
compute_correlations_traffic_env(
    traffic_files=[
        "sensordata_traffic_2024-09-20_2024-09-26.json"
    ],
    env_file="sensordata_environment_2024-09-20_2024-09-26.json",
    start_date=pd.to_datetime('2024-09-20').date(),
    end_date=pd.to_datetime('2024-09-26').date(),
    label="20.–26.09.2024"
)
