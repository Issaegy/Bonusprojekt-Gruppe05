import pandas as pd

# -------------------------------
# Dateien
# -------------------------------
env_files = [
    ("sensordata_environment_2024-09-09_2024-09-15.json", '2024-09-09', '2024-09-15'),
    ("sensordata_environment_2024-09-20_2024-09-26.json", '2024-09-20', '2024-09-26')
]
traffic_files = [
    ("sensordata_traffic_2024-09-09_2024-09-12.json", '2024-09-09', '2024-09-12'),
    ("sensordata_traffic_2024-09-13_2024-09-15.json", '2024-09-13', '2024-09-15'),
    ("sensordata_traffic_2024-09-20_2024-09-26.json", '2024-09-20', '2024-09-26')
]

# -------------------------------
# Environment-Daten laden
# -------------------------------
df_env_list = []
for file, _, _ in env_files:
    df_env_raw = pd.read_json(file)
    df_env = pd.json_normalize(df_env_raw["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date
    df_env_list.append(df_env)

df_env_all = pd.concat(df_env_list, ignore_index=True)

# Sensoren für CO, NO, NO₂, O₃
sensor_labels = {
    'co': 'CO (ppb)',
    'no': 'NO (ppb)',
    'no2': 'NO2 (ppb)',
    'o3': 'O3 (ppb)'
}

df_sensor_means = []
for sensor, name in sensor_labels.items():
    df_sensor = df_env_all[df_env_all['sensorLabel'] == sensor].groupby('date')['value'].mean().reset_index(name=name)
    df_sensor_means.append(df_sensor)

# Merge Sensorwerte
from functools import reduce
df_env_merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), df_sensor_means)

# -------------------------------
# Traffic-Daten laden
# -------------------------------
df_traffic_list = []
for file, _, _ in traffic_files:
    df_traffic_raw = pd.read_json(file)
    df_traffic = pd.json_normalize(df_traffic_raw["sensordata"])
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
    df_traffic['date'] = df_traffic['timestamp'].dt.date
    df_traffic_list.append(df_traffic)

df_traffic_all = pd.concat(df_traffic_list, ignore_index=True)
df_traffic_daily = df_traffic_all.groupby('date').size().reset_index(name='Vehicle Count')

# -------------------------------
# Merge Traffic und Environment
# -------------------------------
df_combined = pd.merge(df_env_merged, df_traffic_daily, on='date', how='inner').sort_values('date')

print("\n📊 Übersicht der Tagesmittel (beide Zeiträume zusammen):")
print(df_combined)

# -------------------------------
# Korrelationen berechnen
# -------------------------------
def compute_correlation(df, start_date, end_date, label):
    df_period = df[(df['date'] >= pd.to_datetime(start_date).date()) &
                   (df['date'] <= pd.to_datetime(end_date).date())]
    print(f"\n🔗 Korrelationen ({label}):")
    for sensor_name in sensor_labels.values():
        if df_period[sensor_name].notna().sum() >= 2:
            corr = df_period[sensor_name].corr(df_period['Vehicle Count'])
            print(f"• Fahrzeuganzahl ↔ {sensor_name}: {corr:.3f}")
        else:
            print(f"• Fahrzeuganzahl ↔ {sensor_name}: Nicht genug Daten")

# Zeitraum 09.–15.09.2024
compute_correlation(df_combined, '2024-09-09', '2024-09-15', "09.–15.09.2024")

# Zeitraum 20.–26.09.2024
compute_correlation(df_combined, '2024-09-20', '2024-09-26', "20.–26.09.2024")

# Beide Zeiträume zusammen
compute_correlation(df_combined, '2024-09-09', '2024-09-26', "Beide Zeiträume zusammen")
import pandas as pd

# -------------------------------
# Dateien
# -------------------------------
env_files = [
    ("sensordata_environment_2024-09-09_2024-09-15.json", '2024-09-09', '2024-09-15'),
    ("sensordata_environment_2024-09-20_2024-09-26.json", '2024-09-20', '2024-09-26')
]
traffic_files = [
    ("sensordata_traffic_2024-09-09_2024-09-12.json", '2024-09-09', '2024-09-12'),
    ("sensordata_traffic_2024-09-13_2024-09-15.json", '2024-09-13', '2024-09-15'),
    ("sensordata_traffic_2024-09-20_2024-09-26.json", '2024-09-20', '2024-09-26')
]

# -------------------------------
# Environment-Daten laden
# -------------------------------
df_env_list = []
for file, _, _ in env_files:
    df_env_raw = pd.read_json(file)
    df_env = pd.json_normalize(df_env_raw["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date
    df_env_list.append(df_env)

df_env_all = pd.concat(df_env_list, ignore_index=True)

# Sensorlabels für Feinstaub
dust_labels = {
    'dust_pm1': 'PM1 (µg/m³)',
    'dust_pm25': 'PM2.5 (µg/m³)',
    'dust_pm10': 'PM10 (µg/m³)'
}

df_dust_means = []
for sensor, name in dust_labels.items():
    df_sensor = df_env_all[df_env_all['sensorLabel'] == sensor].groupby('date')['value'].mean().reset_index(name=name)
    df_dust_means.append(df_sensor)

# Merge Feinstaub-Werte
from functools import reduce
df_env_merged = reduce(lambda left, right: pd.merge(left, right, on='date', how='outer'), df_dust_means)

# -------------------------------
# Traffic-Daten laden
# -------------------------------
df_traffic_list = []
for file, _, _ in traffic_files:
    df_traffic_raw = pd.read_json(file)
    df_traffic = pd.json_normalize(df_traffic_raw["sensordata"])
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
    df_traffic['date'] = df_traffic['timestamp'].dt.date
    df_traffic_list.append(df_traffic)

df_traffic_all = pd.concat(df_traffic_list, ignore_index=True)
df_traffic_daily = df_traffic_all.groupby('date').size().reset_index(name='Vehicle Count')

# -------------------------------
# Merge Traffic und Environment
# -------------------------------
df_combined = pd.merge(df_env_merged, df_traffic_daily, on='date', how='inner').sort_values('date')

print("\n📊 Übersicht der Tagesmittel (beide Zeiträume zusammen):")
print(df_combined)

# -------------------------------
# Korrelationen berechnen
# -------------------------------
def compute_correlation(df, start_date, end_date, label):
    df_period = df[(df['date'] >= pd.to_datetime(start_date).date()) &
                   (df['date'] <= pd.to_datetime(end_date).date())]
    print(f"\n🔗 Korrelationen zwischen Fahrzeuganzahl und Feinstaubwerten ({label}):")
    for sensor_name in dust_labels.values():
        if df_period[sensor_name].notna().sum() >= 2:
            corr = df_period[sensor_name].corr(df_period['Vehicle Count'])
            print(f"• Fahrzeuganzahl ↔ {sensor_name}: {corr:.3f}")
        else:
            print(f"• Fahrzeuganzahl ↔ {sensor_name}: Nicht genug Daten")

# Zeitraum 09.–15.09.2024
compute_correlation(df_combined, '2024-09-09', '2024-09-15', "09.–15.09.2024")

# Zeitraum 20.–26.09.2024
compute_correlation(df_combined, '2024-09-20', '2024-09-26', "20.–26.09.2024")

# Beide Zeiträume zusammen
compute_correlation(df_combined, '2024-09-09', '2024-09-26', "Beide Zeiträume zusammen")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ----------------------------
# Dateien definieren
env_files = [
    "sensordata_environment_2024-09-09_2024-09-15.json",
    "sensordata_environment_2024-09-20_2024-09-26.json"
]

traffic_files = [
    "sensordata_traffic_2024-09-09_2024-09-12.json",
    "sensordata_traffic_2024-09-13_2024-09-15.json",
    "sensordata_traffic_2024-09-20_2024-09-26.json"
]

visits_files = [
    "sensordata_visits_2024-09-09_2024-09-15.json",
    "sensordata_visits_2024-09-20_2024-09-26.json"
]

# ----------------------------
# Umweltdaten laden
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)
env_all = pd.concat(env_dfs)

# AQI hinzufügen
df_us = env_all[(env_all['sensorLabel'] == 'airqualityindex') & (env_all['unit'] == 'USAEPA_AirNow')]
df_us = df_us.groupby('date')['value'].mean().reset_index(name='AQI_USA')

df_eea = env_all[(env_all['sensorLabel'] == 'airqualityindex') & (env_all['unit'] == 'EEA_EAQI')]
df_eea = df_eea.groupby('date')['value'].mean().reset_index(name='AQI_EEA')

# Relevante Parameter filtern
params = ['temperature', 'windspeed', 'rainintensity', 'dust_pm1', 'dust_pm25', 'dust_pm10', 'no', 'no2', 'co']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()
env_daily = env_daily.rename(columns={
    'temperature': 'Temperatur (°C)',
    'windspeed': 'Windgeschwindigkeit (m/s)',
    'rainintensity': 'Regenintensität (mm/h)',
    'dust_pm1': 'PM1 (µg/m³)',
    'dust_pm25': 'PM2.5 (µg/m³)',
    'dust_pm10': 'PM10 (µg/m³)',
    'no': 'NO (ppb)',
    'no2': 'NO2 (ppb)',
    'co': 'CO (ppb)'
})
env_daily = env_daily.merge(df_us, on='date', how='left').merge(df_eea, on='date', how='left')

# ----------------------------
# Verkehrsdaten laden
traffic_dfs = []
for file in traffic_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    daily_count = df.groupby('date').size().reset_index(name='Fahrzeuganzahl')
    traffic_dfs.append(daily_count)
traffic_all = pd.concat(traffic_dfs).groupby('date').sum().reset_index()

# ----------------------------
# Besucherdaten laden
visits_dfs = []
for file in visits_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    daily_visits = df.groupby('date').size().reset_index(name='Besucheranzahl')
    visits_dfs.append(daily_visits)
visits_all = pd.concat(visits_dfs).groupby('date').sum().reset_index()

# ----------------------------
# Merge
merged_df = env_daily.merge(traffic_all, on='date', how='inner').merge(visits_all, on='date', how='inner')

# ----------------------------
# Analysezeiträume definieren
periods = {
    "09.–16.09.2024": ('2024-09-09', '2024-09-16'),
    "20.–26.09.2024": ('2024-09-20', '2024-09-26'),
    "Gesamt 09.–26.09.2024": ('2024-09-09', '2024-09-26')
}

# Variablen für Korrelation
correlation_targets = [
    'PM1 (µg/m³)', 'PM2.5 (µg/m³)', 'PM10 (µg/m³)',
    'NO (ppb)', 'NO2 (ppb)', 'CO (ppb)',
    'Temperatur (°C)', 'Regenintensität (mm/h)', 'Windgeschwindigkeit (m/s)',
    'AQI_USA', 'AQI_EEA'
]

# ----------------------------
# Korrelationen für Besucher und Verkehr getrennt
for period_name, (start_date, end_date) in periods.items():
    period_df = merged_df[
        (merged_df['date'] >= pd.to_datetime(start_date).date()) &
        (merged_df['date'] <= pd.to_datetime(end_date).date())
    ]

    print(f"\n📌 Korrelationen für {period_name} mit Besucheranzahl:")
    for target in correlation_targets:
        corr, p = pearsonr(period_df['Besucheranzahl'], period_df[target])
        significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
        print(f"Besucheranzahl vs. {target}: r = {corr:.3f}, p = {p:.5f} ({significance})")
    
    print(f"\n📌 Korrelationen für {period_name} mit Fahrzeuganzahl:")
    for target in correlation_targets:
        corr, p = pearsonr(period_df['Fahrzeuganzahl'], period_df[target])
        significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
        print(f"Fahrzeuganzahl vs. {target}: r = {corr:.3f}, p = {p:.5f} ({significance})")
    
    # Heatmap
    heatmap_vars = ['Besucheranzahl', 'Fahrzeuganzahl'] + correlation_targets
    plt.figure(figsize=(11, 8))
    sns.heatmap(period_df[heatmap_vars].corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(f'Korrelationsmatrix {period_name}')
    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ----------------------------
# Dateien definieren
traffic_files = [
    "sensordata_traffic_2024-09-09_2024-09-12.json",
    "sensordata_traffic_2024-09-13_2024-09-15.json",
    "sensordata_traffic_2024-09-20_2024-09-26.json"
]

visits_files = [
    "sensordata_visits_2024-09-09_2024-09-15.json",
    "sensordata_visits_2024-09-20_2024-09-26.json"
]

# ----------------------------
# Verkehrsdaten laden
traffic_dfs = []
for file in traffic_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    daily_traffic = df.groupby('date').size().reset_index(name='Fahrzeuganzahl')
    traffic_dfs.append(daily_traffic)
traffic_all = pd.concat(traffic_dfs).groupby('date').sum().reset_index()

# ----------------------------
# Besucherdaten laden
visits_dfs = []
for file in visits_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    daily_visits = df.groupby('date').size().reset_index(name='Besucheranzahl')
    visits_dfs.append(daily_visits)
visits_all = pd.concat(visits_dfs).groupby('date').sum().reset_index()

# ----------------------------
# Merge
merged_df = pd.merge(traffic_all, visits_all, on='date', how='inner')
merged_df = merged_df.sort_values('date')

print("\n📊 Übersicht kombinierter Tagesdaten:")
print(merged_df)

# ----------------------------
# Zeiträume definieren
periods = {
    "09.–16.09.2024": ('2024-09-09', '2024-09-16'),
    "20.–26.09.2024": ('2024-09-20', '2024-09-26'),
    "Gesamtzeitraum 09.–26.09.2024": ('2024-09-09', '2024-09-26')
}

# ----------------------------
# Analyse und Visualisierung
for period_name, (start_date, end_date) in periods.items():
    period_df = merged_df[
        (merged_df['date'] >= pd.to_datetime(start_date).date()) &
        (merged_df['date'] <= pd.to_datetime(end_date).date())
    ]

    # Korrelation berechnen
    corr, p = pearsonr(period_df['Besucheranzahl'], period_df['Fahrzeuganzahl'])
    direction = "↑ steigend" if corr > 0 else "↓ fallend"
    significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"

    print(f"\n📌 {period_name}:")
    print(f"Korrelation Besucheranzahl vs. Fahrzeuganzahl: r = {corr:.3f}, p = {p:.5f} ({direction}, {significance})")

    # Scatterplot mit Trendlinie
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=period_df,
        x='Besucheranzahl',
        y='Fahrzeuganzahl',
        scatter_kws={'s': 80, 'edgecolor': 'black'},
        line_kws={'color': 'red'}
    )
    plt.xlabel('Besucheranzahl')
    plt.ylabel('Fahrzeuganzahl')
    plt.title(f'Besucheranzahl vs. Fahrzeuganzahl ({period_name})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ----------------------------
# JSON-Dateien einlesen
env_files = [
    "sensordata_environment_2024-09-09_2024-09-15.json",
    "sensordata_environment_2024-09-20_2024-09-26.json"
]

dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    dfs.append(df)

env_all = pd.concat(dfs)

# ----------------------------
# Variablen definieren
params = ['no', 'no2', 'co', 'o3',
          'dust_pm1', 'dust_pm25', 'dust_pm10',
          'rainintensity', 'windspeed', 'pressure', 'temperature']

env_filtered = env_all[env_all['sensorLabel'].isin(params)]
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()
env_daily = env_daily.rename(columns={
    'no': 'NO (ppb)',
    'no2': 'NO2 (ppb)',
    'co': 'CO (ppb)',
    'o3': 'O3 (ppb)',
    'dust_pm1': 'PM1 (µg/m³)',
    'dust_pm25': 'PM2.5 (µg/m³)',
    'dust_pm10': 'PM10 (µg/m³)',
    'rainintensity': 'Regenintensität (mm/h)',
    'windspeed': 'Windgeschwindigkeit (m/s)',
    'pressure': 'Luftdruck (hPa)',
    'temperature': 'Temperatur (°C)'
})

# ----------------------------
# Farben zur Unterscheidung
farben = sns.color_palette("tab10", 7)

# ----------------------------
# Scatterplot-Funktion für jede Wettervariable
def scatterplot_wetter_vs_schadstoffe(weather_var):
    plt.figure(figsize=(10, 7))
    
    schadstoffe = ['NO (ppb)', 'NO2 (ppb)', 'CO (ppb)', 'O3 (ppb)',
                   'PM1 (µg/m³)', 'PM2.5 (µg/m³)', 'PM10 (µg/m³)']
    
    for idx, schadstoff in enumerate(schadstoffe):
        sns.regplot(
            data=env_daily,
            x=weather_var,
            y=schadstoff,
            scatter=False,
            color=farben[idx],
            label=f"{schadstoff}"
        )
        corr, p = pearsonr(env_daily[weather_var], env_daily[schadstoff])
        print(f"{schadstoff} vs. {weather_var}: r = {corr:.3f}, p = {p:.5f}")
    
    plt.title(f"Schadstoffe & Feinstaub vs. {weather_var} (Gesamtzeitraum)")
    plt.xlabel(weather_var)
    plt.ylabel("Konzentration")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------
# Für alle Wettervariablen separat erstellen
wettervariablen = ['Temperatur (°C)', 'Windgeschwindigkeit (m/s)', 'Luftdruck (hPa)', 'Regenintensität (mm/h)']

for weather in wettervariablen:
    scatterplot_wetter_vs_schadstoffe(weather)
