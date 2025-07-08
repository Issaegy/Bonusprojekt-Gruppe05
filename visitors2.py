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

# ----------------------------
# Relevante Parameter extrahieren
params = ['temperature', 'windspeed', 'pressure', 'o3', 'no', 'no2', 'co']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()
env_daily = env_daily.rename(columns={
    'temperature': 'Temperatur (°C)',
    'windspeed': 'Windgeschwindigkeit (m/s)',
    'pressure': 'Luftdruck (hPa)',
    'o3': 'O3 (ppb)',
    'no': 'NO (ppb)',
    'no2': 'NO2 (ppb)',
    'co': 'CO (ppb)'
})

# ----------------------------
# Besucherdaten laden
visits_dfs = []
for file in visits_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    visits_daily = df.groupby('date').size().reset_index(name='Besucheranzahl')
    visits_dfs.append(visits_daily)

visits_all = pd.concat(visits_dfs).groupby('date').sum().reset_index()

# ----------------------------
# Merge Besucher + Umweltdaten
merged_df = pd.merge(env_daily, visits_all, on='date', how='inner')

print("\n📊 Übersicht kombinierter Tagesdaten:")
print(merged_df)

# ----------------------------
# Analysezeiträume definieren
periods = {
    "09.–16.09.2024": ('2024-09-09', '2024-09-16'),
    "20.–26.09.2024": ('2024-09-20', '2024-09-26'),
    "Gesamt (09.–26.09.2024)": ('2024-09-09', '2024-09-26')
}

# Variablen für Korrelation
correlation_targets = [
    'Temperatur (°C)',
    'Windgeschwindigkeit (m/s)',
    'Luftdruck (hPa)',
    'O3 (ppb)',
    'NO (ppb)',
    'NO2 (ppb)',
    'CO (ppb)'
]

# ----------------------------
# Korrelationen pro Zeitraum berechnen
for period_name, (start_date, end_date) in periods.items():
    period_df = merged_df[
        (merged_df['date'] >= pd.to_datetime(start_date).date()) &
        (merged_df['date'] <= pd.to_datetime(end_date).date())
    ]
    
    print(f"\n📌 Korrelationen für {period_name}:")
    for target in correlation_targets:
        corr, p = pearsonr(period_df['Besucheranzahl'], period_df[target])
        direction = "↑ steigend" if corr > 0 else "↓ fallend"
        significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
        print(f"Besucheranzahl vs. {target}: r = {corr:.3f}, p = {p:.5f} ({direction}, {significance})")
    
    # Heatmap
    heatmap_vars = ['Besucheranzahl'] + correlation_targets
    plt.figure(figsize=(9, 7))
    sns.heatmap(period_df[heatmap_vars].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Korrelationsmatrix: Besucheranzahl & Umweltparameter ({period_name})')
    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Dateien definieren
env_files = [
    "sensordata_environment_2024-09-09_2024-09-15.json",
    "sensordata_environment_2024-09-20_2024-09-26.json"
]

# ----------------------------
# Temperaturdaten laden
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# ----------------------------
# Temperatur filtern und Tagesmittel berechnen
temp_df = env_all[env_all['sensorLabel'] == 'temperature']
temp_daily = temp_df.groupby('date')['value'].mean().reset_index(name='Temperatur (°C)')
temp_daily = temp_daily.sort_values('date')

# ----------------------------
# Zeiträume definieren
periods = {
    "09.–16.09.2024": ('2024-09-09', '2024-09-16'),
    "20.–26.09.2024": ('2024-09-20', '2024-09-26')
}

# ----------------------------
# Plot für jede Woche
for period_name, (start_date, end_date) in periods.items():
    period_data = temp_daily[
        (temp_daily['date'] >= pd.to_datetime(start_date).date()) &
        (temp_daily['date'] <= pd.to_datetime(end_date).date())
    ]

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=period_data,
        x='date',
        y='Temperatur (°C)',
        marker='o'
    )
    plt.title(f'Temperaturverlauf {period_name}')
    plt.xlabel('Datum')
    plt.ylabel('Temperatur (°C)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Dateien definieren
env_files = [
    "sensordata_environment_2024-09-09_2024-09-15.json",
    "sensordata_environment_2024-09-20_2024-09-26.json"
]

# ----------------------------
# Winddaten laden
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# ----------------------------
# Wind filtern und Tagesmittel berechnen
wind_df = env_all[env_all['sensorLabel'] == 'windspeed']
wind_daily = wind_df.groupby('date')['value'].mean().reset_index(name='Windgeschwindigkeit (m/s)')
wind_daily = wind_daily.sort_values('date')

# ----------------------------
# Zeiträume definieren
periods = {
    "09.–16.09.2024": ('2024-09-09', '2024-09-16'),
    "20.–26.09.2024": ('2024-09-20', '2024-09-26')
}

# ----------------------------
# Plot für jede Woche
for period_name, (start_date, end_date) in periods.items():
    period_data = wind_daily[
        (wind_daily['date'] >= pd.to_datetime(start_date).date()) &
        (wind_daily['date'] <= pd.to_datetime(end_date).date())
    ]

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=period_data,
        x='date',
        y='Windgeschwindigkeit (m/s)',
        marker='o'
    )
    plt.title(f'Windgeschwindigkeitsverlauf {period_name}')
    plt.xlabel('Datum')
    plt.ylabel('Windgeschwindigkeit (m/s)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
