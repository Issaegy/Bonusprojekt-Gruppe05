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

# ----------------------------
# Umweltdaten laden
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# Relevante Parameter
params = ['pressure', 'temperature', 'dust_pm1', 'dust_pm25', 'dust_pm10', 'windspeed', 'rainintensity', 'no', 'no2', 'co']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()
env_daily = env_daily.rename(columns={
    'pressure': 'Luftdruck (hPa)',
    'temperature': 'Temperatur (°C)',
    'dust_pm1': 'PM1 (µg/m³)',
    'dust_pm25': 'PM2.5 (µg/m³)',
    'dust_pm10': 'PM10 (µg/m³)',
    'windspeed': 'Windgeschwindigkeit (m/s)',
    'rainintensity': 'Regenintensität (mm/h)',
    'no': 'NO (ppb)',
    'no2': 'NO2 (ppb)',
    'co': 'CO (ppb)'
})

# ----------------------------
# Fahrzeugdaten laden
traffic_dfs = []
for file in traffic_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    traffic_daily = df.groupby('date').size().reset_index(name='Fahrzeuganzahl')
    traffic_dfs.append(traffic_daily)

traffic_all = pd.concat(traffic_dfs).groupby('date').sum().reset_index()

# ----------------------------
# Merge
merged_df = pd.merge(env_daily, traffic_all, on='date', how='inner')

print("\n📊 Übersicht der kombinierten Tagesdaten:")
print(merged_df)

# ----------------------------
# 1️⃣ Korrelationen und Scatterplots

# Definierte Paare:
pairs = [
    ('Luftdruck (hPa)', 'Temperatur (°C)'),
    ('Temperatur (°C)', 'PM1 (µg/m³)'),
    ('Temperatur (°C)', 'PM2.5 (µg/m³)'),
    ('Temperatur (°C)', 'PM10 (µg/m³)'),
    ('Fahrzeuganzahl', 'Windgeschwindigkeit (m/s)'),
    ('Fahrzeuganzahl', 'Regenintensität (mm/h)'),
    ('Fahrzeuganzahl', 'NO (ppb)'),
    ('Fahrzeuganzahl', 'NO2 (ppb)'),
    ('Fahrzeuganzahl', 'CO (ppb)')
]

print("\n✅ Korrelationen:")

for x_var, y_var in pairs:
    corr, p = pearsonr(merged_df[x_var], merged_df[y_var])
    direction = "↑ steigend" if corr > 0 else "↓ fallend"
    significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
    print(f"{x_var} vs. {y_var}: r = {corr:.3f}, p = {p:.5f} ({direction}, {significance})")
    
    # Scatterplot
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=merged_df,
        x=x_var,
        y=y_var,
        scatter_kws={'s': 80, 'edgecolor': 'black'},
        line_kws={'color': 'red'}
    )
    plt.xlabel(f'{x_var}')
    plt.ylabel(f'{y_var}')
    plt.title(f'{x_var} vs. {y_var}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 2️⃣ Heatmap aller relevanten Variablen
heatmap_vars = ['Luftdruck (hPa)', 'Temperatur (°C)', 'PM1 (µg/m³)', 'PM2.5 (µg/m³)', 'PM10 (µg/m³)',
                'Windgeschwindigkeit (m/s)', 'Regenintensität (mm/h)', 'Fahrzeuganzahl',
                'NO (ppb)', 'NO2 (ppb)', 'CO (ppb)']

plt.figure(figsize=(10, 8))
sns.heatmap(merged_df[heatmap_vars].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korrelationsmatrix aller geprüften Variablen')
plt.tight_layout()
plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 📥 1️⃣ Daten laden
# Environment
df_env = pd.json_normalize(pd.read_json("sensordata_environment_2024-09-09_2024-09-15.json")["sensordata"])
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

# Visits
df_visits = pd.json_normalize(pd.read_json("sensordata_visits_2024-09-09_2024-09-15.json")["sensordata"])
df_visits['timestamp'] = pd.to_datetime(df_visits['timestamp'])
df_visits['date'] = df_visits['timestamp'].dt.date



# 📊 2️⃣ Tagesmittel berechnen
temperature = df_env[df_env['sensorLabel'] == 'temperature'].groupby('date')['value'].mean().reset_index(name='Temperature (°C)')
visitors = df_visits.groupby('date')['visitors'].sum().reset_index(name='Visitors')

# 🔄 3️⃣ Merge
df = temperature.merge(visitors, on='date', how='inner')

# Zeitraum filtern
df = df[(df['date'] >= pd.to_datetime('2024-09-09').date()) & (df['date'] <= pd.to_datetime('2024-09-15').date())]

# ✅ 4️⃣ Korrelation berechnen
correlation = df['Temperature (°C)'].corr(df['Visitors'])
print(f"\n🔗 Korrelation zwischen Besucheranzahl und Temperatur (09.–15.09.2024): {correlation:.3f}")

# 📊 5️⃣ Heatmap visualisieren
plt.figure(figsize=(4,3))
sns.heatmap(df[['Temperature (°C)', 'Visitors']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korrelation: Besucher vs. Temperatur (09.–15.09.2024)")
plt.tight_layout()
plt.show()
import pandas as pd

# -------------------------------
# Dateien
# -------------------------------
env_file = "sensordata_environment_2024-09-09_2024-09-15.json"
visits_file = "sensordata_visits_2024-09-09_2024-09-15.json"

# -------------------------------
# Environment (Temperatur)
# -------------------------------
df_env_raw = pd.read_json(env_file)
df_env = pd.json_normalize(df_env_raw["sensordata"])
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

df_temp = df_env[df_env['sensorLabel'] == 'temperature']
df_temp_daily = df_temp.groupby('date')['value'].mean().reset_index(name='Temperature (°C)')

# -------------------------------
# Besucher
# -------------------------------
df_visits_raw = pd.read_json(visits_file)
df_visits = pd.json_normalize(df_visits_raw["sensordata"])
df_visits['timestamp'] = pd.to_datetime(df_visits['timestamp'])
df_visits['date'] = df_visits['timestamp'].dt.date

df_visits_daily = df_visits.groupby('date')['visitors'].sum().reset_index(name='Visitors')

# -------------------------------
# Merge
# -------------------------------
df_combined = pd.merge(df_temp_daily, df_visits_daily, on='date', how='inner').sort_values('date')

print("\n📊 Übersicht der Tagesmittel (09.–15.09.2024):")
print(df_combined)

# -------------------------------
# Korrelation
# -------------------------------
correlation = df_combined['Visitors'].corr(df_combined['Temperature (°C)'])
print(f"\n🔗 Korrelation zwischen Besucheranzahl und Temperatur (09.–15.09.2024): {correlation:.3f}")

import pandas as pd

# -------------------------------
# Dateien
# -------------------------------
env_file = "sensordata_environment_2024-09-20_2024-09-26.json"
visits_file = "sensordata_visits_2024-09-20_2024-09-26.json"

# -------------------------------
# Environment (Temperatur)
# -------------------------------
df_env_raw = pd.read_json(env_file)
df_env = pd.json_normalize(df_env_raw["sensordata"])
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

df_temp = df_env[df_env['sensorLabel'] == 'temperature']
df_temp_daily = df_temp.groupby('date')['value'].mean().reset_index(name='Temperature (°C)')

# -------------------------------
# Besucher
# -------------------------------
df_visits_raw = pd.read_json(visits_file)
df_visits = pd.json_normalize(df_visits_raw["sensordata"])
df_visits['timestamp'] = pd.to_datetime(df_visits['timestamp'])
df_visits['date'] = df_visits['timestamp'].dt.date

df_visits_daily = df_visits.groupby('date')['visitors'].sum().reset_index(name='Visitors')

# -------------------------------
# Merge
# -------------------------------
df_combined = pd.merge(df_temp_daily, df_visits_daily, on='date', how='inner').sort_values('date')

print("\n📊 Übersicht der Tagesmittel (20.–26.09.2024):")
print(df_combined)

# -------------------------------
# Korrelation
# -------------------------------
correlation = df_combined['Visitors'].corr(df_combined['Temperature (°C)'])
print(f"\n🔗 Korrelation zwischen Besucheranzahl und Temperatur (20.–26.09.2024): {correlation:.3f}")

import pandas as pd

# 1️⃣ Daten laden
df_env_week1 = pd.json_normalize(pd.read_json("sensordata_environment_2024-09-09_2024-09-15.json")["sensordata"])
df_env_week2 = pd.json_normalize(pd.read_json("sensordata_environment_2024-09-20_2024-09-26.json")["sensordata"])

df_env_week1['timestamp'] = pd.to_datetime(df_env_week1['timestamp'])
df_env_week1['date'] = df_env_week1['timestamp'].dt.date

df_env_week2['timestamp'] = pd.to_datetime(df_env_week2['timestamp'])
df_env_week2['date'] = df_env_week2['timestamp'].dt.date

# 2️⃣ Temperatur Tagesmittel berechnen
temp_week1 = df_env_week1[df_env_week1['sensorLabel'] == 'temperature'].groupby('date')['value'].mean().reset_index(name='Temperature (°C)')
temp_week2 = df_env_week2[df_env_week2['sensorLabel'] == 'temperature'].groupby('date')['value'].mean().reset_index(name='Temperature (°C)')

# 3️⃣ Ergebnisse anzeigen
print("\n🌡️ Temperatur Tagesmittel 09.–15.09.2024:")
print(temp_week1)

print("\n🌡️ Temperatur Tagesmittel 20.–26.09.2024:")
print(temp_week2)

# 4️⃣ Durchschnittliche Wochentemperatur berechnen
avg_week1 = temp_week1['Temperature (°C)'].mean()
avg_week2 = temp_week2['Temperature (°C)'].mean()

print(f"\n📊 Durchschnittstemperatur in der ersten Woche (09.–15.09.2024): {avg_week1:.2f} °C")
print(f"📊 Durchschnittstemperatur in der zweiten Woche (20.–26.09.2024): {avg_week2:.2f} °C")
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
# Daten laden
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# ----------------------------
# Relevante Parameter
params = ['dust_pm10', 'dust_pm25', 'o3', 'no2', 'co']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()
env_daily = env_daily.rename(columns={
    'dust_pm10': 'PM10 (µg/m³)',
    'dust_pm25': 'PM2.5 (µg/m³)',
    'o3': 'O3 (ppb)',
    'no2': 'NO2 (ppb)',
    'co': 'CO (ppb)'
})

# ----------------------------
# Grenzwerte definieren
limits = {
    'PM10 (µg/m³)': 50,       # EU Tagesgrenzwert
    'PM2.5 (µg/m³)': 25,      # EU Jahresgrenzwert (Tagesbasis konservativ)
    'O3 (ppb)': 60,           # EU 8-Stundenmittel (vereinfachend)
    'NO2 (ppb)': 21,          # EU Jahresmittel
    'CO (ppb)': 8700          # EU 8-Stundenmittel in ppb
}

# ----------------------------
# Klassifizierung
def classify(row, pollutant):
    value = row[pollutant]
    limit = limits[pollutant]
    if value < 0.8 * limit:
        return "✅ OK"
    elif value < limit:
        return "⚠️ Grenzwertig"
    else:
        return "❌ Hoch"

for pollutant in limits.keys():
    env_daily[f'{pollutant} Status'] = env_daily.apply(lambda row: classify(row, pollutant), axis=1)

# ----------------------------
# Ergebnis anzeigen
pd.set_option('display.max_columns', None)
print("\n📊 Übersicht der Umweltdaten mit Status:")
print(env_daily)

# ----------------------------
# Plot für Visualisierung
melted = env_daily.melt(id_vars='date', value_vars=list(limits.keys()),
                        var_name='Parameter', value_name='Wert')

plt.figure(figsize=(12, 6))
sns.lineplot(data=melted, x='date', y='Wert', hue='Parameter', marker='o')
plt.axhline(y=50, color='gray', linestyle='--', label='PM10 Grenzwert 50 µg/m³')
plt.axhline(y=25, color='gray', linestyle=':', label='PM2.5 Grenzwert 25 µg/m³')
plt.axhline(y=60, color='gray', linestyle='-.', label='O3 Grenzwert 60 ppb')
plt.axhline(y=21, color='gray', linestyle='-', label='NO2 Grenzwert 21 ppb')
plt.xlabel('Datum')
plt.ylabel('Messwert')
plt.title('Verlauf der Umweltdaten im Vergleich zu Grenzwerten')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

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
# Environment (NO)
# -------------------------------
df_env_list = []
for file, start, end in env_files:
    df_env_raw = pd.read_json(file)
    df_env = pd.json_normalize(df_env_raw["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date
    df_env_list.append(df_env)

df_env_all = pd.concat(df_env_list, ignore_index=True)
df_no = df_env_all[df_env_all['sensorLabel'] == 'no']
df_no_daily = df_no.groupby('date')['value'].mean().reset_index(name='NO (ppb)')

# -------------------------------
# Traffic (Fahrzeuganzahl)
# -------------------------------
df_traffic_list = []
for file, start, end in traffic_files:
    df_traffic_raw = pd.read_json(file)
    df_traffic = pd.json_normalize(df_traffic_raw["sensordata"])
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
    df_traffic['date'] = df_traffic['timestamp'].dt.date
    df_traffic_list.append(df_traffic)

df_traffic_all = pd.concat(df_traffic_list, ignore_index=True)
df_traffic_daily = df_traffic_all.groupby('date').size().reset_index(name='Vehicle Count')

# -------------------------------
# Merge
# -------------------------------
df_combined = pd.merge(df_no_daily, df_traffic_daily, on='date', how='inner').sort_values('date')

print("\n📊 Übersicht der Tagesmittel (beide Zeiträume zusammen):")
print(df_combined)

# -------------------------------
# Korrelationen
# -------------------------------

# Zeitraum 09.–15.09.2024
df_period1 = df_combined[(df_combined['date'] >= pd.to_datetime('2024-09-09').date()) &
                         (df_combined['date'] <= pd.to_datetime('2024-09-15').date())]
corr_period1 = df_period1['NO (ppb)'].corr(df_period1['Vehicle Count'])
print(f"\n🔗 Korrelation zwischen Fahrzeuganzahl und NO (09.–15.09.2024): {corr_period1:.3f}")

# Zeitraum 20.–26.09.2024
df_period2 = df_combined[(df_combined['date'] >= pd.to_datetime('2024-09-20').date()) &
                         (df_combined['date'] <= pd.to_datetime('2024-09-26').date())]
corr_period2 = df_period2['NO (ppb)'].corr(df_period2['Vehicle Count'])
print(f"🔗 Korrelation zwischen Fahrzeuganzahl und NO (20.–26.09.2024): {corr_period2:.3f}")

# Beide Zeiträume zusammen
corr_all = df_combined['NO (ppb)'].corr(df_combined['Vehicle Count'])
print(f"🔗 Korrelation zwischen Fahrzeuganzahl und NO (beide Zeiträume zusammen): {corr_all:.3f}")
