import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Dateien definieren
traffic_files = [
    "sensordata_traffic_2024-09-09_2024-09-12.json",
    "sensordata_traffic_2024-09-13_2024-09-15.json",
    "sensordata_traffic_2024-09-20_2024-09-26.json"
]

env_files = [
    "sensordata_environment_2024-09-09_2024-09-15.json",
    "sensordata_environment_2024-09-20_2024-09-26.json"
]

# ----------------------------
# Verkehrsdaten laden und Fahrzeuganzahl pro Tag zählen
traffic_dfs = []
for file in traffic_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    # Jede Zeile = ein Fahrzeug => Zeilen pro Tag zählen
    traffic_daily = df.groupby('date').size().reset_index(name='Fahrzeuganzahl')
    traffic_dfs.append(traffic_daily)

traffic_all = pd.concat(traffic_dfs).groupby('date').sum().reset_index()

# ----------------------------
# Umweltdaten laden und aggregieren
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# Nur relevante Sensoren filtern
pollutants = ['dust_pm1', 'dust_pm25', 'dust_pm10', 'o3', 'co', 'no', 'no2']
env_filtered = env_all[env_all['sensorLabel'].isin(pollutants)]

# Tagesmittel pro Schadstoff berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

# ----------------------------
# Merge Verkehr und Umwelt
merged_df = pd.merge(traffic_all, env_daily, on='date', how='inner')

print("\n📊 Zusammengeführte Tagesdaten:")
print(merged_df)

# ----------------------------
# Korrelation berechnen
correlations = merged_df.corr(numeric_only=True)['Fahrzeuganzahl'][pollutants].sort_values(ascending=False)

print("\n🔗 Korrelation zwischen Fahrzeuganzahl und Schadstoffen:")
print(correlations)

# ----------------------------
# Scatterplots mit Regressionslinie
for pollutant in pollutants:
    plt.figure(figsize=(6, 4))
    sns.regplot(
        data=merged_df,
        x='Fahrzeuganzahl',
        y=pollutant,
        ci=None,
        line_kws={'color': 'red'}
    )
    plt.xlabel("Fahrzeuganzahl pro Tag")
    plt.ylabel(f"{pollutant} (Tagesmittel)")
    plt.title(f"Zusammenhang Fahrzeuganzahl und {pollutant}")
    plt.grid(True, linestyle='--', alpha=0.4)
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
# Umweltdaten laden und vorbereiten
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# Relevante Sensoren filtern
params = ['pressure', 'co', 'no', 'o3']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]

# Tagesmittel pro Parameter berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

print("\n📊 Tagesmittelwerte Druck, CO, NO, O3:")
print(env_daily)

# ----------------------------
# Korrelation berechnen
correlations = env_daily.corr(numeric_only=True)['pressure'][['co', 'no', 'o3']].sort_values(ascending=False)

print("\n🔗 Korrelationen zwischen Luftdruck und Schadstoffen:")
print(correlations)

# ----------------------------
# Scatterplots mit Regressionslinie
for pollutant in ['co', 'no', 'o3']:
    plt.figure(figsize=(6, 4))
    sns.regplot(
        data=env_daily,
        x='pressure',
        y=pollutant,
        ci=None,
        line_kws={'color': 'red'}
    )
    plt.xlabel("Luftdruck (hPa)")
    plt.ylabel(f"{pollutant} (ppb)")
    plt.title(f"Zusammenhang Luftdruck und {pollutant}")
    plt.grid(True, linestyle='--', alpha=0.4)
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
# Umweltdaten laden
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# ----------------------------
# Relevante Parameter filtern
params = ['pressure', 'dust_pm1', 'dust_pm25', 'dust_pm10']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]

# Tagesmittel berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

print("\n📊 Tagesmittelwerte Druck und Feinstaub:")
print(env_daily)

# ----------------------------
# Korrelation berechnen
correlations = env_daily.corr(numeric_only=True)['pressure'][['dust_pm1', 'dust_pm25', 'dust_pm10']].sort_values(ascending=False)

print("\n🔗 Korrelationen zwischen Luftdruck und Feinstaub:")
print(correlations)

# ----------------------------
# Scatterplots mit Regressionslinie
for pollutant in ['dust_pm1', 'dust_pm25', 'dust_pm10']:
    plt.figure(figsize=(6, 4))
    sns.regplot(
        data=env_daily,
        x='pressure',
        y=pollutant,
        ci=None,
        line_kws={'color': 'red'}
    )
    plt.xlabel("Luftdruck (hPa)")
    plt.ylabel(f"{pollutant} (μg/m³)")
    plt.title(f"Zusammenhang Luftdruck und {pollutant}")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------
# Dateien definieren
traffic_files = [
    "sensordata_traffic_2024-09-09_2024-09-12.json",
    "sensordata_traffic_2024-09-13_2024-09-15.json",
    "sensordata_traffic_2024-09-20_2024-09-26.json"
]

env_files = [
    "sensordata_environment_2024-09-09_2024-09-15.json",
    "sensordata_environment_2024-09-20_2024-09-26.json"
]

# ----------------------------
# Verkehrsdaten laden und Fahrzeuganzahl zählen
traffic_dfs = []
for file in traffic_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    traffic_daily = df.groupby('date').size().reset_index(name='Fahrzeuganzahl')
    traffic_dfs.append(traffic_daily)

traffic_all = pd.concat(traffic_dfs).groupby('date').sum().reset_index()

# ----------------------------
# Umweltdaten laden
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# Filter relevante Parameter
params = ['dust_pm1', 'dust_pm25', 'dust_pm10', 'pressure', 'windspeed']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]

# Tagesmittel berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

# ----------------------------
# Merge Verkehrsdaten mit Umweltdaten
merged_df = pd.merge(traffic_all, env_daily, on='date', how='inner')

print("\n📊 Übersicht kombinierte Tagesdaten:")
print(merged_df)

# ----------------------------
# Korrelationsmatrix
correlation_matrix = merged_df[['Fahrzeuganzahl', 'dust_pm1', 'dust_pm25', 'dust_pm10', 'pressure', 'windspeed']].corr()

print("\n🔗 Vollständige Korrelationsmatrix:")
print(correlation_matrix)

# ----------------------------
# Heatmap zur Visualisierung
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Korrelationsmatrix: Feinstaub, Fahrzeuganzahl, Luftdruck, Windgeschwindigkeit")
plt.tight_layout()
plt.show()

# ----------------------------
# Pairplot zur Übersicht aller Kombinationen
sns.pairplot(merged_df[['Fahrzeuganzahl', 'dust_pm1', 'dust_pm25', 'dust_pm10', 'pressure', 'windspeed']],
             kind='reg', plot_kws={'line_kws':{'color':'red'}})
plt.suptitle("Pairplot: Zusammenhänge Feinstaub, Fahrzeuganzahl, Luftdruck, Windgeschwindigkeit", y=1.02)
plt.tight_layout()
plt.show()

# ----------------------------
# Einzelne Scatterplots mit Regressionslinie für jede Kombination
import itertools

parameters = ['Fahrzeuganzahl', 'dust_pm1', 'dust_pm25', 'dust_pm10', 'pressure', 'windspeed']
combinations = list(itertools.combinations(parameters, 2))

for x_param, y_param in combinations:
    plt.figure(figsize=(6, 4))
    sns.regplot(
        data=merged_df,
        x=x_param,
        y=y_param,
        ci=None,
        line_kws={'color': 'red'}
    )
    plt.xlabel(x_param)
    plt.ylabel(y_param)
    plt.title(f"Zusammenhang: {x_param} vs. {y_param}")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
