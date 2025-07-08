import pandas as pd
import matplotlib.pyplot as plt

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

# Filter Feinstaub und Regenintensität
feinstaub_labels = ['dust_pm1', 'dust_pm25', 'dust_pm10']
regen_label = 'rainintensity'

# Feinstaub Tagesmittel
feinstaub_df = env_all[env_all['sensorLabel'].isin(feinstaub_labels)]
feinstaub_daily = feinstaub_df.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

# Regenintensität Tagesmittel
regen_df = env_all[env_all['sensorLabel'] == regen_label]
regen_daily = regen_df.groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')

# Zusammenführen
merged_df = feinstaub_daily.merge(regen_daily, on='date', how='inner')

# ----------------------------
# Plot für alle Feinstaubtypen
for pm in ['dust_pm1', 'dust_pm25', 'dust_pm10']:
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        merged_df['Regenintensität (mm/h)'],
        merged_df[pm],
        c=merged_df[pm],
        cmap='viridis',
        s=120,
        edgecolor='black'
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label(f'{pm.upper()} (µg/m³)')

    plt.xlabel('Regenintensität (mm/h)')
    plt.ylabel(f'{pm.upper()} (µg/m³)')
    plt.title(f'Zusammenhang zwischen Regenintensität und {pm.upper()}')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Achsenstart bei 0
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

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
# Verkehrsdaten laden und Fahrzeuganzahl pro Tag zählen
traffic_dfs = []
for file in traffic_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    traffic_daily = df.groupby('date').size().reset_index(name='Fahrzeuganzahl')
    traffic_dfs.append(traffic_daily)

traffic_all = pd.concat(traffic_dfs).groupby('date').sum().reset_index()

# ----------------------------
# Besucherdaten laden und Besucher pro Tag zählen
visits_dfs = []
for file in visits_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    visits_daily = df.groupby('date').size().reset_index(name='Besucheranzahl')
    visits_dfs.append(visits_daily)

visits_all = pd.concat(visits_dfs).groupby('date').sum().reset_index()

# ----------------------------
# Zusammenführen
merged_df = pd.merge(traffic_all, visits_all, on='date', how='inner')

print("\n📊 Übersicht der kombinierten Tagesdaten:")
print(merged_df)

# ----------------------------
# Korrelation berechnen
corr_value, p_value = pearsonr(merged_df['Fahrzeuganzahl'], merged_df['Besucheranzahl'])

print(f"\n🔗 Pearson-Korrelation zwischen Fahrzeuganzahl und Besucheranzahl: {corr_value:.3f}")
print(f"📈 p-Wert: {p_value:.5f}")

# Interpretation
if p_value < 0.05:
    print("✅ Die Korrelation ist statistisch signifikant (p < 0.05).")
else:
    print("⚠️ Die Korrelation ist nicht statistisch signifikant (p >= 0.05).")

# ----------------------------
# Scatterplot mit Regressionslinie
plt.figure(figsize=(9, 7))
sns.regplot(
    data=merged_df,
    x='Fahrzeuganzahl',
    y='Besucheranzahl',
    line_kws={'color': 'red'},
    scatter_kws={'s': 80, 'edgecolor': 'black'}
)

plt.xlabel('Fahrzeuganzahl pro Tag')
plt.ylabel('Besucheranzahl pro Tag')
plt.title('Zusammenhang zwischen Fahrzeuganzahl und Besucheranzahl')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

env_files = [
    "sensordata_environment_2024-09-09_2024-09-15.json",
    "sensordata_environment_2024-09-20_2024-09-26.json"
]

# ----------------------------
# Verkehrsdaten laden
traffic_dfs = []
for file in traffic_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    traffic_daily = df.groupby('date').size().reset_index(name='Fahrzeuganzahl')
    traffic_dfs.append(traffic_daily)

traffic_all = pd.concat(traffic_dfs).groupby('date').sum().reset_index()

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
# Umweltdaten laden
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# Filter CO, NO, NO2
pollutants = ['co', 'no', 'no2']
env_filtered = env_all[env_all['sensorLabel'].isin(pollutants)]
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

# ----------------------------
# Daten kombinieren
merged_df = traffic_all.merge(visits_all, on='date', how='inner')
merged_df = merged_df.merge(env_daily, on='date', how='inner')

print("\n📊 Übersicht kombinierter Tagesdaten:")
print(merged_df)

# ----------------------------
# 3D-Plot für jeden Schadstoff mit Tageslabels
for pollutant in pollutants:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = merged_df['Fahrzeuganzahl']
    y = merged_df['Besucheranzahl']
    z = merged_df[pollutant]
    c = merged_df[pollutant]
    dates = merged_df['date'].astype(str)

    scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=80)

    # Tageslabel an jedem Punkt
    for xi, yi, zi, label in zip(x, y, z, dates):
        ax.text(xi, yi, zi, label, fontsize=8, color='black')

    ax.set_xlabel('Fahrzeuganzahl')
    ax.set_ylabel('Besucheranzahl')
    ax.set_zlabel(f'{pollutant.upper()} (ppb)')
    ax.set_title(f'3D-Zusammenhang: Fahrzeuge, Besucher, {pollutant.upper()} mit Datum')

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(f'{pollutant.upper()} (ppb)')

    plt.tight_layout()
    plt.show()
