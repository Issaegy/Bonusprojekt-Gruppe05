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
# Umweltdaten laden (Regenintensität)
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# Regenintensität filtern und Tagesmittel berechnen
rain_df = env_all[env_all['sensorLabel'] == 'rainintensity']
rain_daily = rain_df.groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')

# ----------------------------
# Verkehrsdaten laden (Fahrzeuganzahl)
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
merged_df = pd.merge(rain_daily, traffic_all, on='date', how='inner')

print("\n📊 Übersicht kombinierter Regen- und Fahrzeugdaten:")
print(merged_df)

# ----------------------------
# Korrelation berechnen
corr, p = pearsonr(merged_df['Regenintensität (mm/h)'], merged_df['Fahrzeuganzahl'])
direction = "↓ weniger Fahrzeuge bei Regen" if corr < 0 else "↑ mehr Fahrzeuge bei Regen"
significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"

print(f"\n✅ Korrelation Regenintensität vs. Fahrzeuganzahl:")
print(f"r = {corr:.3f}, p = {p:.5f} ({direction}, {significance})")

# ----------------------------
# Scatterplot mit Trendlinie
plt.figure(figsize=(8, 6))
sns.regplot(
    data=merged_df,
    x='Regenintensität (mm/h)',
    y='Fahrzeuganzahl',
    scatter_kws={'s': 80, 'edgecolor': 'black'},
    line_kws={'color': 'red'}
)
plt.xlabel('Regenintensität (mm/h)')
plt.ylabel('Fahrzeuganzahl')
plt.title('Regenintensität vs. Fahrzeuganzahl')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
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
# Umweltdaten laden (NO)
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# NO filtern und Tagesmittel berechnen
no_df = env_all[env_all['sensorLabel'] == 'no']
no_daily = no_df.groupby('date')['value'].mean().reset_index(name='NO (ppb)')

# ----------------------------
# Verkehrsdaten laden (Fahrzeuganzahl)
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
merged_df = pd.merge(no_daily, traffic_all, on='date', how='inner')

print("\n📊 Übersicht kombinierter NO- und Fahrzeugdaten:")
print(merged_df)

# ----------------------------
# Korrelation berechnen
corr, p = pearsonr(merged_df['Fahrzeuganzahl'], merged_df['NO (ppb)'])
direction = "↑ NO steigt mit Fahrzeuganzahl" if corr > 0 else "↓ NO sinkt mit Fahrzeuganzahl"
significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"

print(f"\n✅ Korrelation Fahrzeuganzahl vs. NO:")
print(f"r = {corr:.3f}, p = {p:.5f} ({direction}, {significance})")

# ----------------------------
# Scatterplot mit Trendlinie
plt.figure(figsize=(8, 6))
sns.regplot(
    data=merged_df,
    x='Fahrzeuganzahl',
    y='NO (ppb)',
    scatter_kws={'s': 80, 'edgecolor': 'black'},
    line_kws={'color': 'red'}
)
plt.xlabel('Fahrzeuganzahl')
plt.ylabel('NO (ppb)')
plt.title('Fahrzeuganzahl vs. Stickstoffmonoxid (NO)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
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

# ----------------------------
# Umweltdaten laden (O3, NO, NO2)
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# Filter O3, NO, NO2 und Tagesmittel berechnen
pollutants = ['o3', 'no', 'no2']
pollutants_df = env_all[env_all['sensorLabel'].isin(pollutants)]
pollutants_daily = pollutants_df.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()
pollutants_daily = pollutants_daily.rename(columns={
    'o3': 'O3 (ppb)',
    'no': 'NO (ppb)',
    'no2': 'NO2 (ppb)'
})

print("\n📊 Übersicht der Tagesmittel von O3, NO, NO2:")
print(pollutants_daily)

# ----------------------------
# Korrelationen berechnen
pairs = [
    ('O3 (ppb)', 'NO (ppb)'),
    ('O3 (ppb)', 'NO2 (ppb)')
]

print("\n✅ Korrelationen zwischen O3 und NO / NO2:")

for x_var, y_var in pairs:
    corr, p = pearsonr(pollutants_daily[x_var], pollutants_daily[y_var])
    direction = "↑ steigend" if corr > 0 else "↓ fallend"
    significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
    print(f"{x_var} vs. {y_var}: r = {corr:.3f}, p = {p:.5f} ({direction}, {significance})")

# ----------------------------
# Scatterplots mit Trendlinie
for x_var, y_var in pairs:
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=pollutants_daily,
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
