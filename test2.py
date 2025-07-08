import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

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
# Umweltdaten laden
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# Filter CO-Werte
env_co = env_all[env_all['sensorLabel'] == 'co']
env_co_daily = env_co.groupby('date')['value'].mean().reset_index(name='CO (ppb)')

# ----------------------------
# Daten kombinieren
merged_df = pd.merge(traffic_all, env_co_daily, on='date', how='inner')

print("\n📊 Übersicht kombinierter Daten:")
print(merged_df)

# ----------------------------
# Korrelation berechnen
corr_value, p_value = pearsonr(merged_df['Fahrzeuganzahl'], merged_df['CO (ppb)'])
print(f"\n🔗 Pearson-Korrelation zwischen Fahrzeuganzahl und CO: {corr_value:.3f}")
print(f"📈 p-Wert: {p_value:.5f}")

# Interpretation
if p_value < 0.05:
    print("✅ Ergebnis: Die Korrelation ist statistisch signifikant (p < 0.05).")
    if corr_value > 0:
        print("🪄 Interpretation: Mehr Fahrzeuge korrelieren mit höheren CO-Werten.")
    else:
        print("🪄 Interpretation: Mehr Fahrzeuge korrelieren mit niedrigeren CO-Werten (negativer Zusammenhang).")
else:
    print("⚠️ Ergebnis: Keine signifikante Korrelation (p >= 0.05).")

# ----------------------------
# Scatterplot mit Regressionslinie
plt.figure(figsize=(9, 7))
sns.regplot(
    data=merged_df,
    x='Fahrzeuganzahl',
    y='CO (ppb)',
    line_kws={'color': 'red'},
    scatter_kws={'s': 80, 'edgecolor': 'black'}
)

plt.xlabel('Fahrzeuganzahl pro Tag')
plt.ylabel('CO (ppb)')
plt.title('Zusammenhang zwischen Fahrzeuganzahl und CO-Belastung')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
pollutants = ['co']
weather_params = ['temperature', 'pressure', 'rainintensity', 'windspeed']
all_params = pollutants + weather_params

env_filtered = env_all[env_all['sensorLabel'].isin(all_params)]

# Tagesmittel berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

print("\n📊 Übersicht der kombinierten Tagesmittel:")
print(env_daily)

# ----------------------------
# 2D Scatterplots mit Regressionslinie für CO vs. Wetterparameter
import seaborn as sns

for param in weather_params:
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=env_daily,
        x='co',
        y=param,
        scatter_kws={'s': 80, 'edgecolor': 'black'},
        line_kws={'color': 'red'}
    )
    plt.xlabel('CO (ppb)')
    plt.ylabel(f'{param.capitalize()}')
    plt.title(f'Zusammenhang zwischen CO und {param.capitalize()}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 3D-Scatterplot CO, Wetterparameter, Datum als Labels
for param in weather_params:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = env_daily['co']
    y = env_daily[param]
    z = range(len(env_daily))  # Dummy Z-Achse für visuelle Streuung
    c = env_daily[param]
    dates = env_daily['date'].astype(str)

    scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=80)

    # Datum als Label
    for xi, yi, zi, label in zip(x, y, z, dates):
        ax.text(xi, yi, zi, label, fontsize=8, color='black')

    ax.set_xlabel('CO (ppb)')
    ax.set_ylabel(f'{param.capitalize()}')
    ax.set_zlabel('Index zur Tagetrennung')
    ax.set_title(f'3D-Zusammenhang: CO und {param.capitalize()} mit Datum')

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(f'{param.capitalize()}')

    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
pollutants = ['o3']
weather_params = ['temperature', 'pressure', 'rainintensity', 'windspeed']
all_params = pollutants + weather_params

env_filtered = env_all[env_all['sensorLabel'].isin(all_params)]

# Tagesmittel berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

print("\n📊 Übersicht der kombinierten Tagesmittel:")
print(env_daily)

# ----------------------------
# 2D Scatterplots mit Regressionslinie für O3 vs. Wetterparameter
for param in weather_params:
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=env_daily,
        x='o3',
        y=param,
        scatter_kws={'s': 80, 'edgecolor': 'black'},
        line_kws={'color': 'red'}
    )
    plt.xlabel('O₃ (ppb)')
    plt.ylabel(f'{param.capitalize()}')
    plt.title(f'Zusammenhang zwischen O₃ und {param.capitalize()}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 3D-Scatterplot O3, Wetterparameter, Datum als Labels
for param in weather_params:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = env_daily['o3']
    y = env_daily[param]
    z = range(len(env_daily))  # Dummy Z-Achse zur Trennung der Punkte
    c = env_daily[param]
    dates = env_daily['date'].astype(str)

    scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=80)

    # Datum als Label an jedem Punkt
    for xi, yi, zi, label in zip(x, y, z, dates):
        ax.text(xi, yi, zi, label, fontsize=8, color='black')

    ax.set_xlabel('O₃ (ppb)')
    ax.set_ylabel(f'{param.capitalize()}')
    ax.set_zlabel('Index zur Tagetrennung')
    ax.set_title(f'3D-Zusammenhang: O₃ und {param.capitalize()} mit Datum')

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(f'{param.capitalize()}')

    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
pollutants = ['no']
weather_params = ['temperature', 'pressure', 'rainintensity', 'windspeed']
all_params = pollutants + weather_params

env_filtered = env_all[env_all['sensorLabel'].isin(all_params)]

# Tagesmittel berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

print("\n📊 Übersicht der kombinierten Tagesmittel:")
print(env_daily)

# ----------------------------
# 2D Scatterplots mit Regressionslinie für NO vs. Wetterparameter
for param in weather_params:
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=env_daily,
        x='no',
        y=param,
        scatter_kws={'s': 80, 'edgecolor': 'black'},
        line_kws={'color': 'red'}
    )
    plt.xlabel('NO (ppb)')
    plt.ylabel(f'{param.capitalize()}')
    plt.title(f'Zusammenhang zwischen NO und {param.capitalize()}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 3D-Scatterplot NO, Wetterparameter, Datum als Labels
for param in weather_params:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = env_daily['no']
    y = env_daily[param]
    z = range(len(env_daily))  # Dummy Z-Achse zur Trennung
    c = env_daily[param]
    dates = env_daily['date'].astype(str)

    scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=80)

    # Datum an jedem Punkt
    for xi, yi, zi, label in zip(x, y, z, dates):
        ax.text(xi, yi, zi, label, fontsize=8, color='black')

    ax.set_xlabel('NO (ppb)')
    ax.set_ylabel(f'{param.capitalize()}')
    ax.set_zlabel('Index zur Tagetrennung')
    ax.set_title(f'3D-Zusammenhang: NO und {param.capitalize()} mit Datum')

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(f'{param.capitalize()}')

    plt.tight_layout()
    plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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
pollutants = ['no2']
weather_params = ['temperature', 'pressure', 'rainintensity', 'windspeed']
all_params = pollutants + weather_params

env_filtered = env_all[env_all['sensorLabel'].isin(all_params)]

# Tagesmittel berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

print("\n📊 Übersicht der kombinierten Tagesmittel:")
print(env_daily)

# ----------------------------
# 2D Scatterplots mit Regressionslinie für NO2 vs. Wetterparameter
for param in weather_params:
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=env_daily,
        x='no2',
        y=param,
        scatter_kws={'s': 80, 'edgecolor': 'black'},
        line_kws={'color': 'red'}
    )
    plt.xlabel('NO₂ (ppb)')
    plt.ylabel(f'{param.capitalize()}')
    plt.title(f'Zusammenhang zwischen NO₂ und {param.capitalize()}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 3D-Scatterplot NO2, Wetterparameter, Datum als Labels
for param in weather_params:
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    x = env_daily['no2']
    y = env_daily[param]
    z = range(len(env_daily))  # Dummy Z-Achse für saubere Trennung
    c = env_daily[param]
    dates = env_daily['date'].astype(str)

    scatter = ax.scatter(x, y, z, c=c, cmap='viridis', s=80)

    # Datum als Label an jedem Punkt
    for xi, yi, zi, label in zip(x, y, z, dates):
        ax.text(xi, yi, zi, label, fontsize=8, color='black')

    ax.set_xlabel('NO₂ (ppb)')
    ax.set_ylabel(f'{param.capitalize()}')
    ax.set_zlabel('Index zur Tagetrennung')
    ax.set_title(f'3D-Zusammenhang: NO₂ und {param.capitalize()} mit Datum')

    cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(f'{param.capitalize()}')

    plt.tight_layout()
    plt.show()
