import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

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
params = ['co', 'o3', 'no2', 'windspeed', 'temperature', 'pressure', 'rainintensity']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

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
# Merge
merged_df = pd.merge(env_daily, traffic_all, on='date', how='inner')

print("\n📊 Übersicht kombinierter Tagesmittel:")
print(merged_df)

# ----------------------------
# 1️⃣ Korrelationen berechnen

def calc_corr(x, y, label_x, label_y):
    corr, p = pearsonr(x, y)
    print(f"Korrelation {label_x} vs. {label_y}: r = {corr:.3f}, p = {p:.5f}")

print("\n✅ Korrelationen:")
calc_corr(merged_df['co'], merged_df['windspeed'], 'CO', 'Wind')
calc_corr(merged_df['co'], merged_df['temperature'], 'CO', 'Temperatur')
calc_corr(merged_df['co'], merged_df['pressure'], 'CO', 'Druck')
calc_corr(merged_df['co'], merged_df['rainintensity'], 'CO', 'Regen')

calc_corr(merged_df['o3'], merged_df['windspeed'], 'O₃', 'Wind')
calc_corr(merged_df['o3'], merged_df['temperature'], 'O₃', 'Temperatur')
calc_corr(merged_df['o3'], merged_df['pressure'], 'O₃', 'Druck')
calc_corr(merged_df['o3'], merged_df['rainintensity'], 'O₃', 'Regen')

calc_corr(merged_df['no2'], merged_df['Fahrzeuganzahl'], 'NO₂', 'Fahrzeuganzahl')
calc_corr(merged_df['no2'], merged_df['windspeed'], 'NO₂', 'Wind')

# ----------------------------
# 2️⃣ Scatterplots

plot_pairs = [
    ('co', 'windspeed'),
    ('o3', 'windspeed'),
    ('co', 'temperature'),
    ('o3', 'temperature')
]

for x_var, y_var in plot_pairs:
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=merged_df,
        x=x_var,
        y=y_var,
        scatter_kws={'s': 80, 'edgecolor': 'black'},
        line_kws={'color': 'red'}
    )
    plt.xlabel(f'{x_var.upper()}')
    plt.ylabel(f'{y_var.capitalize()}')
    plt.title(f'{x_var.upper()} vs. {y_var.capitalize()}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 3️⃣ Multivariate Regression: CO ~ Wind + Temp + Pressure + Rain

X = merged_df[['windspeed', 'temperature', 'pressure', 'rainintensity']]
y = merged_df['co']

model = LinearRegression()
model.fit(X, y)

coefficients = model.coef_
intercept = model.intercept_
r_squared = model.score(X, y)

print("\n✅ Multivariate Regression:")
print(f"CO = {coefficients[0]:.4f} * Wind + {coefficients[1]:.4f} * Temp + {coefficients[2]:.4f} * Pressure + {coefficients[3]:.4f} * Rain + {intercept:.4f}")
print(f"R² (Erklärte Varianz): {r_squared:.3f}")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

# ----------------------------
# Dateien
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
# Relevante Parameter
params = ['co', 'no2', 'o3', 'dust_pm10', 'windspeed', 'temperature', 'pressure', 'rainintensity']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

# ----------------------------
# Merge
merged_df = pd.merge(env_daily, traffic_all, on='date', how='inner')

print("\n📊 Übersicht kombinierter Daten:")
print(merged_df)

# ----------------------------
# 1️⃣ Korrelationen
print("\n✅ Korrelationen:")
pairs = [
    ('co', 'windspeed'),
    ('co', 'temperature'),
    ('co', 'pressure'),
    ('co', 'rainintensity'),
    ('o3', 'windspeed'),
    ('o3', 'temperature'),
    ('o3', 'pressure'),
    ('o3', 'rainintensity'),
    ('no2', 'Fahrzeuganzahl'),
    ('no2', 'windspeed'),
    ('no2', 'o3'),
    ('co', 'o3'),
    ('dust_pm10', 'co'),
    ('dust_pm10', 'no2'),
    ('dust_pm10', 'windspeed'),
    ('dust_pm10', 'rainintensity'),
    ('dust_pm10', 'o3')
]

for x_var, y_var in pairs:
    corr, p = pearsonr(merged_df[x_var], merged_df[y_var])
    significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
    print(f"{x_var.upper()} vs. {y_var.upper()}: r = {corr:.3f}, p = {p:.5f} ({significance})")

# ----------------------------
# 2️⃣ Heatmap aller Korrelationen
plt.figure(figsize=(10, 8))
sns.heatmap(
    merged_df[['co', 'no2', 'o3', 'dust_pm10', 'windspeed', 'temperature', 'pressure', 'rainintensity', 'Fahrzeuganzahl']].corr(),
    annot=True, cmap='coolwarm', fmt='.2f'
)
plt.title('Korrelationsmatrix aller relevanten Variablen')
plt.tight_layout()
plt.show()

# ----------------------------
# 3️⃣ Scatterplots mit Trendlinien
plot_pairs = [
    ('no2', 'o3'),
    ('co', 'windspeed'),
    ('dust_pm10', 'windspeed'),
    ('dust_pm10', 'no2'),
    ('o3', 'temperature'),
    ('co', 'temperature'),
    ('o3', 'windspeed')
]

for x_var, y_var in plot_pairs:
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=merged_df,
        x=x_var,
        y=y_var,
        scatter_kws={'s': 80, 'edgecolor': 'black'},
        line_kws={'color': 'red'}
    )
    plt.xlabel(f'{x_var.upper()}')
    plt.ylabel(f'{y_var.upper()}')
    plt.title(f'{x_var.upper()} vs. {y_var.upper()}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ----------------------------
# 4️⃣ Multivariate Regression: CO ~ Wind + Temp + Pressure + Rain
X = merged_df[['windspeed', 'temperature', 'pressure', 'rainintensity']]
y = merged_df['co']

model = LinearRegression()
model.fit(X, y)

coefficients = model.coef_
intercept = model.intercept_
r_squared = model.score(X, y)

print("\n✅ Multivariate Regression für CO:")
print(f"CO = {coefficients[0]:.4f} * Wind + {coefficients[1]:.4f} * Temp + {coefficients[2]:.4f} * Pressure + {coefficients[3]:.4f} * Rain + {intercept:.4f}")
print(f"R² (erklärte Varianz): {r_squared:.3f}")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ----------------------------
# Dateien
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
# Filter für NO, NO2, O3, CO
pollutants = ['no', 'no2', 'o3', 'co']
env_filtered = env_all[env_all['sensorLabel'].isin(pollutants)]

# Tagesmittel berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

print("\n📊 Übersicht der Tagesmittel:")
print(env_daily)

# ----------------------------
# Korrelationen berechnen
print("\n✅ Korrelationen zwischen NO, NO2, O3, CO:")
pairs = [
    ('no', 'no2'),
    ('no', 'o3'),
    ('no', 'co'),
    ('no2', 'o3'),
    ('no2', 'co'),
    ('o3', 'co')
]

for x_var, y_var in pairs:
    corr, p = pearsonr(env_daily[x_var], env_daily[y_var])
    significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
    print(f"{x_var.upper()} vs. {y_var.upper()}: r = {corr:.3f}, p = {p:.5f} ({significance})")

# ----------------------------
# Heatmap der Korrelationen
plt.figure(figsize=(8, 6))
sns.heatmap(
    env_daily[['no', 'no2', 'o3', 'co']].corr(),
    annot=True, cmap='coolwarm', fmt='.2f'
)
plt.title('Korrelationsmatrix: NO, NO₂, O₃, CO')
plt.tight_layout()
plt.show()

# ----------------------------
# Scatterplots mit Trendlinien für alle Paare
for x_var, y_var in pairs:
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=env_daily,
        x=x_var,
        y=y_var,
        scatter_kws={'s': 80, 'edgecolor': 'black'},
        line_kws={'color': 'red'}
    )
    plt.xlabel(f'{x_var.upper()} (ppb)')
    plt.ylabel(f'{y_var.upper()} (ppb)')
    plt.title(f'{x_var.upper()} vs. {y_var.upper()}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
