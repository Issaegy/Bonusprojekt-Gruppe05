import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# ----------------------------------------
# Daten laden und vorbereiten

# ENV Daten
env_files = [
    "sensordata_environment_2024-09-09_2024-09-15.json",
    "sensordata_environment_2024-09-20_2024-09-26.json"
]
df_env_list = [pd.json_normalize(pd.read_json(file)["sensordata"]) for file in env_files]
df_env = pd.concat(df_env_list, ignore_index=True)
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

# Traffic Daten
traffic_files = [
    "sensordata_traffic_2024-09-09_2024-09-12.json",
    "sensordata_traffic_2024-09-13_2024-09-15.json",
    "sensordata_traffic_2024-09-20_2024-09-26.json"
]
df_traffic_list = [pd.json_normalize(pd.read_json(file)["sensordata"]) for file in traffic_files]
df_traffic = pd.concat(df_traffic_list, ignore_index=True)
df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
df_traffic['date'] = df_traffic['timestamp'].dt.date

# Visits Daten
visits_files = [
    "sensordata_visits_2024-09-09_2024-09-15.json",
    "sensordata_visits_2024-09-20_2024-09-26.json"
]
df_visits_list = [pd.json_normalize(pd.read_json(file)["sensordata"]) for file in visits_files]
df_visits = pd.concat(df_visits_list, ignore_index=True)
df_visits['timestamp'] = pd.to_datetime(df_visits['timestamp'])
df_visits['date'] = df_visits['timestamp'].dt.date

# ----------------------------------------
# Tagesmittel und Summen berechnen

ozone = df_env[df_env['sensorLabel'] == 'o3'].groupby('date')['value'].mean().reset_index(name='O3 (ppb)')
aqi = df_env[df_env['sensorLabel'] == 'airqualityindex'].groupby('date')['value'].mean().reset_index(name='AQI')
rain = df_env[df_env['sensorLabel'] == 'rainintensity'].groupby('date')['value'].sum().reset_index(name='Rain (mm)')
temperature = df_env[df_env['sensorLabel'] == 'temperature'].groupby('date')['value'].mean().reset_index(name='Temperature (°C)')
traffic = df_traffic.groupby('date').size().reset_index(name='TrafficCount')
visits = df_visits.groupby('date')['visitors'].sum().reset_index(name='Visitors')

# ----------------------------------------
# Merge in einen DataFrame
df = ozone.merge(aqi, on='date', how='outer')
df = df.merge(rain, on='date', how='outer')
df = df.merge(temperature, on='date', how='outer')
df = df.merge(traffic, on='date', how='outer')
df = df.merge(visits, on='date', how='outer')

df = df.sort_values('date')
df.reset_index(drop=True, inplace=True)

# ----------------------------------------
# 1️⃣ Zeitreihenvisualisierung
plt.figure(figsize=(12, 7))
plt.plot(df['date'], df['AQI'], label='AQI', marker='o')
plt.plot(df['date'], df['O3 (ppb)'], label='O3 (ppb)', marker='x')
plt.plot(df['date'], df['TrafficCount'], label='Traffic Count', marker='s')
plt.plot(df['date'], df['Rain (mm)'], label='Rain (mm)', marker='d')
plt.xticks(rotation=45)
plt.title("Zeitreihen: AQI, Ozon, Verkehr, Regen")
plt.xlabel("Datum")
plt.ylabel("Werte")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------------------
# 2️⃣ Random Forest & Ridge/Lasso Regression zur AQI-Vorhersage

features = ['O3 (ppb)', 'Rain (mm)', 'Temperature (°C)', 'TrafficCount']
df_model = df.dropna(subset=['AQI'] + features).copy()

X = df_model[features]
y = df_model['AQI']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_score = rf.score(X_test, y_test)

print("\n🌲 Random Forest:")
print(f"R² Score: {rf_score:.3f}")
print("Feature Importance:")
for feat, imp in zip(features, rf.feature_importances_):
    print(f"{feat}: {imp:.3f}")

# Ridge Regression
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_score = ridge.score(X_test, y_test)

print("\n🔹 Ridge Regression:")
for feat, coef in zip(features, ridge.coef_):
    print(f"{feat}: {coef:.3f}")
print(f"R² Score: {ridge_score:.3f}")

# Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_score = lasso.score(X_test, y_test)

print("\n🔹 Lasso Regression:")
for feat, coef in zip(features, lasso.coef_):
    print(f"{feat}: {coef:.3f}")
print(f"R² Score: {lasso_score:.3f}")

# ----------------------------------------
# 3️⃣ Analyse: Regen vs. AQI
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Rain (mm)', y='AQI', hue='O3 (ppb)', size='TrafficCount', sizes=(40, 200))
plt.title("Scatter: Regen vs. AQI (Farbkodiert: Ozon, Größe: Verkehr)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

rain_aqi_corr = df[['Rain (mm)', 'AQI']].corr().iloc[0, 1]
print(f"\n🔹 Korrelation zwischen Regen und AQI: {rain_aqi_corr:.3f}")

# ----------------------------------------
# 4️⃣ Plausibilitätscheck Regenwerte
print("\n🔍 Plausibilitätscheck Regen (mm):")
print(df['Rain (mm)'].describe())

plt.figure(figsize=(8, 5))
sns.histplot(df['Rain (mm)'].dropna(), bins=15, kde=True)
plt.title("Verteilung der Regenmengen")
plt.xlabel("Rain (mm)")
plt.ylabel("Häufigkeit")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------------------
# 5️⃣ SettingWithCopyWarning vermeiden
# Beispiel: Statt df['col'] = ... nach Filterung, besser:
df.loc[:, 'Rain (mm)'] = df['Rain (mm)'].fillna(0)

print("\n✅ Alle Schritte erfolgreich durchgeführt. Dieser Code ist bereit für deine Präsentation und zur Integration in dein Dashboard/Analyse-Workflow.")
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ----------------------------
# Dateien laden
env_files = [
    "sensordata_environment_2024-09-09_2024-09-15.json",
    "sensordata_environment_2024-09-20_2024-09-26.json"
]

env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# ----------------------------
# Relevante Parameter
params = ['no', 'no2', 'o3', 'co', 'pressure', 'windspeed', 'rainintensity', 'airqualityindex']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]

# ----------------------------
# Air Quality Index nach Einheit trennen
df_us = env_filtered[(env_filtered['sensorLabel'] == 'airqualityindex') & (env_filtered['unit'] == 'USAEPA_AirNow')]
df_us = df_us.groupby('date')['value'].mean().reset_index(name='AQI_USA')

df_eea = env_filtered[(env_filtered['sensorLabel'] == 'airqualityindex') & (env_filtered['unit'] == 'EEA_EAQI')]
df_eea = df_eea.groupby('date')['value'].mean().reset_index(name='AQI_EEA')

# ----------------------------
# Umweltdaten gruppieren
env_main = env_filtered[env_filtered['sensorLabel'] != 'airqualityindex']
env_daily = env_main.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

# ----------------------------
# Merge
merged_df = env_daily.merge(df_us, on='date', how='left')
merged_df = merged_df.merge(df_eea, on='date', how='left')

print("\n📊 Übersicht kombinierter Daten:")
print(merged_df)

# ----------------------------
# 1️⃣ Korrelationen

print("\n✅ Korrelationen mit Air Quality Index (USAEPA & EEA):")

pollutants = ['no', 'no2', 'o3', 'co']
weather = ['rainintensity', 'windspeed', 'pressure']
aq_indices = ['AQI_USA', 'AQI_EEA']

# Korrelationen zwischen AQI und Schadstoffen
for aqi in aq_indices:
    for pollutant in pollutants:
        corr, p = pearsonr(merged_df[aqi], merged_df[pollutant])
        significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
        print(f"{aqi} vs. {pollutant.upper()}: r = {corr:.3f}, p = {p:.5f} ({significance})")

# Korrelationen zwischen AQI und Wetter
for aqi in aq_indices:
    for w in weather:
        corr, p = pearsonr(merged_df[aqi], merged_df[w])
        significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
        print(f"{aqi} vs. {w.capitalize()}: r = {corr:.3f}, p = {p:.5f} ({significance})")

# ----------------------------
# 2️⃣ Heatmaps

for aqi in aq_indices:
    plt.figure(figsize=(8, 6))
    cols = pollutants + weather + [aqi]
    sns.heatmap(
        merged_df[cols].corr(),
        annot=True, cmap='coolwarm', fmt='.2f'
    )
    plt.title(f'Korrelationsmatrix: {aqi}')
    plt.tight_layout()
    plt.show()

# ----------------------------
# 3️⃣ Scatterplots mit Trendlinien

# AQI vs Schadstoffe
for aqi in aq_indices:
    for pollutant in pollutants:
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=merged_df,
            x=pollutant,
            y=aqi,
            scatter_kws={'s': 80, 'edgecolor': 'black'},
            line_kws={'color': 'red'}
        )
        plt.xlabel(f'{pollutant.upper()} (ppb)')
        plt.ylabel(f'{aqi}')
        plt.title(f'{aqi} vs. {pollutant.upper()}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

# AQI vs Wetter
for aqi in aq_indices:
    for w in weather:
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=merged_df,
            x=w,
            y=aqi,
            scatter_kws={'s': 80, 'edgecolor': 'black'},
            line_kws={'color': 'red'}
        )
        plt.xlabel(f'{w.capitalize()}')
        plt.ylabel(f'{aqi}')
        plt.title(f'{aqi} vs. {w.capitalize()}')
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
params = ['windspeed', 'rainintensity', 'pressure', 'airqualityindex']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]

# AQI nach Einheit trennen
df_us = env_filtered[(env_filtered['sensorLabel'] == 'airqualityindex') & (env_filtered['unit'] == 'USAEPA_AirNow')]
df_us = df_us.groupby('date')['value'].mean().reset_index(name='AQI_USA')

df_eea = env_filtered[(env_filtered['sensorLabel'] == 'airqualityindex') & (env_filtered['unit'] == 'EEA_EAQI')]
df_eea = df_eea.groupby('date')['value'].mean().reset_index(name='AQI_EEA')

# Wetterdaten gruppieren
env_main = env_filtered[env_filtered['sensorLabel'] != 'airqualityindex']
env_daily = env_main.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

# Merge AQI und Wetter
merged_df = env_daily.merge(df_us, on='date', how='left')
merged_df = merged_df.merge(df_eea, on='date', how='left')

print("\n📊 Übersicht der kombinierten Daten:")
print(merged_df)

# ----------------------------
# 1️⃣ Korrelationen
aqis = ['AQI_USA', 'AQI_EEA']
weather = ['windspeed', 'rainintensity', 'pressure']

print("\n✅ Korrelationen zwischen AQI und Wetterparametern:")

for aqi in aqis:
    for w in weather:
        corr, p = pearsonr(merged_df[aqi], merged_df[w])
        direction = "↓ verbessert sich" if corr < 0 else "↑ verschlechtert sich"
        significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
        print(f"{aqi} vs. {w.capitalize()}: r = {corr:.3f}, p = {p:.5f} ({direction}, {significance})")

# ----------------------------
# 2️⃣ Scatterplots mit Trendlinien
for aqi in aqis:
    for w in weather:
        plt.figure(figsize=(8, 6))
        sns.regplot(
            data=merged_df,
            x=w,
            y=aqi,
            scatter_kws={'s': 80, 'edgecolor': 'black'},
            line_kws={'color': 'red'}
        )
        plt.xlabel(f'{w.capitalize()}')
        plt.ylabel(f'{aqi}')
        plt.title(f'{aqi} vs. {w.capitalize()}')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()
