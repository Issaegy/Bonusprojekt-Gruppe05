import pandas as pd
import matplotlib.pyplot as plt

def plot_daily_rain(env_file, start_date, end_date, label):
    # ---------------------------
    # 🌧️ Environment-Daten laden
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date

    # Zeitraum filtern
    df_env_week = df_env[
        (df_env['date'] >= start_date) & (df_env['date'] <= end_date)
    ]

    # ---------------------------
    # Niederschlagsmenge berechnen
    rain = df_env_week[df_env_week['sensorLabel'] == 'rainintensity']
    rain_daily = rain.groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')

    # Umrechnung in mm/Tag
    rain_daily['Niederschlag (mm/Tag)'] = rain_daily['Regenintensität (mm/h)'] * 24

    rain_daily = rain_daily.sort_values('date')

    print(f"\n📊 Durchschnittliche Niederschlagsmenge pro Tag ({label}):")
    print(rain_daily[['date', 'Niederschlag (mm/Tag)']])

    # ---------------------------
    # Diagramm erstellen
    plt.figure(figsize=(10, 6))
    plt.bar(rain_daily['date'].astype(str), rain_daily['Niederschlag (mm/Tag)'], width=0.6, alpha=0.7)
    plt.xlabel("Datum")
    plt.ylabel("Niederschlag (mm/Tag)")
    plt.title(f"📈 Tägliche Niederschlagsmenge ({label})")
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Beispielaufruf für 09.–15.09.2024
plot_daily_rain(
    env_file="sensordata_environment_2024-09-09_2024-09-15.json",
    start_date=pd.to_datetime('2024-09-09').date(),
    end_date=pd.to_datetime('2024-09-15').date(),
    label="09.–15.09.2024"
)

# Beispielaufruf für 20.–26.09.2024
plot_daily_rain(
    env_file="sensordata_environment_2024-09-20_2024-09-26.json",
    start_date=pd.to_datetime('2024-09-20').date(),
    end_date=pd.to_datetime('2024-09-26').date(),
    label="20.–26.09.2024"
)
import pandas as pd
import matplotlib.pyplot as plt

def analyze_wind_speed(env_file, start_date, end_date, label):
    # ---------------------------
    # Daten laden
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date

    # Zeitraum filtern
    df_env_week = df_env[
        (df_env['date'] >= start_date) & (df_env['date'] <= end_date)
    ]

    # ---------------------------
    # Windgeschwindigkeit filtern
    wind = df_env_week[df_env_week['sensorLabel'] == 'windspeed']

    # Tagesmittel, Tagesmax und Tagesmin berechnen
    wind_daily = wind.groupby('date')['value'].agg(['mean', 'max', 'min']).reset_index()
    wind_daily.columns = ['date', 'Windmittel (m/s)', 'Max (m/s)', 'Min (m/s)']
    wind_daily = wind_daily.sort_values('date')

    # ---------------------------
    # Ergebnisse anzeigen
    print(f"\n📊 Tagesübersicht Windgeschwindigkeit ({label}):")
    print(wind_daily)

    # ---------------------------
    # Diagramm erstellen
    plt.figure(figsize=(10, 6))
    plt.plot(wind_daily['date'].astype(str), wind_daily['Windmittel (m/s)'], marker='o', label='Mittel')
    plt.plot(wind_daily['date'].astype(str), wind_daily['Max (m/s)'], marker='x', linestyle='--', label='Max')
    plt.plot(wind_daily['date'].astype(str), wind_daily['Min (m/s)'], marker='x', linestyle='--', label='Min')
    plt.xlabel("Datum")
    plt.ylabel("Windgeschwindigkeit (m/s)")
    plt.title(f"💨 Tägliche Windgeschwindigkeit ({label})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------
# Beispielaufruf für 09.–15.09.2024
analyze_wind_speed(
    env_file="sensordata_environment_2024-09-09_2024-09-15.json",
    start_date=pd.to_datetime('2024-09-09').date(),
    end_date=pd.to_datetime('2024-09-15').date(),
    label="09.–15.09.2024"
)

# Beispielaufruf für 20.–26.09.2024
analyze_wind_speed(
    env_file="sensordata_environment_2024-09-20_2024-09-26.json",
    start_date=pd.to_datetime('2024-09-20').date(),
    end_date=pd.to_datetime('2024-09-26').date(),
    label="20.–26.09.2024"
)
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
params = ['pressure', 'windspeed', 'dust_pm10']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]

# Tagesmittel berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

print("\n📊 Tagesmittelwerte für 3D-Plot:")
print(env_daily)

# ----------------------------
# 3D Scatterplot mit Datum-Labels
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x = env_daily['pressure']
y = env_daily['windspeed']
z = env_daily['dust_pm10']
dates = env_daily['date'].astype(str)

scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=60)

# Datum als Label an jedem Punkt
for xi, yi, zi, label in zip(x, y, z, dates):
    ax.text(xi, yi, zi, label, fontsize=8, color='black')

ax.set_xlabel('Luftdruck (hPa)')
ax.set_ylabel('Windgeschwindigkeit (m/s)')
ax.set_zlabel('Feinstaub PM10 (µg/m³)')
ax.set_title('Zusammenhang: Luftdruck, Windgeschwindigkeit, Feinstaub PM10 mit Datum')

fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='Feinstaub PM10 (µg/m³)')
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

# Filter Luftdruck und Windgeschwindigkeit
params = ['pressure', 'windspeed']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]

# Tagesmittel berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

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
# Daten kombinieren
merged_df = pd.merge(env_daily, traffic_all, on='date', how='inner')

print("\n📊 Tagesmittelwerte für 3D-Plot (Luftdruck, Windgeschwindigkeit, Fahrzeuganzahl):")
print(merged_df)

# ----------------------------
# 3D Scatterplot mit Datum-Labels
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x = merged_df['pressure']
y = merged_df['windspeed']
z = merged_df['Fahrzeuganzahl']
dates = merged_df['date'].astype(str)

scatter = ax.scatter(x, y, z, c=z, cmap='plasma', s=60)

# Datum als Label an jedem Punkt
for xi, yi, zi, label in zip(x, y, z, dates):
    ax.text(xi, yi, zi, label, fontsize=8, color='black')

ax.set_xlabel('Luftdruck (hPa)')
ax.set_ylabel('Windgeschwindigkeit (m/s)')
ax.set_zlabel('Fahrzeuganzahl pro Tag')
ax.set_title('Zusammenhang: Luftdruck, Windgeschwindigkeit, Fahrzeuganzahl (mit Datum)')

fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='Fahrzeuganzahl')
plt.tight_layout()
plt.show()
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

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

# Filter relevante Parameter
params = ['pressure', 'windspeed', 'dust_pm10']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]

# Tagesmittel berechnen
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()

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
# Daten kombinieren
merged_df = pd.merge(env_daily, traffic_all, on='date', how='inner').dropna()

print("\n📊 Übersicht verwendete Daten:")
print(merged_df)

# ----------------------------
# 3D-Plot
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x = merged_df['Fahrzeuganzahl']
y = merged_df['windspeed']
z = merged_df['dust_pm10']
c = merged_df['pressure']

p = ax.scatter(x, y, z, c=c, cmap='viridis', s=80)

ax.set_xlabel('Fahrzeuganzahl')
ax.set_ylabel('Windgeschwindigkeit (m/s)')
ax.set_zlabel('Feinstaub PM10 (µg/m³)')
ax.set_title('3D-Zusammenhang: Fahrzeuganzahl, Windgeschwindigkeit, PM10 (Farbe: Luftdruck)')

cb = fig.colorbar(p, ax=ax, shrink=0.5, aspect=10)
cb.set_label('Luftdruck (hPa)')

plt.tight_layout()
plt.show()

# ----------------------------
# Multivariate Regression: PM10 = a * Fahrzeuganzahl + b * Wind + c * Druck + d
X = merged_df[['Fahrzeuganzahl', 'windspeed', 'pressure']]
y = merged_df['dust_pm10']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print("\n🔎 Multivariate lineare Regression:")
print(f"PM10 = {model.coef_[0]:.4f} * Fahrzeuganzahl + {model.coef_[1]:.4f} * Windgeschwindigkeit + {model.coef_[2]:.4f} * Luftdruck + {model.intercept_:.4f}")
print(f"R² (Erklärte Varianz): {r2:.4f}")

# Interpretation
print("\n🪄 Interpretation:")
print(f"- Steigt um {model.coef_[0]:.4f} µg/m³ pro zusätzlichem Fahrzeug pro Tag.")
print(f"- Steigt um {model.coef_[1]:.4f} µg/m³ pro m/s Windgeschwindigkeit.")
print(f"- Steigt um {model.coef_[2]:.4f} µg/m³ pro hPa Luftdruck.")
print(f"- {r2:.2%} der PM10-Varianz wird durch Fahrzeuganzahl, Windgeschwindigkeit und Luftdruck erklärt.")

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np

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

# Filter relevante Parameter
params = ['pressure', 'windspeed', 'dust_pm10']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]

# Tagesmittel berechnen
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
merged_df = pd.merge(env_daily, traffic_all, on='date', how='inner').dropna()

print("\n📊 Verwendete Datenübersicht:")
print(merged_df)

# ----------------------------
# Multivariate Regression
X = merged_df[['Fahrzeuganzahl', 'windspeed', 'pressure']]
y = merged_df['dust_pm10']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print("\n🔎 Multivariate Regression:")
print(f"PM10 = {model.coef_[0]:.4f} * Fahrzeuganzahl + {model.coef_[1]:.4f} * Windgeschwindigkeit + {model.coef_[2]:.4f} * Luftdruck + {model.intercept_:.4f}")
print(f"R²: {r2:.4f}")

# ----------------------------
# 1️⃣ 3D-Plot: PM10 vs. Wind vs. Druck, Farbe: Fahrzeuganzahl
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

x = merged_df['pressure']
y_wind = merged_df['windspeed']
z_pm10 = merged_df['dust_pm10']
c_cars = merged_df['Fahrzeuganzahl']

scatter = ax.scatter(x, y_wind, z_pm10, c=c_cars, cmap='plasma', s=80)

ax.set_xlabel('Luftdruck (hPa)')
ax.set_ylabel('Windgeschwindigkeit (m/s)')
ax.set_zlabel('Feinstaub PM10 (µg/m³)')
ax.set_title('PM10 vs. Windgeschwindigkeit vs. Luftdruck (Farbe: Fahrzeuganzahl)')

cb = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
cb.set_label('Fahrzeuganzahl')

plt.tight_layout()
plt.show()

# ----------------------------
# 2️⃣ Scatterplots
# PM10 vs. Fahrzeuganzahl, Farbe: Wind
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['Fahrzeuganzahl'], merged_df['dust_pm10'], c=merged_df['windspeed'], cmap='viridis', s=80)
plt.colorbar(label='Windgeschwindigkeit (m/s)')
plt.xlabel('Fahrzeuganzahl')
plt.ylabel('Feinstaub PM10 (µg/m³)')
plt.title('PM10 vs. Fahrzeuganzahl (Farbe: Windgeschwindigkeit)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# PM10 vs. Windgeschwindigkeit, Farbe: Fahrzeuganzahl
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['windspeed'], merged_df['dust_pm10'], c=merged_df['Fahrzeuganzahl'], cmap='plasma', s=80)
plt.colorbar(label='Fahrzeuganzahl')
plt.xlabel('Windgeschwindigkeit (m/s)')
plt.ylabel('Feinstaub PM10 (µg/m³)')
plt.title('PM10 vs. Windgeschwindigkeit (Farbe: Fahrzeuganzahl)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# PM10 vs. Druck, Farbe: Wind
plt.figure(figsize=(8, 6))
plt.scatter(merged_df['pressure'], merged_df['dust_pm10'], c=merged_df['windspeed'], cmap='cividis', s=80)
plt.colorbar(label='Windgeschwindigkeit (m/s)')
plt.xlabel('Luftdruck (hPa)')
plt.ylabel('Feinstaub PM10 (µg/m³)')
plt.title('PM10 vs. Luftdruck (Farbe: Windgeschwindigkeit)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------
# 3️⃣ Residuenplot
residuals = y - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='teal', s=80)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Vorhergesagte PM10 (µg/m³)')
plt.ylabel('Residuen (µg/m³)')
plt.title('Residuenplot: Vorhergesagte vs. gemessene PM10')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ----------------------------
# 4️⃣ Heatmap aller Korrelationen
corr_matrix = merged_df[['dust_pm10', 'Fahrzeuganzahl', 'windspeed', 'pressure']].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Korrelationsmatrix: PM10, Fahrzeuganzahl, Windgeschwindigkeit, Luftdruck')
plt.tight_layout()
plt.show()
