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
# Umweltdaten laden (AQI)
env_dfs = []
for file in env_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_dfs.append(df)

env_all = pd.concat(env_dfs)

# AQI nach Einheit trennen und Tagesmittel berechnen
df_us = env_all[(env_all['sensorLabel'] == 'airqualityindex') & (env_all['unit'] == 'USAEPA_AirNow')]
df_us = df_us.groupby('date')['value'].mean().reset_index(name='AQI_USA')

df_eea = env_all[(env_all['sensorLabel'] == 'airqualityindex') & (env_all['unit'] == 'EEA_EAQI')]
df_eea = df_eea.groupby('date')['value'].mean().reset_index(name='AQI_EEA')

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
# Besucherdaten laden (Besucheranzahl)
visits_dfs = []
for file in visits_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    visits_daily = df.groupby('date').size().reset_index(name='Besucheranzahl')
    visits_dfs.append(visits_daily)

visits_all = pd.concat(visits_dfs).groupby('date').sum().reset_index()

# ----------------------------
# Mergen aller Daten
merged_df = traffic_all.merge(visits_all, on='date', how='inner')
merged_df = merged_df.merge(df_us, on='date', how='left')
merged_df = merged_df.merge(df_eea, on='date', how='left')

print("\n📊 Übersicht der kombinierten Tagesdaten:")
print(merged_df)

# ----------------------------
# 1️⃣ Korrelationen
pairs = [
    ('Besucheranzahl', 'Fahrzeuganzahl'),
    ('Besucheranzahl', 'AQI_USA'),
    ('Besucheranzahl', 'AQI_EEA'),
    ('Fahrzeuganzahl', 'AQI_USA'),
    ('Fahrzeuganzahl', 'AQI_EEA')
]

print("\n✅ Korrelationen:")
for x_var, y_var in pairs:
    corr, p = pearsonr(merged_df[x_var], merged_df[y_var])
    direction = "↑ steigend" if corr > 0 else "↓ fallend"
    significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
    print(f"{x_var} vs. {y_var}: r = {corr:.3f}, p = {p:.5f} ({direction}, {significance})")

# ----------------------------
# 2️⃣ Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    merged_df[['Besucheranzahl', 'Fahrzeuganzahl', 'AQI_USA', 'AQI_EEA']].corr(),
    annot=True, cmap='coolwarm', fmt='.2f'
)
plt.title('Korrelationsmatrix: Besucheranzahl, Fahrzeuganzahl, AQI')
plt.tight_layout()
plt.show()

# ----------------------------
# 3️⃣ Scatterplots mit Trendlinie
for x_var, y_var in pairs:
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

import pandas as pd
import matplotlib.pyplot as plt

def analyze_rain_sum(env_file, start_date, end_date, label):
    # Environment-Daten laden
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date

    # Filter auf Zeitraum
    df_week = df_env[(df_env['date'] >= start_date) & (df_env['date'] <= end_date)]

    # Regenfilter
    rain = df_week[df_week['sensorLabel'] == 'rainintensity']

    # Niederschlagsmenge pro Tag berechnen:
    # Annahme: rainintensity = mm/h, Messung alle 15 Minuten (0.25h)
    rain['rain_amount'] = rain['value'] * 0.25  # wenn andere Intervalle → hier anpassen
    rain_daily_sum = rain.groupby('date')['rain_amount'].sum().reset_index(name='Gesamtniederschlag (mm)')

    print(f"\n📊 Gesamtniederschlag pro Tag ({label}):")
    print(rain_daily_sum)

    print(f"\n🌧️ Gesamtniederschlag in der Woche {label}: {rain_daily_sum['Gesamtniederschlag (mm)'].sum():.2f} mm")

    # Visualisierung
    plt.figure(figsize=(10,6))
    plt.bar(rain_daily_sum['date'].astype(str), rain_daily_sum['Gesamtniederschlag (mm)'], color='skyblue')
    plt.title(f"Niederschlagsmenge pro Tag ({label})")
    plt.xlabel("Datum")
    plt.ylabel("Gesamtniederschlag (mm)")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# Analyse für 09.–15.09.2024
analyze_rain_sum(
    env_file="sensordata_environment_2024-09-09_2024-09-15.json",
    start_date=pd.to_datetime('2024-09-09').date(),
    end_date=pd.to_datetime('2024-09-15').date(),
    label="09.–15.09.2024"
)

# Analyse für 20.–26.09.2024
analyze_rain_sum(
    env_file="sensordata_environment_2024-09-20_2024-09-26.json",
    start_date=pd.to_datetime('2024-09-20').date(),
    end_date=pd.to_datetime('2024-09-26').date(),
    label="20.–26.09.2024"
)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# -----------------------
# Funktion für beide Zeiträume
def analyze_aqi_dependency(traffic_files, visits_file, env_file, start_date, end_date, label):
    # -----------------------
    # Daten laden
    # Traffic
    df_traffic = pd.concat(
        [pd.json_normalize(pd.read_json(file)["sensordata"]) for file in traffic_files],
        ignore_index=True
    )
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
    df_traffic['date'] = df_traffic['timestamp'].dt.date

    # Visits
    df_visits = pd.json_normalize(pd.read_json(visits_file)["sensordata"])
    df_visits['timestamp'] = pd.to_datetime(df_visits['timestamp'])
    df_visits['date'] = df_visits['timestamp'].dt.date

    # Environment
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date

    # Zeitraum filtern
    mask = (df_env['date'] >= start_date) & (df_env['date'] <= end_date)
    df_env = df_env[mask]
    df_traffic = df_traffic[(df_traffic['date'] >= start_date) & (df_traffic['date'] <= end_date)]
    df_visits = df_visits[(df_visits['date'] >= start_date) & (df_visits['date'] <= end_date)]

    # -----------------------
    # Features berechnen
    aqi = df_env[df_env['sensorLabel'] == 'airqualityindex'].groupby('date')['value'].mean().reset_index(name='AirQualityIndex')
    o3 = df_env[df_env['sensorLabel'] == 'o3'].groupby('date')['value'].mean().reset_index(name='O3 (ppb)')
    rain = df_env[df_env['sensorLabel'] == 'rainintensity'].groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')
    autos = df_traffic.groupby('date').size().reset_index(name='Fahrzeuganzahl')
    visitors = df_visits.groupby('date')['visitors'].sum().reset_index(name='Besucheranzahl')

    # -----------------------
    # Merge
    df = aqi.merge(o3, on='date', how='left')
    df = df.merge(rain, on='date', how='left')
    df = df.merge(autos, on='date', how='left')
    df = df.merge(visitors, on='date', how='left')
    df = df.sort_values('date')

    print(f"\n📊 Übersicht der kombinierten Daten ({label}):")
    print(df)

    # -----------------------
    # 1️⃣ Korrelation AQI vs Ozon
    corr_o3 = df['AirQualityIndex'].corr(df['O3 (ppb)'])
    print(f"\n✅ Korrelation AQI vs. Ozon ({label}): {corr_o3:.3f}")

    # -----------------------
    # 2️⃣ Multiple Lineare Regression
    features = ['Fahrzeuganzahl', 'Besucheranzahl', 'Regenintensität (mm/h)']
    X = df[features]
    y = df['AirQualityIndex']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)
    coefs = model.coef_
    r2 = model.score(X_scaled, y)

    print(f"\n✅ Multiple Lineare Regression ({label}):")
    for feat, coef in zip(features, coefs):
        print(f"{feat}: {coef:.3f}")
    print(f"R² der Regression: {r2:.3f}")

    # -----------------------
    # 3️⃣ Farbplot AQI vs Fahrzeuge, farbkodiert nach Regen
    plt.figure(figsize=(9,6))
    scatter = plt.scatter(df['Fahrzeuganzahl'], df['AirQualityIndex'],
                          c=df['Regenintensität (mm/h)'], cmap='Blues', s=120, edgecolor='k')
    for idx, row in df.iterrows():
        plt.annotate(str(row['date']), (row['Fahrzeuganzahl'], row['AirQualityIndex']),
                     textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
    plt.colorbar(scatter, label='Regenintensität (mm/h)')
    plt.xlabel("Fahrzeuganzahl pro Tag")
    plt.ylabel("AirQualityIndex")
    plt.title(f"AQI vs. Fahrzeuganzahl farbkodiert nach Regen ({label})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # -----------------------
    # 4️⃣ Artefaktprüfung am letzten Tag
    last_date = pd.to_datetime(end_date).date()
    rain_last = df[df['date'] == last_date]['Regenintensität (mm/h)'].values[0]
    aqi_last = df[df['date'] == last_date]['AirQualityIndex'].values[0]
    mean_rain = df['Regenintensität (mm/h)'].mean()
    std_rain = df['Regenintensität (mm/h)'].std()

    print(f"\n✅ Am {last_date} betrug die Regenintensität: {rain_last:.3f} mm/h")
    print(f"✅ Am {last_date} betrug der AQI: {aqi_last:.2f}")
    if rain_last > mean_rain + 2 * std_rain:
        print("⚠️ Achtung: Extrem hoher Regen am letzten Tag – mögliche Artefakte prüfen (Sensorverfälschung).")
    else:
        print("✅ Kein außergewöhnlicher Regen am letzten Tag, keine Artefakte erwartet.")

# ---------------------------------------
# Analyse für **09.–15.09.2024**
analyze_aqi_dependency(
    traffic_files=["sensordata_traffic_2024-09-09_2024-09-12.json", "sensordata_traffic_2024-09-13_2024-09-15.json"],
    visits_file="sensordata_visits_2024-09-09_2024-09-15.json",
    env_file="sensordata_environment_2024-09-09_2024-09-15.json",
    start_date=pd.to_datetime('2024-09-09').date(),
    end_date=pd.to_datetime('2024-09-15').date(),
    label="09.–15.09.2024"
)

# ---------------------------------------
# Analyse für **20.–26.09.2024**
analyze_aqi_dependency(
    traffic_files=["sensordata_traffic_2024-09-20_2024-09-26.json"],
    visits_file="sensordata_visits_2024-09-20_2024-09-26.json",
    env_file="sensordata_environment_2024-09-20_2024-09-26.json",
    start_date=pd.to_datetime('2024-09-20').date(),
    end_date=pd.to_datetime('2024-09-26').date(),
    label="20.–26.09.2024"
)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

# ---------------------
# Daten laden

# Environment
df_env = pd.json_normalize(pd.read_json("sensordata_environment_2024-09-20_2024-09-26.json")["sensordata"])
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

# Traffic
df_traffic = pd.json_normalize(pd.read_json("sensordata_traffic_2024-09-20_2024-09-26.json")["sensordata"])
df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
df_traffic['date'] = df_traffic['timestamp'].dt.date

# Visits
df_visits = pd.json_normalize(pd.read_json("sensordata_visits_2024-09-20_2024-09-26.json")["sensordata"])
df_visits['timestamp'] = pd.to_datetime(df_visits['timestamp'])
df_visits['date'] = df_visits['timestamp'].dt.date

# ---------------------
# Features berechnen

# AQI
aqi = df_env[df_env['sensorLabel'] == 'airqualityindex'].groupby('date')['value'].mean().reset_index(name='AirQualityIndex')

# Ozon
o3 = df_env[df_env['sensorLabel'] == 'o3'].groupby('date')['value'].mean().reset_index(name='O3 (ppb)')

# Regen
rain = df_env[df_env['sensorLabel'] == 'rainintensity'].groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')

# Fahrzeuge pro Tag
autos = df_traffic.groupby('date').size().reset_index(name='Fahrzeuganzahl')

# Besucher pro Tag
visitors = df_visits.groupby('date')['visitors'].sum().reset_index(name='Besucheranzahl')

# ---------------------
# Merge alle Daten
df = aqi.merge(o3, on='date', how='left')
df = df.merge(rain, on='date', how='left')
df = df.merge(autos, on='date', how='left')
df = df.merge(visitors, on='date', how='left')

print("\n📊 Übersicht der kombinierten Daten:")
print(df)

# ---------------------
# 1️⃣ Prüfen: Korrelation AQI vs Ozon

corr_o3 = df['AirQualityIndex'].corr(df['O3 (ppb)'])
print(f"\n✅ Korrelation AQI vs. Ozon: {corr_o3:.3f}")

# ---------------------
# 2️⃣ Multiple Lineare Regression

features = ['Fahrzeuganzahl', 'Besucheranzahl', 'Regenintensität (mm/h)']
X = df[features]
y = df['AirQualityIndex']

# Standardisieren
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Regression
model = LinearRegression()
model.fit(X_scaled, y)
coefs = model.coef_

print("\n✅ Multiple Lineare Regression (standardisierte Koeffizienten):")
for feat, coef in zip(features, coefs):
    print(f"{feat}: {coef:.3f}")

print(f"R² der Regression: {model.score(X_scaled, y):.3f}")

# ---------------------
# 3️⃣ Plot: AQI vs. Fahrzeuge farbkodiert nach Regen

plt.figure(figsize=(9,6))
scatter = plt.scatter(df['Fahrzeuganzahl'], df['AirQualityIndex'],
                      c=df['Regenintensität (mm/h)'], cmap='Blues', s=120, edgecolor='k')

for idx, row in df.iterrows():
    plt.annotate(str(row['date']), (row['Fahrzeuganzahl'], row['AirQualityIndex']),
                 textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)

plt.colorbar(scatter, label='Regenintensität (mm/h)')
plt.xlabel("Fahrzeuganzahl pro Tag")
plt.ylabel("AirQualityIndex")
plt.title("AQI vs. Fahrzeuganzahl (Farbkodiert nach Regenintensität)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ---------------------
# 4️⃣ Check 26.09 auf Extremregen und mögliche Messartefakte

rain_26 = df[df['date'] == pd.to_datetime('2024-09-26').date()]['Regenintensität (mm/h)'].values[0]
aqi_26 = df[df['date'] == pd.to_datetime('2024-09-26').date()]['AirQualityIndex'].values[0]

print(f"\n✅ Am 26.09. betrug die Regenintensität: {rain_26:.3f} mm/h")
print(f"✅ Am 26.09. betrug der AQI: {aqi_26:.2f}")

if rain_26 > df['Regenintensität (mm/h)'].mean() + 2 * df['Regenintensität (mm/h)'].std():
    print("⚠️ Achtung: Am 26.09. liegt eine extrem hohe Regenintensität vor – Prüfe Messartefakte oder Effekte auf Sensoren.")
else:
    print("✅ Kein außergewöhnlicher Regen am 26.09., keine Artefakte zu erwarten.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Funktion für einen Zeitraum

def analyze_gases(env_file, traffic_file, visits_file, start_date, end_date, label):

    # -----------------------------
    # Daten laden
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date

    df_traffic = pd.json_normalize(pd.read_json(traffic_file)["sensordata"])
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
    df_traffic['date'] = df_traffic['timestamp'].dt.date

    df_visits = pd.json_normalize(pd.read_json(visits_file)["sensordata"])
    df_visits['timestamp'] = pd.to_datetime(df_visits['timestamp'])
    df_visits['date'] = df_visits['timestamp'].dt.date

    # -----------------------------
    # Tagesmittel berechnen
    gases = {}
    for gas, label_gas in zip(['co', 'no', 'no2'], ['CO (ppb)', 'NO (ppb)', 'NO2 (ppb)']):
        df_gas = df_env[df_env['sensorLabel'] == gas].groupby('date')['value'].mean().reset_index(name=label_gas)
        gases[label_gas] = df_gas

    aqi = df_env[df_env['sensorLabel'] == 'airqualityindex'].groupby('date')['value'].mean().reset_index(name='AirQualityIndex')
    rain = df_env[df_env['sensorLabel'] == 'rainintensity'].groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')
    autos = df_traffic.groupby('date').size().reset_index(name='Fahrzeuganzahl')
    visitors = df_visits.groupby('date')['visitors'].sum().reset_index(name='Besucheranzahl')

    # -----------------------------
    # Merge
    df = aqi.merge(rain, on='date', how='left')
    for df_gas in gases.values():
        df = df.merge(df_gas, on='date', how='left')
    df = df.merge(autos, on='date', how='left')
    df = df.merge(visitors, on='date', how='left')
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].sort_values('date')

    print(f"\n📊 Übersicht der kombinierten Daten ({label}):")
    print(df)

    # -----------------------------
    # Korrelationen
    corr_cols = ['AirQualityIndex', 'CO (ppb)', 'NO (ppb)', 'NO2 (ppb)', 'Regenintensität (mm/h)', 'Fahrzeuganzahl', 'Besucheranzahl']
    corr_matrix = df[corr_cols].corr()
    print(f"\n🔗 Korrelationen ({label}):")
    print(corr_matrix)

    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Korrelationen ({label})")
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Multiple Lineare Regression mit NaN-Handling
    features = ['Fahrzeuganzahl', 'Besucheranzahl', 'Regenintensität (mm/h)', 'CO (ppb)', 'NO (ppb)', 'NO2 (ppb)']
    X = df[features]
    y = df['AirQualityIndex']

    # NaN-Handling
    before_rows = len(X)
    data = pd.concat([X, y], axis=1).dropna()
    after_rows = len(data)
    if before_rows != after_rows:
        print(f"⚠️ Hinweis: {before_rows - after_rows} Zeile(n) mit NaN-Werten für die Regression entfernt.")

    X = data[features]
    y = data['AirQualityIndex']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    print(f"\n✅ Multiple Lineare Regression ({label}):")
    for feat, coef in zip(features, model.coef_):
        print(f"{feat}: {coef:.3f}")
    print(f"R² der Regression: {model.score(X_scaled, y):.3f}")

    # -----------------------------
    # Scatterplots: AQI vs Gase
    for gas in ['CO (ppb)', 'NO (ppb)', 'NO2 (ppb)']:
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(df[gas], df['AirQualityIndex'],
                              c=df['Regenintensität (mm/h)'], cmap='viridis', s=150, edgecolor='k')
        for idx, row in df.iterrows():
            plt.annotate(str(row['date']), (row[gas], row['AirQualityIndex']),
                         textcoords="offset points", xytext=(0,8), ha='center', fontsize=8)
        plt.colorbar(scatter, label='Regenintensität (mm/h)')
        plt.xlabel(gas)
        plt.ylabel("AirQualityIndex")
        plt.title(f"AQI vs. {gas} ({label})")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


# ================================
# Analyse für **09.–15.09.2024**
analyze_gases(
    env_file="sensordata_environment_2024-09-09_2024-09-15.json",
    traffic_file="sensordata_traffic_2024-09-09_2024-09-12.json",  # zweite Datei kann falls gewünscht kombiniert werden
    visits_file="sensordata_visits_2024-09-09_2024-09-15.json",
    start_date=pd.to_datetime('2024-09-09').date(),
    end_date=pd.to_datetime('2024-09-15').date(),
    label="09.–15.09.2024"
)

# ================================
# Analyse für **20.–26.09.2024**
analyze_gases(
    env_file="sensordata_environment_2024-09-20_2024-09-26.json",
    traffic_file="sensordata_traffic_2024-09-20_2024-09-26.json",
    visits_file="sensordata_visits_2024-09-20_2024-09-26.json",
    start_date=pd.to_datetime('2024-09-20').date(),
    end_date=pd.to_datetime('2024-09-26').date(),
    label="20.–26.09.2024"
)
import pandas as pd
import matplotlib.pyplot as plt

def plot_hourly_gas_patterns(env_file, start_date, end_date, label):
    # -----------------------
    # Daten laden
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date
    df_env['hour'] = df_env['timestamp'].dt.hour

    # Filter auf Zeitraum
    df_period = df_env[(df_env['date'] >= start_date) & (df_env['date'] <= end_date)]

    # Sensoren für Analyse
    sensors = {
        'airqualityindex': 'AirQualityIndex',
        'o3': 'O3 (ppb)',
        'no': 'NO (ppb)',
        'no2': 'NO2 (ppb)'
    }

    # Daten pro Stunde aggregieren
    hourly_dfs = {}
    for sensor_label, sensor_name in sensors.items():
        df_sensor = df_period[df_period['sensorLabel'] == sensor_label]
        df_hourly = df_sensor.groupby(['date', 'hour'])['value'].mean().reset_index(name=sensor_name)
        hourly_dfs[sensor_name] = df_hourly

    # -----------------------
    # Plot erstellen
    fig, axs = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    for idx, (sensor_name, df_hourly) in enumerate(hourly_dfs.items()):
        for single_date in sorted(df_hourly['date'].unique()):
            df_day = df_hourly[df_hourly['date'] == single_date]
            axs[idx].plot(df_day['hour'], df_day[sensor_name], marker='o', label=str(single_date))

        axs[idx].set_ylabel(sensor_name)
        axs[idx].grid(True, linestyle='--', alpha=0.5)
        axs[idx].legend(fontsize=8, loc='upper right')

    axs[-1].set_xlabel("Stunde des Tages (0–23)")
    fig.suptitle(f"Stündliche Verläufe: AQI, O3, NO, NO2 ({label})", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

# ========================
# Beispielaufruf 09.–15.09.2024
plot_hourly_gas_patterns(
    env_file="sensordata_environment_2024-09-09_2024-09-15.json",
    start_date=pd.to_datetime('2024-09-09').date(),
    end_date=pd.to_datetime('2024-09-15').date(),
    label="09.–15.09.2024"
)

# ========================
# Beispielaufruf 20.–26.09.2024
plot_hourly_gas_patterns(
    env_file="sensordata_environment_2024-09-20_2024-09-26.json",
    start_date=pd.to_datetime('2024-09-20').date(),
    end_date=pd.to_datetime('2024-09-26').date(),
    label="20.–26.09.2024"
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Funktion zur wöchentlichen Analyse
def analyze_weekly_correlations(env_file, traffic_file, visits_file, start_date, end_date, label):

    # --- Daten laden ---
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date

    df_traffic = pd.json_normalize(pd.read_json(traffic_file)["sensordata"])
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
    df_traffic['date'] = df_traffic['timestamp'].dt.date

    df_visits = pd.json_normalize(pd.read_json(visits_file)["sensordata"])
    df_visits['timestamp'] = pd.to_datetime(df_visits['timestamp'])
    df_visits['date'] = df_visits['timestamp'].dt.date

    # --- Tagesmittel berechnen ---
    gases = {}
    for gas, label_gas in zip(['o3', 'no', 'no2', 'co'], ['O3 (ppb)', 'NO (ppb)', 'NO2 (ppb)', 'CO (ppb)']):
        df_gas = df_env[df_env['sensorLabel'] == gas].groupby('date')['value'].mean().reset_index(name=label_gas)
        gases[label_gas] = df_gas

    # Feinstaub
    for pm, label_pm in zip(['dust_pm1', 'dust_pm25', 'dust_pm10'], ['PM1 (µg/m³)', 'PM2.5 (µg/m³)', 'PM10 (µg/m³)']):
        df_pm = df_env[df_env['sensorLabel'] == pm].groupby('date')['value'].mean().reset_index(name=label_pm)
        gases[label_pm] = df_pm

    # Weitere Messgrößen
    aqi = df_env[df_env['sensorLabel'] == 'airqualityindex'].groupby('date')['value'].mean().reset_index(name='AirQualityIndex')
    rain = df_env[df_env['sensorLabel'] == 'rainintensity'].groupby('date')['value'].mean().reset_index(name='Regenintensität (mm/h)')
    autos = df_traffic.groupby('date').size().reset_index(name='Fahrzeuganzahl')
    visitors = df_visits.groupby('date')['visitors'].sum().reset_index(name='Besucheranzahl')

    # --- Merge ---
    df = aqi.merge(rain, on='date', how='left')
    for df_gas in gases.values():
        df = df.merge(df_gas, on='date', how='left')
    df = df.merge(autos, on='date', how='left')
    df = df.merge(visitors, on='date', how='left')
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].sort_values('date')

    print(f"\n📊 Übersicht der kombinierten Tagesmittel ({label}):")
    print(df)

    # --- Korrelationen über die gesamte Woche ---
    corr_cols = ['AirQualityIndex', 'O3 (ppb)', 'NO (ppb)', 'NO2 (ppb)', 'CO (ppb)',
                 'PM1 (µg/m³)', 'PM2.5 (µg/m³)', 'PM10 (µg/m³)',
                 'Regenintensität (mm/h)', 'Fahrzeuganzahl', 'Besucheranzahl']

    corr_matrix = df[corr_cols].corr()

    print(f"\n🔗 Wöchentliche Korrelationen ({label}):")
    print(corr_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Wöchentliche Korrelationen ({label})")
    plt.tight_layout()
    plt.show()

    # --- Optional: Scatterplots für die gesamte Woche ---
    scatter_pairs = [('Fahrzeuganzahl', 'AirQualityIndex'),
                     ('Besucheranzahl', 'AirQualityIndex'),
                     ('O3 (ppb)', 'AirQualityIndex'),
                     ('NO2 (ppb)', 'AirQualityIndex'),
                     ('CO (ppb)', 'AirQualityIndex')]

    for x_col, y_col in scatter_pairs:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(df[x_col], df[y_col],
                              c=df['Regenintensität (mm/h)'], cmap='viridis', s=120, edgecolor='k')
        for idx, row in df.iterrows():
            plt.annotate(str(row['date']), (row[x_col], row[y_col]),
                         textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)
        plt.colorbar(scatter, label='Regenintensität (mm/h)')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} vs. {x_col} ({label})")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

# =============================
# Analyse für 09.–15.09.2024
analyze_weekly_correlations(
    env_file="sensordata_environment_2024-09-09_2024-09-15.json",
    traffic_file="sensordata_traffic_2024-09-09_2024-09-12.json",  # falls gewünscht: zusammenführen mit _2024-09-13_2024-09-15.json
    visits_file="sensordata_visits_2024-09-09_2024-09-15.json",
    start_date=pd.to_datetime('2024-09-09').date(),
    end_date=pd.to_datetime('2024-09-15').date(),
    label="09.–15.09.2024"
)

# =============================
# Analyse für 20.–26.09.2024
analyze_weekly_correlations(
    env_file="sensordata_environment_2024-09-20_2024-09-26.json",
    traffic_file="sensordata_traffic_2024-09-20_2024-09-26.json",
    visits_file="sensordata_visits_2024-09-20_2024-09-26.json",
    start_date=pd.to_datetime('2024-09-20').date(),
    end_date=pd.to_datetime('2024-09-26').date(),
    label="20.–26.09.2024"
)
