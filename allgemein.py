import pandas as pd

# Environment-Datei laden und normalisieren
df_env_raw = pd.read_json("sensordata_environment_2024-09-09_2024-09-15.json")
df_env = pd.json_normalize(df_env_raw["sensordata"])

# Verfügbare SensorLabels (Messgrößen) prüfen
print("\n📋 Verfügbare Messgrößen in 'sensorLabel':")
print(df_env['sensorLabel'].unique())

# Optional: Zeige die verfügbaren Einheiten der Messungen
print("\n📏 Verfügbare Einheiten in 'unit':")
print(df_env['unit'].unique())

# Optional: Zeige alle Kombinationen aus sensorLabel und unit, um Übersicht zu erhalten
print("\n🔍 Übersicht aller Kombinationen von Messgröße und Einheit:")
print(df_env[['sensorLabel', 'unit']].drop_duplicates().reset_index(drop=True))
import pandas as pd

# JSON-Datei laden
df_env_raw = pd.read_json("sensordata_environment_2024-09-09_2024-09-15.json")
df_env_raw1 = pd.read_json("sensordata_environment_2024-09-20_2024-09-26.json")

# sensordata extrahieren und normalisieren
df_env = pd.json_normalize(df_env_raw["sensordata"])
df_env1 = pd.json_normalize(df_env_raw1["sensordata"])

# Spaltennamen als Liste extrahieren
spaltennamen = df_env.columns.tolist()
spaltennamen1 = df_env.columns.tolist()
# Ausgabe
print("\n📋 Spalten in 'sensordata_environment_2024-09-09_2024-09-15.json':")
print(spaltennamen)
print("\n📋 Spalten in 'sensordata_environment_2024-09-20_2024-09-26.json':")
print(spaltennamen1)
import pandas as pd

# Environment-Datei laden und normalisieren
df_env_raw = pd.read_json("sensordata_environment_2024-09-09_2024-09-15.json")
df_env = pd.json_normalize(df_env_raw["sensordata"])

# Alle eindeutigen Sensor-Namen prüfen
print("\n📋 Eindeutige Sensoren in 'deviceName':")
print(df_env['deviceName'].unique())

# Optional: Alle eindeutigen Sensor-Typen prüfen
print("\n🛠️ Eindeutige Sensor-Typen in 'deviceType':")
print(df_env['deviceType'].unique())

# Optional: Alle eindeutigen gemessenen Größen prüfen
print("\n📊 Eindeutige Messgrößen in 'sensorLabel':")
print(df_env['sensorLabel'].unique())

# Optional: Übersicht als Tabelle ausgeben (kombiniert)
print("\n🔍 Übersicht aller Kombinationen aus deviceName, deviceType und sensorLabel:")
overview = df_env[['deviceName', 'deviceType', 'sensorLabel']].drop_duplicates().reset_index(drop=True)
print(overview)
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Funktion zur NO₂ Analyse
def analyze_no2(env_file, start_date, end_date, label):
    # Environment-Daten laden
    df_env = pd.json_normalize(pd.read_json(env_file)["sensordata"])
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date

    # Filter Zeitraum
    df_env = df_env[(df_env['date'] >= start_date) & (df_env['date'] <= end_date)]

    # NO₂ herausfiltern und Tagesmittel berechnen
    df_no2 = df_env[df_env['sensorLabel'] == 'no2']
    no2_daily = df_no2.groupby('date')['value'].mean().reset_index(name='NO2 (ppb)')

    # Ergebnisse anzeigen
    print(f"\n📊 NO₂ (ppb) Tagesmittel ({label}):")
    print(no2_daily)

    # Visualisierung
    plt.figure(figsize=(9,5))
    plt.bar(no2_daily['date'].astype(str), no2_daily['NO2 (ppb)'],
            color='slateblue', alpha=0.8, label="NO₂ (ppb) Tagesmittel")
    plt.axhline(y=21, color='red', linestyle='--', label='WHO-Jahresrichtwert ~21 ppb')
    plt.title(f"NO₂ (ppb) Tagesmittel ({label})")
    plt.xlabel("Datum")
    plt.ylabel("NO₂ (ppb)")
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 📦 Traffic-Daten zusammenführen für 09.–15.09.2024
df_traffic_1 = pd.json_normalize(pd.read_json("sensordata_traffic_2024-09-09_2024-09-12.json")["sensordata"])
df_traffic_2 = pd.json_normalize(pd.read_json("sensordata_traffic_2024-09-13_2024-09-15.json")["sensordata"])
df_traffic = pd.concat([df_traffic_1, df_traffic_2], ignore_index=True)

df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
df_traffic['date'] = df_traffic['timestamp'].dt.date

# Fahrzeuganzahl pro Tag berechnen und anzeigen
autos_per_day = df_traffic.groupby('date').size().reset_index(name='Fahrzeuganzahl')
print("\n🚗 Fahrzeuganzahl pro Tag (09.–15.09.2024):")
print(autos_per_day)

# -----------------------------
# Analyse Woche 09.–15.09.2024
analyze_no2(
    env_file="sensordata_environment_2024-09-09_2024-09-15.json",
    start_date=pd.to_datetime('2024-09-09').date(),
    end_date=pd.to_datetime('2024-09-15').date(),
    label="09.–15.09.2024"
)

# -----------------------------
# Analyse Woche 20.–26.09.2024
analyze_no2(
    env_file="sensordata_environment_2024-09-20_2024-09-26.json",
    start_date=pd.to_datetime('2024-09-20').date(),
    end_date=pd.to_datetime('2024-09-26').date(),
    label="20.–26.09.2024"
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def ozone_model_analysis(env_files, traffic_files, visits_files, start_date, end_date, label):
    # 📥 1️⃣ Daten laden
    df_env_list = [pd.json_normalize(pd.read_json(file)["sensordata"]) for file in env_files]
    df_env = pd.concat(df_env_list, ignore_index=True)
    df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
    df_env['date'] = df_env['timestamp'].dt.date

    df_traffic_list = [pd.json_normalize(pd.read_json(file)["sensordata"]) for file in traffic_files]
    df_traffic = pd.concat(df_traffic_list, ignore_index=True)
    df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
    df_traffic['date'] = df_traffic['timestamp'].dt.date

    df_visits_list = [pd.json_normalize(pd.read_json(file)["sensordata"]) for file in visits_files]
    df_visits = pd.concat(df_visits_list, ignore_index=True)
    df_visits['timestamp'] = pd.to_datetime(df_visits['timestamp'])
    df_visits['date'] = df_visits['timestamp'].dt.date

    # 📊 2️⃣ Tagesmittel berechnen
    ozone = df_env[df_env['sensorLabel'] == 'o3'].groupby('date')['value'].mean().reset_index(name='O3 (ppb)')
    temperature = df_env[df_env['sensorLabel'] == 'temperature'].groupby('date')['value'].mean().reset_index(name='Temperature (°C)')
    sunshine = df_env[df_env['sensorLabel'] == 'sunshine'].groupby('date')['value'].mean().reset_index(name='Sunshine (h)')
    wind = df_env[df_env['sensorLabel'] == 'windspeed'].groupby('date')['value'].mean().reset_index(name='Wind (m/s)')
    traffic = df_traffic.groupby('date').size().reset_index(name='TrafficCount')
    visits = df_visits.groupby('date')['visitors'].sum().reset_index(name='Visitors')

    # 🔄 3️⃣ Merge
    df = ozone.merge(temperature, on='date', how='outer')
    df = df.merge(sunshine, on='date', how='outer')
    df = df.merge(wind, on='date', how='outer')
    df = df.merge(traffic, on='date', how='outer')
    df = df.merge(visits, on='date', how='outer')

    # Zeitraum filtern
    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].sort_values('date')

    # 🩹 Fehlende Werte behandeln
    df = df.dropna(subset=['O3 (ppb)'])  # Zielvariable muss vorhanden sein

    # Nur numerische Spalten zum Auffüllen auswählen
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    # ✅ Übersicht
    print(f"\n📊 Übersicht der kombinierten Tagesmittel ({label}):")
    print(df)

    # 🔗 4️⃣ Heatmap der Korrelationen
    corr_matrix = df[['O3 (ppb)', 'Temperature (°C)', 'Sunshine (h)', 'Wind (m/s)', 'TrafficCount', 'Visitors']].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title(f"Korrelation Ozon vs. Parameter ({label})")
    plt.tight_layout()
    plt.show()

    # 🤖 5️⃣ Ozon-Prognosemodell
    X = df[['Temperature (°C)', 'Sunshine (h)', 'Wind (m/s)', 'TrafficCount']]
    y = df['O3 (ppb)']

    # Ersetze NaNs in Features durch Spaltenmittel (nochmals robust)
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    if len(X_scaled) >= 2:
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)

        print(f"\n✅ Ozonmodell ({label}):")
        for feature, coef in zip(['Temperature (°C)', 'Sunshine (h)', 'Wind (m/s)', 'TrafficCount'], model.coef_):
            print(f"{feature}: {coef:.3f}")
        print(f"R² des Modells: {score:.3f}")
    else:
        print(f"\n⚠️ Nicht genügend Daten für Modelltraining ({label})")

    # 📈 6️⃣ Bürgerdashboard-Widget Plot
    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(df['TrafficCount'], df['O3 (ppb)'],
                          c=df['Temperature (°C)'], cmap='plasma', s=150, edgecolor='k')
    for idx, row in df.iterrows():
        plt.annotate(str(row['date']), (row['TrafficCount'], row['O3 (ppb)']),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

    plt.colorbar(scatter, label='Temperature (°C)')
    plt.xlabel("Traffic Count")
    plt.ylabel("O3 (ppb)")
    plt.title(f"Ozon vs. Verkehr (Farbkodiert: Temperatur) ({label})")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# =====================================================
# Ausführung für **09.–15.09.2024**
ozone_model_analysis(
    env_files=["sensordata_environment_2024-09-09_2024-09-15.json"],
    traffic_files=["sensordata_traffic_2024-09-09_2024-09-12.json", "sensordata_traffic_2024-09-13_2024-09-15.json"],
    visits_files=["sensordata_visits_2024-09-09_2024-09-15.json"],
    start_date=pd.to_datetime('2024-09-09').date(),
    end_date=pd.to_datetime('2024-09-15').date(),
    label="09.–15.09.2024"
)

# Ausführung für **20.–26.09.2024**
ozone_model_analysis(
    env_files=["sensordata_environment_2024-09-20_2024-09-26.json"],
    traffic_files=["sensordata_traffic_2024-09-20_2024-09-26.json"],
    visits_files=["sensordata_visits_2024-09-20_2024-09-26.json"],
    start_date=pd.to_datetime('2024-09-20').date(),
    end_date=pd.to_datetime('2024-09-26').date(),
    label="20.–26.09.2024"
)
import pandas as pd

df = pd.json_normalize(pd.read_json("sensordata_traffic_2024-09-09_2024-09-12.json")["sensordata"])
print(df.columns)
print(df.head())
import pandas as pd

# ----------------------------
# Dateien definieren
traffic_files = [
    "sensordata_traffic_2024-09-09_2024-09-12.json",
    "sensordata_traffic_2024-09-13_2024-09-15.json",
    "sensordata_traffic_2024-09-20_2024-09-26.json"
]

# Verkehrsdaten laden und Fahrzeuganzahl pro Tag zählen
traffic_dfs = []
for file in traffic_files:
    df = pd.json_normalize(pd.read_json(file)["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    traffic_daily = df.groupby('date').size().reset_index(name='Fahrzeuganzahl')
    traffic_dfs.append(traffic_daily)

# Alle Tage zusammenführen und summieren (falls doppelt)
traffic_all = pd.concat(traffic_dfs).groupby('date').sum().reset_index()

# Nach Datum sortieren
traffic_all = traffic_all.sort_values('date')

# Übersicht anzeigen
print("\n🚗 Übersicht der Fahrzeuganzahl pro Tag:")
print(traffic_all)
