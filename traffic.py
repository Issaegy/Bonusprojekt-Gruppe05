import pandas as pd
import matplotlib.pyplot as plt

# 1️⃣ Dateien definieren
dateien = [
    'sensordata_environment_2024-09-09_2024-09-15.json',
    'sensordata_environment_2024-09-20_2024-09-26.json'
]

# 2️⃣ Daten laden und zusammenführen
df_list = []
for datei in dateien:
    df_raw = pd.read_json(datei)
    df = pd.json_normalize(df_raw["sensordata"])
    df_list.append(df)

df_env = pd.concat(df_list, ignore_index=True)

# 3️⃣ Zeit vorbereiten
df_env['timestamp'] = pd.to_datetime(df_env['timestamp'])
df_env['date'] = df_env['timestamp'].dt.date

# 4️⃣ Relevante Sensorlabels
sensor_labels = {
    'dust_pm1': 'PM1 (μg/m³)',
    'dust_pm10': 'PM10 (μg/m³)',
    'dust_pm25': 'PM2.5 (μg/m³)',
    'no2': 'NO₂ (ppb)',
    'co': 'CO (ppb)',
    'temperature': 'Temperatur (°C)'
}

# 5️⃣ Tagesmittelwerte je Messgröße berechnen
env_summary = []
for label, description in sensor_labels.items():
    df_sensor = df_env[df_env['sensorLabel'] == label]
    if not df_sensor.empty:
        daily_avg = df_sensor.groupby('date')['value'].mean().reset_index(name=description)
        env_summary.append(daily_avg)
    else:
        print(f"⚠️ Keine Daten für {label} vorhanden.")

# 6️⃣ Tagesmittelwerte kombinieren
if env_summary:
    df_env_merged = env_summary[0]
    for df in env_summary[1:]:
        df_env_merged = pd.merge(df_env_merged, df, on='date', how='outer')
    df_env_merged = df_env_merged.sort_values('date').reset_index(drop=True)
    
    print("\n📊 Übersicht der Tagesdurchschnittswerte:")
    print(df_env_merged)

    # 7️⃣ Korrelationen berechnen
    corr_matrix = df_env_merged.drop(columns=['date']).corr()
    print("\n🔗 Korrelationen zwischen den Messgrößen:")
    print(corr_matrix.round(3))

    # 8️⃣ Grafiken mit Korrelation im Titel erstellen
    # Referenzparameter wählen, gegen den Korrelationen angezeigt werden
    referenz = 'Temperatur (°C)'  # <== hier Referenzparameter anpassen

    for col in df_env_merged.columns[1:]:
        if col == referenz:
            continue  # sich selbst nicht mit sich korrelieren

        corr_value = df_env_merged[referenz].corr(df_env_merged[col])

        plt.figure(figsize=(10, 5))
        plt.plot(df_env_merged['date'], df_env_merged[col], marker='o')
        plt.title(f"Tagesdurchschnitt {col} | Korrelation mit {referenz}: {corr_value:.2f}")
        plt.xlabel("Datum")
        plt.ylabel(col)
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # Optional: Heatmap der Korrelationen
    import seaborn as sns
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Korrelationen zwischen den Messgrößen")
    plt.tight_layout()
    plt.show()

else:
    print("⚠️ Keine Daten für die ausgewählten Messgrößen gefunden.")

import pandas as pd
import folium
from folium.plugins import MarkerCluster

# 1️⃣ Traffic-Dateien einlesen und kombinieren
traffic_files = [
    'sensordata_traffic_2024-09-09_2024-09-12.json',
    'sensordata_traffic_2024-09-13_2024-09-15.json',
    'sensordata_traffic_2024-09-20_2024-09-26.json'
]

df_list = []
for file in traffic_files:
    df_raw = pd.read_json(file)
    df = pd.json_normalize(df_raw["sensordata"])
    df_list.append(df)

df_traffic = pd.concat(df_list, ignore_index=True)

# 2️⃣ Vorverarbeitung
df_traffic['timestamp'] = pd.to_datetime(df_traffic['timestamp'])
df_traffic['date'] = df_traffic['timestamp'].dt.date

# Sicherstellen, dass lat/lon vorhanden sind
if 'lat' not in df_traffic.columns or 'lon' not in df_traffic.columns:
    print("❌ 'lat' oder 'lon' fehlt in den Daten.")
else:
    # 3️⃣ Mittelpunkt für Karte berechnen
    center_lat = df_traffic['lat'].mean()
    center_lon = df_traffic['lon'].mean()

    # 4️⃣ Folium-Karte erstellen
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)
    marker_cluster = MarkerCluster().add_to(m)

    # 5️⃣ Zusätzliche Information: Anzahl Autos pro Sensor (falls deviceName vorhanden)
    if 'deviceName' in df_traffic.columns:
        autos_per_device = df_traffic.groupby('deviceName').size().reset_index(name='anzahl_autos')
    else:
        autos_per_device = pd.DataFrame(columns=['deviceName', 'anzahl_autos'])

    # 6️⃣ Marker hinzufügen
    for idx, row in df_traffic.iterrows():
        lat = row['lat']
        lon = row['lon']
        street = row.get('street', 'Straße unbekannt')
        device = row.get('deviceName', 'Sensor unbekannt')
        timestamp = row['timestamp']
        additional_info = ""

        # Anzahl Autos pro Gerät hinzufügen
        anzahl_autos = autos_per_device[autos_per_device['deviceName'] == device]['anzahl_autos']
        autos_text = f"{int(anzahl_autos.values[0])} Fahrzeuge" if not anzahl_autos.empty else "n/v"

        # Alle weiteren relevanten Informationen hinzufügen
        for col in ['speed', 'direction', 'vehicleType']:
            if col in row and pd.notna(row[col]):
                additional_info += f"<br>{col}: {row[col]}"

        popup_text = (
            f"<b>🚦 Gerät:</b> {device}"
            f"<br><b>🛣️ Straße:</b> {street}"
            f"<br><b>📅 Zeitpunkt:</b> {timestamp}"
            f"<br><b>🚗 Anzahl Fahrzeuge (gesamt):</b> {autos_text}"
            f"{additional_info}"
        )

        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='blue', icon='car', prefix='fa')
        ).add_to(marker_cluster)

    # 7️⃣ Karte speichern
    m.save("traffic_map.html")
    print("✅ Interaktive Verkehrssensor-Karte als 'traffic_map.html' gespeichert. Im Browser öffnen, um alle Sensorpunkte zu sehen.")
