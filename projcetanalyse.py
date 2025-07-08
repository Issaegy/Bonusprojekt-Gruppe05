import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster

print("🚀 Starte die Analyse...", flush=True)

# ---------- 1. Dateien laden und vorbereiten ----------
def lade_und_annotiere(path, zeitraum_label):
    df_raw = pd.read_json(path)
    df = pd.json_normalize(df_raw["sensordata"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["Zeitraum"] = zeitraum_label
    return df

df1 = lade_und_annotiere("sensordata_environment_2024-09-09_2024-09-15.json", "09–15 Sep")
df2 = lade_und_annotiere("sensordata_environment_2024-09-20_2024-09-26.json", "20–26 Sep")
df_all = pd.concat([df1, df2], ignore_index=True)

print(f"📦 Gesamtanzahl Messwerte: {len(df_all)}")
print(f"🧪 Messgrößen: {df_all['sensorLabel'].nunique()}")
print(f"📍 Sensoren: {df_all['deviceId'].nunique()}")

# ---------- 2. Mittelwerte berechnen ----------
avg_values = df_all.groupby(
    ["deviceId", "deviceName", "lat", "lon", "sensorLabel", "unit", "Zeitraum"]
)["value"].mean().reset_index()

pivot = avg_values.pivot_table(
    index=["deviceId", "deviceName", "lat", "lon", "sensorLabel", "unit"],
    columns="Zeitraum",
    values="value"
).reset_index()

# Differenz berechnen
if "09–15 Sep" in pivot.columns and "20–26 Sep" in pivot.columns:
    pivot["Δ"] = pivot["20–26 Sep"] - pivot["09–15 Sep"]
else:
    pivot["Δ"] = 0

# ---------- 3. Interaktive Karte erstellen ----------
m = folium.Map(location=[pivot["lat"].mean(), pivot["lon"].mean()], zoom_start=6)
cluster = MarkerCluster().add_to(m)

sensor_legenden = {}

for _, row in pivot.iterrows():
    name = row["deviceName"]
    beschreibung = row["sensorLabel"]
    einheit = row["unit"]

    sensor_legenden.setdefault(name, set()).add(beschreibung)

    # Farblogik: Rot = Anstieg, Grün = Rückgang
    delta = row["Δ"]
    color = "green" if delta < 0 else "red"

    popup = f"""
    <b>Sensor:</b> {name}<br>
    <b>Messgröße:</b> {beschreibung} ({einheit})<br>
    <b>09–15 Sep:</b> {row.get('09–15 Sep', 'n/a'):.2f}<br>
    <b>20–26 Sep:</b> {row.get('20–26 Sep', 'n/a'):.2f}<br>
    <b>Δ:</b> {delta:.2f}
    """

    folium.CircleMarker(
        location=(row["lat"], row["lon"]),
        radius=6,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=popup,
    ).add_to(cluster)

# ---------- 4. Legende zur Karte hinzufügen ----------
legende_html = """
<div style="
    position: fixed;
    bottom: 30px;
    left: 30px;
    width: 280px;
    height: auto;
    z-index: 9999;
    background-color: white;
    border:2px solid grey;
    border-radius: 8px;
    padding: 10px;
    font-size: 14px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.3);
">
    <b>🧪 Sensor-Messgrößen</b><br>
    <hr style="margin:5px 0;">
    <ul style="list-style: none; padding-left: 0; margin: 0;">
        <li><b>🌡 Temperatur</b> (°C)</li>
        <li><b>🌬 Windgeschwindigkeit</b> (m/s)</li>
        <li><b>💧 Luftfeuchtigkeit</b> (%)</li>
        <li><b>🌫 Luftdruck</b> (hPa)</li>
        <li><b>🌧 Niederschlag</b> (mm)</li>
        <li><b>☁️ Feinstaub</b> (PM₁ / PM₂.₅ / PM₁₀)</li>
        <li><b>🧪 Gase</b> (NO, NO₂, CO, O₃ in ppb)</li>
        <li><b>🌍 Luftqualitätsindex</b> (AQI)</li>
    </ul>
    <hr style="margin:5px 0;">
    <span style="color:red;">🔴 = Anstieg</span><br>
    <span style="color:green;">🟢 = Rückgang</span>
</div>
"""

m.get_root().html.add_child(folium.Element(legende_html))

# ---------- 5. Karte speichern ----------
m.save("sensor_vergleichskarte.html")
print("🗺 Karte gespeichert als 'sensor_vergleichskarte.html'")

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# 1. JSON-Datei laden
def lade_daten(pfad):
    df_raw = pd.read_json(pfad)
    df = pd.json_normalize(df_raw["sensordata"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = lade_daten("sensordata_environment_2024-09-09_2024-09-15.json")

# 2. Reverse Geocoding vorbereiten
geolocator = Nominatim(user_agent="sensoranalyse")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

# 3. Standortnamen pro Gerät speichern (einmalig)
standorte = {}

for deviceId, gruppe in df.groupby("deviceId"):
    lat = gruppe["lat"].iloc[0]
    lon = gruppe["lon"].iloc[0]
    location = reverse((lat, lon), language="de")

    if location and location.raw.get("address"):
        addr = location.raw["address"]
        straße = addr.get("road") or addr.get("pedestrian") or addr.get("suburb") or "Unbekannte Straße"
    else:
        straße = f"{lat:.3f}, {lon:.3f}"

    standorte[deviceId] = straße

# 4. Alle sensorLabels durchgehen und je einen Plot erzeugen
for sensorLabel, sensor_df in df.groupby("sensorLabel"):
    plt.figure(figsize=(14, 6))
    einheit = sensor_df["unit"].iloc[0]
    geräte = sensor_df["deviceId"].unique()
    farben = cm.get_cmap("tab10", len(geräte))

    for i, (deviceId, device_data) in enumerate(sensor_df.groupby("deviceId")):
        device_name = device_data["deviceName"].iloc[0]
        adresse = standorte.get(deviceId, "Unbekannt")

        legenden_label = f"{device_name} – {sensorLabel} ({einheit})\n📍{adresse}"

        plt.plot(
            device_data["timestamp"],
            device_data["value"],
            label=legenden_label,
            color=farben(i),
            marker="o",
            linestyle="-",
            markersize=2,
            linewidth=1,
        )

    plt.title(f"📈 Zeitverlauf: {sensorLabel} ({einheit})", fontsize=16)
    plt.xlabel("Zeit")
    plt.ylabel(f"Wert ({einheit})")
    plt.legend(title="Sensorgerät & Standort", fontsize=9)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

print("Neue Ananlyse gestartet....")

def lade_daten(pfad):
    df_raw = pd.read_json(pfad)
    df = pd.json_normalize(df_raw["sensordata"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

df = lade_daten("sensordata_environment_2024-09-20_2024-09-26.json")

# 2. Reverse Geocoding vorbereiten
geolocator = Nominatim(user_agent="sensoranalyse")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

# 3. Standortnamen pro Gerät speichern (einmalig)
standorte = {}

for deviceId, gruppe in df.groupby("deviceId"):
    lat = gruppe["lat"].iloc[0]
    lon = gruppe["lon"].iloc[0]
    location = reverse((lat, lon), language="de")

    if location and location.raw.get("address"):
        addr = location.raw["address"]
        straße = addr.get("road") or addr.get("pedestrian") or addr.get("suburb") or "Unbekannte Straße"
    else:
        straße = f"{lat:.3f}, {lon:.3f}"

    standorte[deviceId] = straße

# 4. Alle sensorLabels durchgehen und je einen Plot erzeugen
for sensorLabel, sensor_df in df.groupby("sensorLabel"):
    plt.figure(figsize=(14, 6))
    einheit = sensor_df["unit"].iloc[0]
    geräte = sensor_df["deviceId"].unique()
    farben = cm.get_cmap("tab10", len(geräte))

    for i, (deviceId, device_data) in enumerate(sensor_df.groupby("deviceId")):
        device_name = device_data["deviceName"].iloc[0]
        adresse = standorte.get(deviceId, "Unbekannt")

        legenden_label = f"{device_name} – {sensorLabel} ({einheit})\n📍{adresse}"

        plt.plot(
            device_data["timestamp"],
            device_data["value"],
            label=legenden_label,
            color=farben(i),
            marker="o",
            linestyle="-",
            markersize=2,
            linewidth=1,
        )

    plt.title(f"📈 Zeitverlauf: {sensorLabel} ({einheit})", fontsize=16)
    plt.xlabel("Zeit")
    plt.ylabel(f"Wert ({einheit})")
    plt.legend(title="Sensorgerät & Standort", fontsize=9)
    plt.tight_layout()
    plt.grid(True)
    plt.show()

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# 1. Daten zusammenführen (falls noch nicht geschehen)
# df_all = pd.concat([df1, df2], ignore_index=True)

# 2. Reverse Geocoding vorbereiten
geolocator = Nominatim(user_agent="sensoranalyse")
reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

# 3. Adressen für jedes Gerät bestimmen
standorte = {}
for deviceId, gruppe in df_all.groupby("deviceId"):
    lat = gruppe["lat"].iloc[0]
    lon = gruppe["lon"].iloc[0]
    location = reverse((lat, lon), language="de")
    if location and location.raw.get("address"):
        addr = location.raw["address"]
        straße = addr.get("road") or addr.get("pedestrian") or addr.get("suburb") or "Unbekannte Straße"
    else:
        straße = f"{lat:.3f}, {lon:.3f}"
    standorte[deviceId] = straße

# 4. Plot für jede sensorLabel-Gruppe
for sensorLabel, sensor_df in df_all.groupby("sensorLabel"):
    plt.figure(figsize=(14, 6))
    einheit = sensor_df["unit"].iloc[0]
    geräte = sensor_df["deviceId"].unique()
    farben = cm.get_cmap("tab10", len(geräte))

    for i, deviceId in enumerate(geräte):
        for zeitraum, zeitraum_df in sensor_df[sensor_df["deviceId"] == deviceId].groupby("Zeitraum"):
            if zeitraum_df.empty:
                continue
            device_name = zeitraum_df["deviceName"].iloc[0]
            adresse = standorte.get(deviceId, "Unbekannt")
            farbe = farben(i)

            label = f"{device_name} – {sensorLabel} ({einheit})\n📍{adresse} – {zeitraum}"

            plt.plot(
                zeitraum_df["timestamp"],
                zeitraum_df["value"],
                label=label,
                color=farbe,
                linestyle="--" if zeitraum == "09–15 Sep" else "-",  # verschiedene Linien für Vergleich
                marker="o",
                markersize=3,
                linewidth=1,
            )

    plt.title(f"📈 Vergleich: {sensorLabel} ({einheit})", fontsize=16)
    plt.xlabel("Zeit")
    plt.ylabel(f"Wert ({einheit})")
    plt.legend(title="Sensor & Zeitraum", fontsize=9)
    plt.tight_layout()
    plt.grid(True)
    plt.show()
