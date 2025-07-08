import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Funktion zur Analyse von Besuchsdaten
def analyze_visits(file_name, label):
    print(f"\n📂 Datei: {file_name} ({label})")

    # Laden und Normalisieren
    df_raw = pd.read_json(file_name)
    df = pd.json_normalize(df_raw["sensordata"])

    # Alle Spalten ausgeben
    print("\n📋 Alle Spalten:")
    print(df.columns.tolist())

    # Zeitspalte vorbereiten
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # ---------------------------
    # 1️⃣ Besucher pro Tag
    visitors_per_day = df.groupby('date')['visitors'].sum().reset_index()
    print("\n🚶‍♂️ Besucher pro Tag:")
    print(visitors_per_day)

    # Plot Besucher pro Tag
    plt.figure(figsize=(8,5))
    plt.bar(visitors_per_day['date'].astype(str), visitors_per_day['visitors'], color='skyblue')
    plt.title(f"Besucher pro Tag ({label})")
    plt.xlabel("Datum")
    plt.ylabel("Anzahl Besucher")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # 2️⃣ Durchschnittliche Aufenthaltsdauer pro Tag
    duration_per_day = df.groupby('date')['avgDuration'].mean().reset_index()
    print("\n⏱ Durchschnittliche Aufenthaltsdauer pro Tag (Minuten):")
    print(duration_per_day)

    # Plot Aufenthaltsdauer
    plt.figure(figsize=(8,5))
    plt.plot(duration_per_day['date'].astype(str), duration_per_day['avgDuration'], marker='o')
    plt.title(f"Ø Aufenthaltsdauer pro Tag ({label})")
    plt.xlabel("Datum")
    plt.ylabel("Ø Aufenthaltsdauer (Minuten)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # 3️⃣ Besucher pro Standort
    visitors_per_location = df.groupby('name')['visitors'].sum().reset_index().sort_values('visitors', ascending=False)
    print("\n📍 Besucher pro Standort (gesamt im Zeitraum):")
    print(visitors_per_location)

    # Plot Besucher pro Standort (Top 10)
    plt.figure(figsize=(10,6))
    top_locations = visitors_per_location.head(10)
    plt.barh(top_locations['name'], top_locations['visitors'], color='seagreen')
    plt.title(f"Top 10 Standorte nach Besuchern ({label})")
    plt.xlabel("Anzahl Besucher")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # ---------------------------
    # 4️⃣ Heatmap-Vorbereitung: Besucher pro GPS-Punkt
    gps_visits = df.groupby(['lat', 'lon'])['visitors'].sum().reset_index()
    print("\n🗺️ Besucher pro GPS-Punkt (für Heatmap-Verwendung):")
    print(gps_visits.head())

    # Optional: Rückgabe für spätere Vergleiche oder Heatmap-Erstellung
    return visitors_per_day, duration_per_day, visitors_per_location, gps_visits

# ---------------------------
# Analyse für 09.–15.09.2024
visits_1 = analyze_visits(
    "sensordata_visits_2024-09-09_2024-09-15.json",
    "09.–15.09.2024"
)

# ---------------------------
# Analyse für 20.–26.09.2024
visits_2 = analyze_visits(
    "sensordata_visits_2024-09-20_2024-09-26.json",
    "20.–26.09.2024"
)
import pandas as pd
import folium

# ---------------------------
# Funktion zur Erstellung der Besuchskarte mit Farbwahl
def create_visits_map(file_name, label, output_file, marker_color):
    print(f"\n📂 Erstelle Besuchskarte für: {file_name}")

    # Daten laden
    df_raw = pd.read_json(file_name)
    df = pd.json_normalize(df_raw["sensordata"])

    # Zeitspalte
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Gruppierung nach Standort
    df_grouped = df.groupby(['name', 'lat', 'lon']).agg({
        'visitors': 'sum',
        'avgDuration': 'mean'
    }).reset_index()

    # Zentrum berechnen
    center_lat = df_grouped['lat'].mean()
    center_lon = df_grouped['lon'].mean()

    # Interaktive Karte
    visit_map = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)

    # Marker pro Standort
    for idx, row in df_grouped.iterrows():
        popup_text = (
            f"<b>📍 Standort:</b> {row['name']}<br>"
            f"<b>👥 Besucher:</b> {int(row['visitors'])}<br>"
            f"<b>⏱ Ø Aufenthaltsdauer:</b> {row['avgDuration']:.1f} min<br>"
            f"<b>🌐 Koordinaten:</b> ({row['lat']:.5f}, {row['lon']:.5f})"
        )

        radius = max(5, min(row['visitors'] / 10, 50))

        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=radius,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.6,
            popup=folium.Popup(popup_text, max_width=300)
        ).add_to(visit_map)

    # Karte speichern
    visit_map.save(output_file)
    print(f"✅ Besuchskarte erstellt und gespeichert als: {output_file}")

# ---------------------------
# Besuchskarte für 09.–15.09.2024 (Gelb)
create_visits_map(
    "sensordata_visits_2024-09-09_2024-09-15.json",
    "09.–15.09.2024",
    "visits_map1.html",
    marker_color='yellow'
)

# ---------------------------
# Besuchskarte für 20.–26.09.2024 (Blau)
create_visits_map(
    "sensordata_visits_2024-09-20_2024-09-26.json",
    "20.–26.09.2024",
    "visits_map2.html",
    marker_color='blue'
)

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------
# Funktion zur Analyse der Top-Hotspots
def analyze_top_hotspots(file_name, label):
    print(f"\n📂 Analysiere Datei: {file_name} ({label})")

    # Daten laden
    df_raw = pd.read_json(file_name)
    df = pd.json_normalize(df_raw["sensordata"])

    # Zeit vorbereiten
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Gruppierung: Besucher pro Standort summieren
    visitors_per_location = (
        df.groupby('name')['visitors']
        .sum()
        .reset_index()
        .sort_values('visitors', ascending=False)
    )

    print("\n📊 Top 10 Standorte nach Besucherzahl:")
    print(visitors_per_location.head(10))

    # Visualisierung: Balkendiagramm der Top 10
    top10 = visitors_per_location.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top10['name'], top10['visitors'], color='skyblue')
    plt.xlabel("Anzahl Besucher")
    plt.title(f"Top 10 Standorte nach Besucherzahl ({label})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# ---------------------------
# Analyse für 09.–15.09.2024
analyze_top_hotspots(
    "sensordata_visits_2024-09-09_2024-09-15.json",
    "09.–15.09.2024"
)

# ---------------------------
# Analyse für 20.–26.09.2024
analyze_top_hotspots(
    "sensordata_visits_2024-09-20_2024-09-26.json",
    "20.–26.09.2024"
)
import pandas as pd
import matplotlib.pyplot as plt

def analyze_avg_duration(file_name, label):
    print(f"\n📂 Analysiere Aufenthaltsdauer: {file_name} ({label})")

    # Daten laden
    df_raw = pd.read_json(file_name)
    df = pd.json_normalize(df_raw["sensordata"])

    # Zeit vorbereiten
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Durchschnittliche Aufenthaltsdauer pro Standort
    avg_duration_per_location = (
        df.groupby('name')['avgDuration']
        .mean()
        .reset_index()
        .sort_values('avgDuration', ascending=False)
    )

    print("\n⏱ Top 10 Standorte mit längster Aufenthaltsdauer:")
    print(avg_duration_per_location.head(10))

    # Visualisierung
    top10 = avg_duration_per_location.head(10)
    plt.figure(figsize=(10, 6))
    plt.barh(top10['name'], top10['avgDuration'], color='salmon')
    plt.xlabel("Ø Aufenthaltsdauer (Minuten)")
    plt.title(f"Top 10 Aufenthaltsdauer pro Standort ({label})")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Analyse Aufenthaltsdauer für beide Wochen
analyze_avg_duration(
    "sensordata_visits_2024-09-09_2024-09-15.json",
    "09.–15.09.2024"
)

analyze_avg_duration(
    "sensordata_visits_2024-09-20_2024-09-26.json",
    "20.–26.09.2024"
)
import pandas as pd
import folium
from folium.plugins import HeatMap

# ---------------------------
# Funktion zur Heatmap-Erstellung
def create_visits_heatmap(file_name, label, output_file):
    print(f"\n📂 Erstelle Heatmap für: {file_name} ({label})")

    # Daten laden
    df_raw = pd.read_json(file_name)
    df = pd.json_normalize(df_raw["sensordata"])

    # Zeitspalte
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # GPS + Besucher extrahieren
    heat_data = df[['lat', 'lon', 'visitors']].dropna()
    heat_data = heat_data[heat_data['visitors'] > 0]  # nur valide Punkte

    # Daten in (lat, lon, weight) umwandeln
    heat_list = heat_data[['lat', 'lon', 'visitors']].values.tolist()

    # Kartenmittelpunkt bestimmen
    center_lat = heat_data['lat'].mean()
    center_lon = heat_data['lon'].mean()

    # Karte initialisieren
    visit_map = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)

    # Heatmap Layer hinzufügen
    HeatMap(
        heat_list,
        min_opacity=0.4,
        max_zoom=18,
        radius=20,
        blur=15,
        gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 0.8: 'red'}
    ).add_to(visit_map)

    # Speichern
    visit_map.save(output_file)
    print(f"✅ Heatmap erstellt und gespeichert als: {output_file}")

# ---------------------------
# Heatmap für 09.–15.09.2024
create_visits_heatmap(
    "sensordata_visits_2024-09-09_2024-09-15.json",
    "09.–15.09.2024",
    "visits_heatmap_09_15_09_2024.html"
)

# ---------------------------
# Heatmap für 20.–26.09.2024
create_visits_heatmap(
    "sensordata_visits_2024-09-20_2024-09-26.json",
    "20.–26.09.2024",
    "visits_heatmap_20_26_09_2024.html"
)
import pandas as pd
import folium
from folium.plugins import MarkerCluster

# ---------------------------
# Funktion zur Erstellung einer Clusterkarte
def create_visits_clustermap(file_name, label, output_file):
    print(f"\n📂 Erstelle Clusterkarte für: {file_name} ({label})")

    # Daten laden
    df_raw = pd.read_json(file_name)
    df = pd.json_normalize(df_raw["sensordata"])

    # Zeitspalte
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date

    # Gruppierung nach Standort
    df_grouped = df.groupby(['name', 'lat', 'lon']).agg({
        'visitors': 'sum',
        'avgDuration': 'mean'
    }).reset_index()

    # Kartenmittelpunkt
    center_lat = df_grouped['lat'].mean()
    center_lon = df_grouped['lon'].mean()

    # Karte initialisieren
    visit_map = folium.Map(location=[center_lat, center_lon], zoom_start=13, control_scale=True)

    # Cluster hinzufügen
    marker_cluster = MarkerCluster().add_to(visit_map)

    # Marker pro Standort hinzufügen
    for idx, row in df_grouped.iterrows():
        popup_text = (
            f"<b>📍 Standort:</b> {row['name']}<br>"
            f"<b>👥 Besucher:</b> {int(row['visitors'])}<br>"
            f"<b>⏱ Ø Aufenthaltsdauer:</b> {row['avgDuration']:.1f} min<br>"
            f"<b>🌐 Koordinaten:</b> ({row['lat']:.5f}, {row['lon']:.5f})"
        )
        folium.Marker(
            location=[row['lat'], row['lon']],
            popup=folium.Popup(popup_text, max_width=300),
            icon=folium.Icon(color='blue' if label == "09.–15.09.2024" else 'green', icon='info-sign')
        ).add_to(marker_cluster)

    # Speichern
    visit_map.save(output_file)
    print(f"✅ Clusterkarte erstellt und gespeichert als: {output_file}")

# ---------------------------
# Clusterkarte für 09.–15.09.2024
create_visits_clustermap(
    "sensordata_visits_2024-09-09_2024-09-15.json",
    "09.–15.09.2024",
    "visits_clustermap_09_15_09_2024.html"
)

# ---------------------------
# Clusterkarte für 20.–26.09.2024
create_visits_clustermap(
    "sensordata_visits_2024-09-20_2024-09-26.json",
    "20.–26.09.2024",
    "visits_clustermap_20_26_09_2024.html"
)
import pandas as pd

# Besucherdaten laden
def load_visitors(file):
    df_raw = pd.read_json(file)
    df = pd.json_normalize(df_raw["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    visitors_per_day = df.groupby('date')['visitors'].sum().reset_index()
    visitors_per_day.columns = ['date', 'visitors']
    return visitors_per_day

# CO-Daten laden
def load_co(file):
    df_raw = pd.read_json(file)
    df = pd.json_normalize(df_raw["sensordata"])
    df = df[df['sensorLabel'] == 'co']
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    co_per_day = df.groupby('date')['value'].mean().reset_index()
    co_per_day.columns = ['date', 'co_avg']
    return co_per_day

# ---------------------------
# Dateien anpassen:
visitors_file = "sensordata_visits_2024-09-20_2024-09-26.json"
co_file = "sensordata_environment_2024-09-20_2024-09-26.json"

# Daten laden
visitors = load_visitors(visitors_file)
co = load_co(co_file)

# Daten zusammenführen
combined = pd.merge(visitors, co, on='date', how='inner')
print("\n📊 Übersicht Besucher und CO pro Tag:")
print(combined)

# Korrelation berechnen
correlation = combined['visitors'].corr(combined['co_avg'])
print(f"\n🔗 Korrelation Besucheranzahl vs. CO-Wert: {correlation:.3f}")
import pandas as pd

# Besucherdaten laden
def load_visitors(file):
    df_raw = pd.read_json(file)
    df = pd.json_normalize(df_raw["sensordata"])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    visitors_per_day = df.groupby('date')['visitors'].sum().reset_index()
    visitors_per_day.columns = ['date', 'visitors']
    return visitors_per_day

# Umweltparameter laden
def load_environment(file, sensor_label, column_name):
    df_raw = pd.read_json(file)
    df = pd.json_normalize(df_raw["sensordata"])
    df = df[df['sensorLabel'] == sensor_label]
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    env_per_day = df.groupby('date')['value'].mean().reset_index()
    env_per_day.columns = ['date', column_name]
    return env_per_day

# ---------------------------
# Dateien anpassen
visitors_file = "sensordata_visits_2024-09-20_2024-09-26.json"
env_file = "sensordata_environment_2024-09-20_2024-09-26.json"

# Besucher laden
visitors = load_visitors(visitors_file)

# Umweltparameter laden
pm1 = load_environment(env_file, "dust_pm1", "pm1")
pm10 = load_environment(env_file, "dust_pm10", "pm10")
pm25 = load_environment(env_file, "dust_pm25", "pm25")
no2 = load_environment(env_file, "no2", "no2")
co = load_environment(env_file, "co", "co")
temp = load_environment(env_file, "temperature", "temperature")
rain = load_environment(env_file, "rainintensity", "rain")
wind = load_environment(env_file, "windspeed", "wind")

# Daten zusammenführen
combined = visitors
for df in [pm1, pm10, pm25, no2, co, temp, rain, wind]:
    combined = pd.merge(combined, df, on='date', how='left')

print("\n📊 Übersicht Besucher & Umweltparameter pro Tag:")
print(combined)

# Korrelationen berechnen
print("\n🔗 Korrelationen mit Besucheranzahl:")
for col in combined.columns[1:]:
    corr = combined['visitors'].corr(combined[col])
    print(f"Korrelation Besucher vs. {col}: {corr:.3f}")
import pandas as pd

def check_rain_on_date(env_file, check_date):
    print(f"\n📂 Datei: {env_file} | 📅 Datum: {check_date}")

    try:
        # Laden und Normalisieren
        df_raw = pd.read_json(env_file)
        df = pd.json_normalize(df_raw["sensordata"])

        # Filtern auf 'rainintensity'
        df_rain = df[df['sensorLabel'] == 'rainintensity']

        # Zeit aufbereiten
        df_rain['timestamp'] = pd.to_datetime(df_rain['timestamp'])
        df_rain['date'] = df_rain['timestamp'].dt.date

        # Auf gewünschten Tag filtern
        check_date_obj = pd.to_datetime(check_date).date()
        df_rain_day = df_rain[df_rain['date'] == check_date_obj]

        if not df_rain_day.empty:
            avg_rain = df_rain_day['value'].mean()
            print(f"🌧️ Durchschnittliche Regenintensität am {check_date}: {avg_rain:.3f} mm/h")
        else:
            print(f"✅ Kein Regen am {check_date} in den Daten erfasst.")
    except Exception as e:
        print(f"❌ Fehler beim Verarbeiten von {env_file}: {e}")

# ---------------------------
# 16.09.2024 prüfen
check_rain_on_date(
    "sensordata_environment_2024-09-09_2024-09-15.json",
    "2024-09-16"
)

# ---------------------------
# 26.09.2024 prüfen
check_rain_on_date(
    "sensordata_environment_2024-09-20_2024-09-26.json",
    "2024-09-26"
)
import pandas as pd

def analyze_rain_vs_dust(env_file, label):
    print(f"\n📊 Analyse für {label} ({env_file})")

    try:
        # Laden und Normalisieren
        df_raw = pd.read_json(env_file)
        df = pd.json_normalize(df_raw["sensordata"])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date

        # Funktion: Tagesdurchschnitt je SensorLabel
        def daily_avg(sensor_label, col_name):
            df_temp = df[df['sensorLabel'] == sensor_label]
            return df_temp.groupby('date')['value'].mean().reset_index().rename(columns={'value': col_name})

        rain = daily_avg('rainintensity', 'rain')
        pm1 = daily_avg('dust_pm1', 'pm1')
        pm10 = daily_avg('dust_pm10', 'pm10')
        pm25 = daily_avg('dust_pm25', 'pm25')

        # Zusammenführen
        combined = rain.merge(pm1, on='date', how='outer') \
                       .merge(pm10, on='date', how='outer') \
                       .merge(pm25, on='date', how='outer') \
                       .fillna(0)

        print("\n🌧️ Regen und Feinstaub pro Tag:")
        print(combined)

        # Korrelationen berechnen
        corr_pm1 = combined['rain'].corr(combined['pm1'])
        corr_pm10 = combined['rain'].corr(combined['pm10'])
        corr_pm25 = combined['rain'].corr(combined['pm25'])

        print(f"\n🔗 Korrelation Regen vs. PM1: {corr_pm1:.3f}")
        print(f"🔗 Korrelation Regen vs. PM10: {corr_pm10:.3f}")
        print(f"🔗 Korrelation Regen vs. PM2.5: {corr_pm25:.3f}")

        return combined

    except Exception as e:
        print(f"❌ Fehler beim Verarbeiten: {e}")

# ---------------------------
# Zeitraum 09.–15.09.2024
df_period1 = analyze_rain_vs_dust(
    "sensordata_environment_2024-09-09_2024-09-15.json",
    "09.–15.09.2024"
)

# ---------------------------
# Zeitraum 20.–26.09.2024
df_period2 = analyze_rain_vs_dust(
    "sensordata_environment_2024-09-20_2024-09-26.json",
    "20.–26.09.2024"
)
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

visits_files = [
    "sensordata_visits_2024-09-09_2024-09-15.json",
    "sensordata_visits_2024-09-20_2024-09-26.json"
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
# AQI nach Einheit aufteilen
df_us = env_all[(env_all['sensorLabel'] == 'airqualityindex') & (env_all['unit'] == 'USAEPA_AirNow')]
df_us = df_us.groupby('date')['value'].mean().reset_index(name='AQI_USA')

df_eea = env_all[(env_all['sensorLabel'] == 'airqualityindex') & (env_all['unit'] == 'EEA_EAQI')]
df_eea = df_eea.groupby('date')['value'].mean().reset_index(name='AQI_EEA')

# ----------------------------
# Umweltdaten filtern und Tagesmittel berechnen
params = ['temperature', 'windspeed', 'pressure', 'co', 'no', 'no2', 'o3']
env_filtered = env_all[env_all['sensorLabel'].isin(params)]
env_daily = env_filtered.groupby(['date', 'sensorLabel'])['value'].mean().unstack().reset_index()
env_daily = env_daily.rename(columns={
    'temperature': 'Temperatur (°C)',
    'windspeed': 'Windgeschwindigkeit (m/s)',
    'pressure': 'Luftdruck (hPa)',
    'co': 'CO (ppb)',
    'no': 'NO (ppb)',
    'no2': 'NO2 (ppb)',
    'o3': 'O3 (ppb)'
})

# AQI hinzufügen
env_daily = env_daily.merge(df_us, on='date', how='left')
env_daily = env_daily.merge(df_eea, on='date', how='left')

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
# Merge Besucher + Umweltdaten
merged_df = pd.merge(env_daily, visits_all, on='date', how='inner')

print("\n📊 Übersicht der kombinierten Tagesdaten:")
print(merged_df)

# ----------------------------
# Korrelationen prüfen
correlation_targets = [
    'Temperatur (°C)',
    'Windgeschwindigkeit (m/s)',
    'Luftdruck (hPa)',
    'AQI_USA',
    'AQI_EEA',
    'CO (ppb)',
    'NO (ppb)',
    'NO2 (ppb)',
    'O3 (ppb)'
]

print("\n✅ Korrelationen zwischen Besucheranzahl und Umweltdaten:")
for target in correlation_targets:
    corr, p = pearsonr(merged_df['Besucheranzahl'], merged_df[target])
    direction = "↑ steigend" if corr > 0 else "↓ fallend"
    significance = "✅ signifikant" if p < 0.05 else "⚠️ nicht signifikant"
    print(f"Besucheranzahl vs. {target}: r = {corr:.3f}, p = {p:.5f} ({direction}, {significance})")

# ----------------------------
# Heatmap
heatmap_vars = ['Besucheranzahl'] + correlation_targets
plt.figure(figsize=(10, 8))
sns.heatmap(merged_df[heatmap_vars].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Korrelationsmatrix: Besucheranzahl vs. Umweltdaten')
plt.tight_layout()
plt.show()

# ----------------------------
# Scatterplots
for target in correlation_targets:
    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=merged_df,
        x='Besucheranzahl',
        y=target,
        scatter_kws={'s': 80, 'edgecolor': 'black'},
        line_kws={'color': 'red'}
    )
    plt.xlabel('Besucheranzahl')
    plt.ylabel(f'{target}')
    plt.title(f'Besucheranzahl vs. {target}')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
