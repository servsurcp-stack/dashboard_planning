import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import linregress
from wordcloud import WordCloud
import folium
from streamlit_folium import folium_static
import numpy as np

@st.cache_data(ttl=300)
def load_data(query="SELECT * FROM db_planning;"):
    conn = st.connection("postgresql", type="sql")
    df = conn.query(query, ttl="10m")
    # Normalisations utiles
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "mois" in df.columns:
        df["mois"] = df["mois"].astype(str)
    if "visite" in df.columns:
        df["visite"] = df["visite"].astype(str)
    if "agent" in df.columns:
        df["agent"] = df["agent"].astype(str)
    if "visitetype" in df.columns:
        df["visitetype"] = df["visitetype"].astype(str)
    if "lieu" in df.columns:
        df["lieu"] = df["lieu"].astype(str)
    return df

def compute_passages_by_lieu(df):
    agg = (
        df.groupby(["lieu", "visitetype", "agent"], dropna=False)
          .agg(passages=("passage_id", "count"))
          .reset_index()
    )
    return agg

def compute_passages_by_weekday(df):
    df["weekday"] = df["date"].dt.day_name()
    weekday_map = {
        "Monday": "Lundi",
        "Tuesday": "Mardi",
        "Wednesday": "Mercredi",
        "Thursday": "Jeudi",
        "Friday": "Vendredi",
        "Saturday": "Samedi",
        "Sunday": "Dimanche"
    }
    df["weekday"] = df["weekday"].map(weekday_map)
    days_order = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    agg_weekday = (
        df.groupby("weekday", dropna=False)
          .agg(passages=("passage_id", "count"))
          .reset_index()
    )
    agg_weekday["weekday"] = pd.Categorical(agg_weekday["weekday"], categories=days_order, ordered=True)
    agg_weekday = agg_weekday.sort_values("weekday")
    return agg_weekday

def main():
    st.title("Dashboard Passages par Visite")

    st.sidebar.header("Filtrer les données")
    df = load_data()

    if df.empty:
        st.warning("Aucune donnée récupérée. Vérifie le nom de la table et la connexion.")
        return

    st.sidebar.subheader("Plage de dates")
    min_date = df["date"].min().to_pydatetime().date() if not df["date"].isna().all() else datetime(2025, 1, 1).date()
    max_date = df["date"].max().to_pydatetime().date() if not df["date"].isna().all() else datetime(2025, 12, 31).date()

    start_date = st.sidebar.date_input(
        "Date de début",
        value=min_date,
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD"
    )

    end_date = st.sidebar.date_input(
        "Date de fin",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        format="YYYY-MM-DD"
    )

    if end_date < start_date:
        st.sidebar.error("La date de fin doit être postérieure ou égale à la date de début.")
        return

    df_filtered = df[
        (df["date"] >= pd.Timestamp(start_date)) &
        (df["date"] <= pd.Timestamp(end_date))
    ]

    agents_list = sorted(df_filtered["agent"].dropna().unique().tolist())
    visitetype_list = sorted(df_filtered["visitetype"].dropna().unique().tolist())
    lieu_list = sorted(df_filtered["lieu"].dropna().unique().tolist())

    selected_agents = st.sidebar.multiselect("Agent", options=agents_list, default=agents_list)
    selected_visitetype = st.sidebar.multiselect("Type de visite", options=visitetype_list, default=visitetype_list)
    selected_lieux = st.sidebar.multiselect("Lieu", options=lieu_list, default=lieu_list)

    df_filtered = df_filtered[
        df_filtered["agent"].isin(selected_agents) &
        df_filtered["visitetype"].isin(selected_visitetype) &
        df_filtered["lieu"].isin(selected_lieux)
    ]

    if df_filtered.empty:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        return

    agg_lieu = compute_passages_by_lieu(df_filtered)

    st.markdown("**Nombre de passages par lieu**")
    st.dataframe(agg_lieu.sort_values("passages", ascending=False).reset_index(drop=True).head(200))

    chart_lieu = (
        alt.Chart(agg_lieu)
        .mark_bar()
        .encode(
            x=alt.X("passages:Q", title="Nombre de passages"),
            y=alt.Y("lieu:N", sort='-x', title="Lieu"),
            color=alt.Color("visitetype:N", title="Type de visite"),
            tooltip=["lieu", "visitetype", "agent", "passages"]
        )
        .properties(height=500)
    )

    st.altair_chart(chart_lieu, use_container_width=True)

    csv_lieu = agg_lieu.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger les agrégats par lieu CSV", data=csv_lieu, file_name="passages_par_lieu.csv", mime="text/csv")

    st.markdown("**Fréquence des passages par jour de la semaine**")
    agg_weekday = compute_passages_by_weekday(df_filtered)

    chart_weekday = (
        alt.Chart(agg_weekday)
        .mark_bar()
        .encode(
            x=alt.X("weekday:N", title="Jour de la semaine", sort=["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]),
            y=alt.Y("passages:Q", title="Nombre de passages"),
            tooltip=["weekday", "passages"]
        )
        .properties(height=300)
    )

    st.altair_chart(chart_weekday, use_container_width=True)

    st.markdown("### Feature 1: Heatmap de Couverture des Lieux par Agent et Période")
    pivot = df_filtered.pivot_table(index='agent', columns='jour', values='passage_id', aggfunc='count', fill_value=0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, cmap='YlGnBu', ax=ax)
    st.pyplot(fig)

    st.markdown("### Feature 2: Tendances Temporelles avec Prévisions Simples")
    df_trend = df_filtered.set_index('date')['passage_id'].resample('D').count().reset_index()
    df_trend = df_trend.rename(columns={'passage_id': 'passages'})
    df_trend['moving_avg'] = df_trend['passages'].rolling(window=7).mean()
    x = np.arange(len(df_trend))
    slope, intercept, _, _, _ = linregress(x, df_trend['passages'].fillna(0))
    df_trend['trend'] = intercept + slope * x
    chart_trend = alt.Chart(df_trend.melt(id_vars='date')).mark_line().encode(x='date:T', y='value:Q', color='variable:N')
    st.altair_chart(chart_trend, use_container_width=True)

    st.markdown("### Feature 3: Analyse des Commentaires avec Word Cloud")
    if 'commentaire' in df_filtered.columns and not df_filtered['commentaire'].dropna().empty:
        text = ' '.join(df_filtered['commentaire'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.info("Aucun commentaire disponible pour l'analyse.")

    st.markdown("### Feature 4: Carte Géographique des Lieux Visités")
    lieu_agg = df_filtered.groupby('lieu').agg(passages=('passage_id', 'count')).reset_index()
    coords_map = {
        "Rennes": (48.1147, -1.6794),
        "Nantes": (47.2184, -1.5536),
        "Orleans": (47.9032, 1.9093),
        "Bordeaux": (44.8378, -0.5792),
        "Toulouse": (43.6047, 1.4442),
        "Montpellier": (43.6119, 3.8767),
        "Corbas": (45.6667, 4.9000),
        "Marseille": (43.2965, 5.3698),
        "Clermont": (45.7772, 3.0870),
        "Lille": (50.6292, 3.0573),
        "Nice": (43.7102, 7.2620),
        "Caen": (49.1829, -0.3707),
        "Nancy": (48.6921, 6.1844),
        "Dijon": (47.3220, 5.0415),
        "Tremblay": (48.9500, 2.5667),
        "Lisses": (48.6000, 2.4167),
        "Ferrieres": (48.8167, 2.7167),
        "Pantin": (48.8932, 2.4096),
        "Chartres": (48.4439, 1.4892),
        "Reims": (49.2583, 4.0317),
        "Niort": (46.3231, -0.4588),
        "Strasbourg": (48.5734, 7.7521),
        "Chambery": (45.5646, 5.9178),
        "Rouen": (49.4432, 1.0999),
        "Quincieux": (45.9000, 4.7667),
        "WILLEBROEK": (51.0604, 4.3600),
        "Artenay": (48.0833, 1.8833),
        "Moins": (45.7167, 4.9000),
        "Brebieres": (50.3333, 3.0667),
        "Compans": (48.9833, 2.6667),
        "VICHY - COLIS PRIVE": (46.1266, 3.4208),
        "ST ETIENNE - COLIS PRIVE": (45.4397, 4.3872),
        "ST BRIEUC - COLIS PRIVE": (48.5142, -2.7652),
        "BREST - COLIS PRIVE": (48.3904, -4.4861),
        "VANNES - COLIS PRIVE": (47.6582, -2.7605),
        "LORIENT - COLIS PRIVE": (47.7483, -3.3658),
        "BOURGES - COLIS PRIVE": (47.0810, 2.3986),
        "CHATEAUROUX - COLIS PRIVE": (46.8096, 1.6904),
        "POITIERS - COLIS PRIVE": (46.5802, 0.3404),
        "FREJUS - COLIS PRIVE": (43.4332, 6.7370),
        "RODEZ-COLIS PRIVE": (44.3526, 2.5774),
        "BRIVE-COLIS PRIVE": (45.1596, 1.5331),
        "VALENCE -COLIS PRIVE": (44.9334, 4.8924),
        "ROANNE-COLIS PRIVE": (46.0360, 4.0683),
        "CAHORS-COLIS PRIVE": (44.4491, 1.4366),
        "METZ-COLIS PRIVE": (49.1193, 6.1757),
        "COMPIEGNE-COLIS PRIVE": (49.4179, 2.8261),
        "BEAUVAIS-COLIS PRIVE": (49.4291, 2.0807),
        "ALENCON-COLIS PRIVE": (48.4329, 0.0919),
        "COLMAR-COLIS PRIVE": (48.0794, 7.3585),
        "ANNECY-COLIS PRIVE": (45.8992, 6.1296),
        "MAUREPAS.-COLIS PRIVE": (48.7667, 1.9167),
        "AMIENS-COLIS PRIVE": (49.8941, 2.2957),
        "TOULON-COLIS PRIVE": (43.1242, 5.9280),
        "AVIGNON-COLIS PRIVE": (43.9493, 4.8055),
        "AUXERRE-COLIS PRIVE": (47.7986, 3.5733)
    }
    m = folium.Map(location=[46.6034, 1.8883], zoom_start=5)
    for _, row in lieu_agg.iterrows():
        lieu_name = row['lieu'].split(' - ')[-1] if ' - ' in row['lieu'] else row['lieu']
        if lieu_name in coords_map:
            folium.CircleMarker(
                location=coords_map[lieu_name],
                radius=row['passages'] / 10,
                popup=f"{row['lieu']}: {row['passages']} passages",
                color='blue', fill=True
            ).add_to(m)
    folium_static(m)

    st.markdown("### Feature 5: KPI et Alertes Personnalisables")
    total_passages = len(df_filtered)
    avg_per_agent = df_filtered.groupby('agent')['passage_id'].count().mean()
    if selected_visitetype:
        df_total_by_type = df[df["visitetype"].isin(selected_visitetype)]
    else:
        df_total_by_type = df
    total_lieux = df_total_by_type['lieu'].nunique()
    visited_lieux = df_filtered['lieu'].unique()
    missing_lieux = [lieu for lieu in df_total_by_type['lieu'].unique() if lieu not in visited_lieux]
    coverage_pct = (len(visited_lieux) / total_lieux) * 100 if total_lieux > 0 else 0
    st.metric("Total Passages", total_passages)
    st.metric("Moyenne par Agent", f"{avg_per_agent:.2f}")
    st.metric("Couverture des Lieux (%)", f"{coverage_pct:.1f}%")
    if coverage_pct < 90:
        st.warning("Alerte : Couverture des lieux inférieure à 90% !")
        st.markdown("**Lieux non visités :**")
        missing_df = pd.DataFrame({"lieu_manquant": missing_lieux})
        st.dataframe(missing_df)
        csv_missing = missing_df.to_csv(index=False).encode("utf-8")
        st.download_button("Télécharger la liste des lieux non visités CSV", data=csv_missing, file_name="lieux_non_visites.csv", mime="text/csv")

if __name__ == "__main__":
    main()