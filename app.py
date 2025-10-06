import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st
import altair as alt
from datetime import datetime

# Charger les variables d'environnement
load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

@st.cache_data(ttl=300)
def load_data(query="SELECT * FROM db_planning;"):
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
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
    # Créer une colonne pour le jour de la semaine en anglais
    df["weekday"] = df["date"].dt.day_name()
    # Mappage des jours en français
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
    # Définir l'ordre des jours
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

    # Chargement des données
    st.sidebar.header("Filtrer les données")
    df = load_data()

    if df.empty:
        st.warning("Aucune donnée récupérée. Vérifie le nom de la table et la connexion.")
        return

    # Préfiltre : dates de début et de fin séparées
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

    # Vérifier que la date de fin est postérieure ou égale à la date de début
    if end_date < start_date:
        st.sidebar.error("La date de fin doit être postérieure ou égale à la date de début.")
        return

    # Filtrer les données par plage de dates
    df_filtered = df[
        (df["date"] >= pd.Timestamp(start_date)) &
        (df["date"] <= pd.Timestamp(end_date))
    ]

    # Filtres existants + filtre par lieu
    agents_list = sorted(df_filtered["agent"].dropna().unique().tolist())
    visitetype_list = sorted(df_filtered["visitetype"].dropna().unique().tolist())
    lieu_list = sorted(df_filtered["lieu"].dropna().unique().tolist())

    selected_agents = st.sidebar.multiselect("Agent", options=agents_list, default=agents_list)
    selected_visitetype = st.sidebar.multiselect("Type de visite", options=visitetype_list, default=visitetype_list)
    selected_lieux = st.sidebar.multiselect("Lieu", options=lieu_list, default=lieu_list)

    # Appliquer les filtres supplémentaires
    df_filtered = df_filtered[
        df_filtered["agent"].isin(selected_agents) &
        df_filtered["visitetype"].isin(selected_visitetype) &
        df_filtered["lieu"].isin(selected_lieux)
    ]

    # Vérifier si des données restent après filtrage
    if df_filtered.empty:
        st.warning("Aucune donnée ne correspond aux filtres sélectionnés.")
        return

    # Calculs pour le premier graphique (par lieu)
    agg_lieu = compute_passages_by_lieu(df_filtered)

    st.markdown("**Nombre de passages par lieu**")
    st.dataframe(agg_lieu.sort_values("passages", ascending=False).reset_index(drop=True).head(200))

    # Graphique interactif Altair pour passages par lieu
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

    # Option de téléchargement CSV pour agrégats par lieu
    csv_lieu = agg_lieu.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger les agrégats par lieu CSV", data=csv_lieu, file_name="passages_par_lieu.csv", mime="text/csv")

    # Deuxième graphique : fréquence par jour de la semaine
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

if __name__ == "__main__":
    main()