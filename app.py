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
import io

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
    if "region" in df.columns:
        df["region"] = df["region"].astype(str)
    if not df.empty and 'passage_id' in df.columns:
        df = df.drop_duplicates(subset=['passage_id'], keep='first')
    return df



def compute_passages_by_lieu(df):
    # ✅ Comptage DIRECT par lieu (pas de merge complexe)
    agg = (df[df["visite"].notna()]
           .groupby("lieu")["passage_id"]
           .count()
           .reset_index(name="passages"))

    # ✅ Tous les lieux avec 0 si non visités
    all_lieux = df["lieu"].drop_duplicates()
    result = all_lieux.to_frame(name="lieu")
    result = result.merge(agg, on="lieu", how="left")
    result["passages"] = result["passages"].fillna(0).astype(int)

    # ✅ Remplir les autres colonnes pour compatibilité
    result["visitetype"] = "Inconnu"
    result["agent"] = "Inconnu"
    result["region"] = "Inconnu"

    # Remplacer par vraies valeurs pour lieux visités
    visites = df[df["visite"].notna()][["lieu", "visitetype", "agent", "region"]]
    for _, row in visites.drop_duplicates("lieu").iterrows():
        mask = result["lieu"] == row["lieu"]
        result.loc[mask, "visitetype"] = row["visitetype"]
        result.loc[mask, "agent"] = row["agent"]
        result.loc[mask, "region"] = row["region"]
    return result

def compute_passages_by_weekday(df):
    # Exclure les lignes sans date valide (lieux non visités)
    df = df[df["date"].notna()].copy()
    df["weekday"] = df["date"].dt.day_name()
    weekday_map = {
        "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
        "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"
    }
    df["weekday"] = df["weekday"].map(weekday_map)
    days_order = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
    
    agg_weekday = (df.groupby("weekday", dropna=False)
                  .agg(passages=("passage_id", "count"))
                  .reset_index())
    agg_weekday["weekday"] = pd.Categorical(agg_weekday["weekday"], categories=days_order, ordered=True)
    agg_weekday = agg_weekday.sort_values("weekday")
    return agg_weekday

def suggest_visit_days_for_lieux(df, lieux, default_msg="Aucune donnée historique"):
    weekday_map_en_to_fr = {
        "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
        "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"
    }
    days_order = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
    results = {}
    
    for lieu in lieux:
        df_lieu = df[(df["lieu"] == lieu) & (df["date"].notna())]
        if df_lieu.empty or "date" not in df_lieu.columns:
            results[lieu] = default_msg
            continue
        
        counts = (df_lieu.assign(weekday=df_lieu["date"].dt.day_name())
                 .groupby("weekday", dropna=False)
                 .agg(passages=("passage_id", "count"))
                 .reset_index())
        
        full_week_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        counts = counts.set_index("weekday").reindex(full_week_en, fill_value=0).reset_index()
        counts["weekday_fr"] = counts["weekday"].map(weekday_map_en_to_fr)
        counts = counts.dropna(subset=["weekday_fr"]).copy()
        
        if counts.empty:
            results[lieu] = default_msg
            continue
        
        min_val = counts["passages"].min()
        least_days = counts[counts["passages"] == min_val]["weekday_fr"].tolist()
        least_days_sorted = [d for d in days_order if d in least_days]
        results[lieu] = ", ".join(least_days_sorted) if least_days_sorted else default_msg
    return results

def compute_passages_by_agent_weekday(df):
    # On garde seulement les lignes avec une vraie date (visites réelles)
    d = df[df["date"].notna()].copy()
    d["weekday"] = d["date"].dt.day_name()
    weekday_map = {
        "Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
        "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi", "Sunday": "Dimanche"
    }
    d["weekday"] = d["weekday"].map(weekday_map)
    days_order = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
    
    agg = (d.groupby(["agent", "weekday"], dropna=False)
          .agg(passages=("passage_id", "count"))
          .reset_index())
    
    # Assurer toutes les combinaisons agent × jour (pour voir les 0)
    agents = agg["agent"].unique()
    full = (pd.MultiIndex.from_product([agents, days_order], names=["agent", "weekday"])
            .to_frame(index=False)
            .merge(agg, on=["agent", "weekday"], how="left"))
    full["passages"] = full["passages"].fillna(0).astype(int)
    return full, days_order

def main():
    st.set_page_config(page_title="Dashboard Planning Visites", layout="wide")
    st.title("🚀 Dashboard Planning des visites")
    
    st.markdown("---")  
    df = load_data()
    
    
    # Sidebar - Filtres
    st.sidebar.header("🔧 **Filtrer les données**")
    
    # Pré-filtre : supprimer les dimanches pour les visites réelles, conserver les lieux non visités
    if "date" in df.columns:
        df = df[~(df["date"].dt.day_name() == "Sunday") | (df["visite"].isna())].copy()
    
    if df.empty:
        st.error("❌ Aucune donnée récupérée. Vérifiez le nom de la table et la connexion.")
        return
    
    # Sidebar - Plage de dates
    st.sidebar.subheader("📅 Plage de dates")
    min_date_data = df[df["date"].notna()]["date"].min().to_pydatetime().date() if not df["date"].isna().all() else datetime(2025, 1, 1).date()
    max_date_data = df[df["date"].notna()]["date"].max().to_pydatetime().date() if not df["date"].isna().all() else datetime(2026, 12, 31).date()
    
    default_start_date = max(datetime(2026, 1, 1).date(), min_date_data)
    default_end_date = min(datetime(2026, 12, 31).date(), max_date_data)
    
    start_date = st.sidebar.date_input("Date de début", value=default_start_date, min_value=min_date_data, max_value=max_date_data, format="DD/MM/YYYY")
    end_date = st.sidebar.date_input("Date de fin", value=default_end_date, min_value=min_date_data, max_value=max_date_data, format="DD/MM/YYYY")
    
    if end_date < start_date:
        st.sidebar.error("❌ La date de fin doit être postérieure ou égale à la date de début.")
        return
    
    # Filtrer par date, mais conserver les lieux non visités
    df_filtered = df[((df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))) | (df["visite"].isna())]
    
    # Listes pour les filtres
    agents_list = sorted(df["agent"].dropna().unique().tolist())
    visitetype_list = sorted(df["visitetype"].dropna().unique().tolist())
    lieu_list = sorted(df["lieu"].dropna().unique().tolist())
    region_list = sorted(df["region"].dropna().unique().tolist())
    
    # Initialiser session_state pour persistance
    if "selected_agents" not in st.session_state:
        st.session_state.selected_agents = agents_list
    if "selected_visitetype" not in st.session_state:
        st.session_state.selected_visitetype = visitetype_list
    if "selected_lieux" not in st.session_state:
        st.session_state.selected_lieux = lieu_list
    if "selected_regions" not in st.session_state:
        st.session_state.selected_regions = region_list
    
    # Bouton réinitialiser
    if st.sidebar.button("🔄 Réinitialiser les filtres", use_container_width=True):
        st.session_state.selected_agents = agents_list
        st.session_state.selected_visitetype = visitetype_list
        st.session_state.selected_lieux = lieu_list
        st.session_state.selected_regions = region_list
        st.sidebar.success("✅ Filtres réinitialisés !")
    
    # Filtres multiselect avec persistance
    selected_agents = st.sidebar.multiselect("👤 Agent", options=agents_list, 
                                            default=st.session_state.selected_agents, key="agents_multiselect")
    selected_visitetype = st.sidebar.multiselect("📋 Type de visite", options=visitetype_list, 
                                                default=st.session_state.selected_visitetype, key="visitetype_multiselect")
    selected_lieux = st.sidebar.multiselect("📍 Lieu", options=lieu_list, 
                                           default=st.session_state.selected_lieux, key="lieux_multiselect")
    selected_regions = st.sidebar.multiselect("🌍 Région", options=region_list, 
                                             default=st.session_state.selected_regions, key="regions_multiselect")
    
    # Mettre à jour session_state
    st.session_state.selected_agents = selected_agents
    st.session_state.selected_visitetype = selected_visitetype
    st.session_state.selected_lieux = selected_lieux
    st.session_state.selected_regions = selected_regions
    
    # Appliquer les filtres
    df_filtered = df_filtered[df_filtered["agent"].isin(selected_agents) & 
                             df_filtered["visitetype"].isin(selected_visitetype) & 
                             df_filtered["lieu"].isin(selected_lieux) & 
                             df_filtered["region"].isin(selected_regions)]
    
    if df_filtered.empty:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés.")
        return
    
    # 1️⃣ Passages par lieu
    st.markdown("### 📊 Nombre de passages par lieu")
    agg_lieu = compute_passages_by_lieu(df_filtered)
    st.dataframe(agg_lieu.sort_values("passages", ascending=False).reset_index(drop=True).head(200), use_container_width=True)
    
    chart_lieu = (alt.Chart(agg_lieu)
                 .mark_bar()
                 .encode(x=alt.X("passages:Q", title="Nombre de passages"),
                        y=alt.Y("lieu:N", sort='-x', title="Lieu"),
                        color=alt.Color("visitetype:N", title="Type de visite"),
                        tooltip=["lieu", "visitetype", "agent", "region", "passages"])
                 .properties(height=500))
    st.altair_chart(chart_lieu, use_container_width=True)
    
    # 2️⃣ Fréquence par jour de la semaine
    st.markdown("### 📅 Fréquence des passages par jour de la semaine")
    agg_weekday = compute_passages_by_weekday(df_filtered)
    chart_weekday = (alt.Chart(agg_weekday)
                    .mark_bar()
                    .encode(x=alt.X("weekday:N", title="Jour de la semaine", 
                                   sort=["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]),
                           y=alt.Y("passages:Q", title="Nombre de passages"),
                           tooltip=["weekday", "passages"])
                    .properties(height=300))
    st.altair_chart(chart_weekday, use_container_width=True)
    
    # 3️⃣ Heatmap Agent x Jour
    st.markdown("### 🔥 Heatmap de Couverture des Lieux par Agent et Jour")
    agg_agent_weekday, days_order = compute_passages_by_agent_weekday(df_filtered)
    pivot = agg_agent_weekday.pivot_table(index="agent", columns="weekday", values="passages", fill_value=0)
    pivot = pivot.reindex(columns=days_order)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot, annot=True, cmap="YlGnBu", ax=ax, fmt=".0f", cbar_kws={'label': 'Nombre de passages'})
    plt.title("Couverture par Agent et Jour de la Semaine")
    st.pyplot(fig)
    
    # 4️⃣ Tendances temporelles
    st.markdown("### 📈 Tendances Temporelles avec Prévisions")
    df_trend = df_filtered[df_filtered["date"].notna()].set_index('date')['passage_id'].resample('D').count().reset_index()
    df_trend = df_trend.rename(columns={'passage_id': 'passages'})
    df_trend['weekday'] = df_trend['date'].dt.day_name()
    df_trend = df_trend[df_trend['weekday'] != 'Sunday'].reset_index(drop=True)
    df_trend['moving_avg_7D'] = df_trend['passages'].rolling(window=7).mean()
    
    x = np.arange(len(df_trend))
    slope, intercept, _, _, _ = linregress(x, df_trend['passages'].fillna(0))
    df_trend['trend'] = intercept + slope * x
    
    chart_trend = alt.Chart(df_trend.melt(id_vars='date')).mark_line().encode(
        x='date:T', y='value:Q', color='variable:N'
    ).properties(width=900, height=400)
    st.altair_chart(chart_trend, use_container_width=True)
    
    # 5️⃣ Carte géographique
    st.markdown("### 🗺️ Carte Géographique des Lieux Visités")
    lieu_agg = df_filtered[df_filtered["visite"].notna()].groupby('lieu').agg(
        passages=('passage_id', 'count')).reset_index()
    
    coords_map = {
        "Rennes": (48.1147, -1.6794), "Nantes": (47.2184, -1.5536), "Orleans": (47.9032, 1.9093),
        "Bordeaux": (44.8378, -0.5792), "Toulouse": (43.6047, 1.4442), "Montpellier": (43.6119, 3.8767),
        "Corbas": (45.6667, 4.9000), "Marseille": (43.3916, 5.2333), "SIEGE": (43.2965, 5.3698),
        "Clermont": (45.7772, 3.0870), "Lille": (50.6292, 3.0573), "Nice": (43.7102, 7.2620),
        "Caen": (49.1829, -0.3707), "Nancy": (48.6921, 6.1844), "Dijon": (47.3220, 5.0415),
        "Tremblay": (48.9500, 2.5667), "Lisses": (48.6000, 2.4167), "Ferrieres": (48.8167, 2.7167),
        "Pantin": (48.8932, 2.4096), "Chartres": (48.4439, 1.4892), "Reims": (49.2583, 4.0317),
        "Niort": (46.3231, -0.4588), "Strasbourg": (48.5734, 7.7521), "Chambery": (45.5646, 5.9178),
        "Rouen": (49.4432, 1.0999), "Quincieux": (45.9000, 4.7667), "WILLEBROEK": (51.0604, 4.3600),
        "Artenay": (48.0833, 1.8833), "Moins": (45.7167, 4.9000), "Brebieres": (50.3333, 3.0667),
        "Compans": (48.9833, 2.6667), "VICHY - COLIS PRIVE": (46.1266, 3.4208),
        "ST ETIENNE - COLIS PRIVE": (45.4397, 4.3872), "ST BRIEUC - COLIS PRIVE": (48.5142, -2.7652),
        "BREST - COLIS PRIVE": (48.3904, -4.4861), "VANNES - COLIS PRIVE": (47.6582, -2.7605),
        "LORIENT - COLIS PRIVE": (47.7483, -3.3658), "BOURGES - COLIS PRIVE": (47.0810, 2.3986),
        "CHATEAUROUX - COLIS PRIVE": (46.8096, 1.6904), "POITIERS - COLIS PRIVE": (46.5802, 0.3404),
        "FREJUS - COLIS PRIVE": (43.4332, 6.7370), "RODEZ-COLIS PRIVE": (44.3526, 2.5774),
        "BRIVE-COLIS PRIVE": (45.1596, 1.5331), "VALENCE -COLIS PRIVE": (44.9334, 4.8924),
        "ROANNE-COLIS PRIVE": (46.0360, 4.0683), "CAHORS-COLIS PRIVE": (44.4491, 1.4366),
        "METZ-COLIS PRIVE": (49.1193, 6.1757), "COMPIEGNE-COLIS PRIVE": (49.4179, 2.8261),
        "BEAUVAIS-COLIS PRIVE": (49.4291, 2.0807), "ALENCON-COLIS PRIVE": (48.4329, 0.0919),
        "COLMAR-COLIS PRIVE": (48.0794, 7.3585), "ANNECY-COLIS PRIVE": (45.8992, 6.1296),
        "MAUREPAS.-COLIS PRIVE": (48.7667, 1.9167), "AMIENS-COLIS PRIVE": (49.8941, 2.2957),
        "TOULON-COLIS PRIVE": (43.1242, 5.9280), "AVIGNON-COLIS PRIVE": (43.9493, 4.8055),
        "AUXERRE-COLIS PRIVE": (47.7986, 3.5733)
    }
    
    m = folium.Map(location=[46.6034, 1.8883], zoom_start=5)
    for _, row in lieu_agg.iterrows():
        lieu_name = row['lieu'].split(' - ')[-1] if ' - ' in row['lieu'] else row['lieu']
        if lieu_name in coords_map:
            folium.CircleMarker(
                location=coords_map[lieu_name],
                radius=max(5, row['passages'] / 5),
                popup=f"{row['lieu']}: {row['passages']} passages",
                color='blue', fill=True, fillOpacity=0.7
            ).add_to(m)
    folium_static(m, width=900, height=500)
    
    # 6️⃣ KPI et Alertes
    st.markdown("### 🎯 KPI et Alertes")
    total_passages = len(df_filtered[df_filtered["visite"].notna()])
    avg_per_agent = df_filtered[df_filtered["visite"].notna()].groupby('agent')['passage_id'].count().mean() if not df_filtered[df_filtered["visite"].notna()].empty else 0
    total_lieux = df['lieu'].nunique()
    visited_lieux = df_filtered[df_filtered["visite"].notna()]['lieu'].unique()
    missing_lieux = [lieu for lieu in df['lieu'].unique() if lieu not in visited_lieux]
    coverage_pct = (len(visited_lieux) / total_lieux) * 100 if total_lieux > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Passages", total_passages)
    col2.metric("Moyenne par Agent", f"{avg_per_agent:.1f}")
    col3.metric("Couverture des Lieux", f"{coverage_pct:.1f}%")
    
    # Lieux manquants
    show_all_missing = st.checkbox("👀 Afficher toutes les agences manquantes", value=True)
    if coverage_pct < 90 or show_all_missing:
        st.markdown("**🚨 Lieux non visités :**")
        missing_df = pd.DataFrame({"lieu_manquant": missing_lieux})
        suggestions = suggest_visit_days_for_lieux(df, missing_lieux)
        missing_df["jour_de_visite_suggere"] = missing_df["lieu_manquant"].map(suggestions)
        
        jours_fr = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi"]
        for jour in jours_fr:
            col_name = f"nb_passage_{jour.lower()}"
            missing_df[col_name] = 0
        
        if "date" in df.columns:
            df_counts = (df[df["date"].notna()]
                        .assign(weekday=df["date"].dt.day_name())
                        .groupby(["lieu", "weekday"], dropna=False)
                        .agg(passages=("passage_id", "count"))
                        .reset_index())
            en_to_fr = {"Monday": "Lundi", "Tuesday": "Mardi", "Wednesday": "Mercredi",
                       "Thursday": "Jeudi", "Friday": "Vendredi", "Saturday": "Samedi"}
            df_counts["weekday_fr"] = df_counts["weekday"].map(en_to_fr)
            
            for idx, row in missing_df.iterrows():
                lieu = row["lieu_manquant"]
                subset = df_counts[df_counts["lieu"] == lieu]
                for _, r in subset.iterrows():
                    jour_fr = r["weekday_fr"]
                    if pd.isna(jour_fr): continue
                    col = f"nb_passage_{jour_fr.lower()}"
                    if col in missing_df.columns:
                        missing_df.at[idx, col] = int(r["passages"])
        
        st.dataframe(missing_df)
        
        # Download Excel
        try:
            towrite = io.BytesIO()
            with pd.ExcelWriter(towrite, engine="openpyxl") as writer:
                missing_df.to_excel(writer, index=False, sheet_name="lieux_non_visites")
            towrite.seek(0)
            st.download_button("📥 Télécharger XLSX", data=towrite.getvalue(), 
                             file_name="lieux_non_visites.xlsx", 
                             mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except Exception as e:
            st.error(f"Erreur Excel : {e}")

if __name__ == "__main__":
    main()
