import pandas as pd
import numpy as np
import unicodedata
import re
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# Fonction pour supprimer les accents et normaliser les chaînes
def normalize_string(s):
    if pd.isna(s):
        return s
    s = unicodedata.normalize('NFKD', str(s)).encode('ASCII', 'ignore').decode('ASCII')
    s = re.sub(r'[^\w\s-]', '_', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

# Lire le fichier Excel
df1 = pd.read_excel("./data/planning surete.xlsx", sheet_name="Planning", header=2, engine="openpyxl")
df1.rename(columns={"Unnamed: 5" : "Eric VEAUTE 2", "Unnamed: 7":"J-C REPIQUET 2", "Unnamed: 9": "Jonathan CANU 2"},inplace=True)

df2 = pd.read_excel("./data/PLANNING 2026.xlsx", sheet_name="Planning", header=2, engine="openpyxl")
df2.rename(columns={"Unnamed: 5" : "Eric VEAUTE 2", "Unnamed: 7":"J-C REPIQUET 2", "Unnamed: 9": "Jonathan CANU 2", "Unnamed: 11": "Christophe BISSON 2", "Unnamed: 13": "Line VINAY 2" },inplace=True)

df = pd.concat([df1, df2])

# Normaliser les colonnes textuelles dans df
for col in df.columns:
    if df[col].dtype == "object":  # Appliquer uniquement aux colonnes textuelles
        df[col] = df[col].apply(normalize_string)

# 1) Identifier les colonnes 'passage'
people = ["Eric VEAUTE", "J-C REPIQUET", "Jonathan CANU", "Christophe BISSON", "Line VINAY"]
cols_passage = [c for c in df.columns for p in people if c == p or c.startswith(p + " ")]

if not cols_passage:
    raise ValueError("Aucune colonne 'passage' trouvée. Vérifie les noms des colonnes.")

id_cols = [c for c in df.columns if c not in cols_passage]

# 2) Passer en format long (une ligne = un passage)
long = df.melt(
    id_vars=id_cols,
    value_vars=cols_passage,
    var_name="Passage",
    value_name="Valeur"
)

# Garder seulement les passages renseignés (non NaN et non vides)
mask_non_empty = long["Valeur"].notna() & long["Valeur"].astype(str).str.strip().ne("")
long = long[mask_non_empty].copy().reset_index(drop=True)

# 3) Extraire Agent
extra = long["Passage"].str.extract(r"^(.*?)(?:\s+(\d+))?$")
long["Agent"] = extra[0].str.strip()

# Normaliser les noms d'agent
long["Agent"] = long["Agent"].apply(normalize_string)

# --- Anonymiser certains agents: ERIC VEAUTE/ERIC VAUTE -> 1, J-C REPIQUET -> 2, Jonathan CANU -> 3
def _normalize_key_for_map(s):
    if pd.isna(s):
        return s
    key = unicodedata.normalize('NFKD', str(s)).encode('ASCII', 'ignore').decode('ASCII')
    key = re.sub(r"[^\w\s]", "", key)
    key = re.sub(r"\s+", " ", key).strip().upper()
    return key

ANON_MAP = {
    "ERIC VEAUTE": "1",
    "ERIC VAUTE": "1",
    "J C REPIQUET": "2",
    "JC REPIQUET": "2",
    "JONATHAN CANU": "3",
    "CHRISTOPHE BISSON": "4",
    "LINE VINAY": "5"
}

def anonymize_agent_name(s):
    """Retourne le code anonymisé si le nom correspond, sinon renvoie la valeur d'origine."""
    if pd.isna(s):
        return s
    k = _normalize_key_for_map(s)
    return ANON_MAP.get(k, s)

# Appliquer l'anonymisation sur la colonne 'Agent'
long["Agent"] = long["Agent"].apply(anonymize_agent_name)

# 4) Ordonner et trier pour lecture
front = [c for c in ["Date", "Mois", "Jour", "Semaine"] if c in long.columns]
others_id = [c for c in id_cols if c not in front]
order = front + others_id + ["Agent", "Passage", "Valeur"]
long = long[order]

# S'assurer que 'Date' soit bien triée si c'est une date
if "Date" in long.columns:
    try:
        long["Date"] = pd.to_datetime(long["Date"], errors='coerce')
        today = pd.Timestamp.now().normalize()  # Date d'aujourd'hui sans heure
        long = long[long["Date"] <= today].reset_index(drop=True)
    except Exception as e:
        print(f"Erreur lors de la conversion ou du filtrage de 'Date': {e}")
    long = long.sort_values(["Date", "Agent"], kind="stable").reset_index(drop=True)
else:
    long = long.sort_values("Agent", kind="stable").reset_index(drop=True)

# --- 1) Renommer 'Valeur' -> 'Visite'
if "Valeur" in long.columns:
    long = long.rename(columns={"Valeur": "Visite"})

# Normaliser la colonne 'Visite'
long["Visite"] = long["Visite"].apply(normalize_string)

# --- 2) Créer 'VisiteType' selon les règles métier
def classify_visite(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip().upper()
    if s == "SIEGE":
        return "SIEGE"
    if s.startswith("H"):
        return "HUB"
    if s.startswith("CE"):
        return "Agence"
    if s.startswith("AT"):
        return "Antenne"
    if s.startswith("DSP"):
        return "DSP"
    return "Autre"

long["VisiteType"] = long["Visite"].apply(classify_visite)

# --- 3) Créer 'Passage_ID' comme entier pour les visites existantes
long.insert(0, "Passage_ID", range(1, len(long) + 1))

# --- 4) Supprimer la colonne 'Passage'
if "Passage" in long.columns:
    long = long.drop(columns=["Passage"])
    if "Compteur" in long.columns:
        long = long.drop(columns=["Compteur"])

# --- 5) Créer le dictionnaire LIEU_MAP à partir de la feuille "Listes"
df_listes = pd.read_excel("./data/PLANNING 2026.xlsx", sheet_name="Listes", header=1, engine="openpyxl")
LIEU_MAP = dict(zip(df_listes["Code Agence"], df_listes["Nom"]))

# Exclure les entrées non pertinentes
non_lieux = ["SIEGE", "RTT", "CP", "FERIE", "TTRAVAIL", "FORMATION", "RECUP"]
LIEU_MAP = {k: v for k, v in LIEU_MAP.items() if k not in non_lieux}

# --- 6) Normaliser le code (CE/H -> 2 chiffres, upper, sans espaces)
def normalize_code(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip().upper()
    s = re.sub(r"\s+", "", s)
    m = re.match(r"^(CE|H)(\d{1,2})$", s)
    if m:
        pref, num = m.groups()
        return f"{pref}{int(num):02d}"
    return s

# --- 7) Construire "Lieu" = "Code - Nom" avec fallback propre
if "Visite" not in long.columns:
    raise KeyError("La colonne 'Visite' est introuvable dans le DataFrame 'long'.")

codes_norm = long["Visite"].apply(normalize_code)
lieu_nom = codes_norm.map(LIEU_MAP)

mask_has_name = codes_norm.notna() & lieu_nom.notna() & (lieu_nom.astype(str).str.strip() != "")

long["Lieu"] = np.where(
    mask_has_name,
    codes_norm.astype(str) + " - " + lieu_nom.apply(normalize_string).astype(str),
    long["Visite"].astype(str).str.strip()
)

# --- 8) Créer la colonne 'Region' à partir de 'Visite'
REGION_MAP = {
    "CE07": "EST", "AT2601": "EST", "CE09": "EST", "AT0301": "EST", "AT1901": "EST",
    "CE14": "EST", "CE24": "EST", "AT6801": "EST", "CE26": "EST", "AT7401": "EST",
    "CE28": "EST", "AT4201": "EST", "AT4202": "EST",
    "CE10": "NORD", "AT8001": "NORD", "CE13": "NORD", "AT5701": "NORD",
    "CE15": "NORD", "AT6001": "NORD", "AT6002": "NORD", "CE17": "NORD",
    "CE20": "NORD", "CE27": "NORD",
    "CE01": "OUEST", "AT2201": "OUEST", "AT2901": "OUEST", "CE02": "OUEST",
    "AT5601": "OUEST", "AT5602": "OUEST", "CE03": "OUEST", "AT1801": "OUEST",
    "AT3601": "OUEST", "AT8901": "OUEST", "CE12": "OUEST", "AT6101": "OUEST",
    "CE16": "OUEST", "AT7801": "OUEST", "CE18": "OUEST", "CE19": "OUEST",
    "CE21": "OUEST", "AT8601": "OUEST",
    "CE04": "SUD", "CE05": "SUD", "AT1201": "SUD", "AT4601": "SUD",
    "CE06": "SUD", "CE08": "SUD", "AT8301": "SUD", "AT8401": "SUD",
    "CE11": "SUD", "AT8302": "SUD",
    "H10": "HUB", "H03": "HUB", "H07": "HUB", "H18": "HUB", "CE40": "BeLux",
    "DSP": "DSP"
}

long["Region"] = codes_norm.map(REGION_MAP).fillna("Inconnu")

# --- 9) Filtrer pour exclure VisiteType == 'Autre' et les non-lieux
long = long[~long["Visite"].isin(non_lieux)]
long = long[long["VisiteType"] != "Autre"]

# --- 10) Ajouter tous les lieux de LIEU_MAP non présents dans long
all_lieux = []
last_passage_id = long["Passage_ID"].max() if not long.empty else 0
current_passage_id = last_passage_id + 1



for code, nom in LIEU_MAP.items():
    print(code, nom)
    lieu_str = f"{code} - {nom}" if nom.strip() else code
    if lieu_str not in long["Lieu"].values:
        # Créer une ligne pour le lieu non visité
        new_row = {
            "Passage_ID": current_passage_id,
            "Visite": pd.NA,  # Pas de visite pour ce lieu
            "VisiteType": classify_visite(code),
            "Region": REGION_MAP.get(code, "Inconnu"),
            "Lieu": lieu_str,
            "Agent": "Inconnu"  # Agent inconnu pour les lieux non visités
        }
        # Ajouter les colonnes de id_cols (comme Date, Mois, etc.) avec des valeurs par défaut (NaN)
        for col in id_cols:
            new_row[col] = pd.NA
        all_lieux.append(new_row)
        current_passage_id += 1

# Ajouter les lieux non visités au DataFrame
if all_lieux:
    df_all_lieux = pd.DataFrame(all_lieux)
    long = pd.concat([long, df_all_lieux], ignore_index=True)

# --- 11) Mettre les colonnes dans un ordre pratique
front = [c for c in ["Passage_ID", "Date", "Mois", "Jour", "Semaine",
                     "Agent", "Visite", "VisiteType", "Region", "Lieu"]
         if c in long.columns]
others = [c for c in long.columns if c not in front]
long = long[front + others]

# --- 12) Convertir les noms des colonnes en minuscules
long.columns = [col.lower() for col in long.columns]

# --- 13) Exporter vers CSV
long.to_csv("./data/planning.csv", index=False)