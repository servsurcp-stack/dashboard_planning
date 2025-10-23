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
df = pd.read_excel("planning surete.xlsx", sheet_name="Planning", header=2, engine="openpyxl")

# Renommer les colonnes spécifiques
df.rename(columns={"Unnamed: 5": "Eric VEAUTE 2", "Unnamed: 7": "J-C REPIQUET 2", "Unnamed: 9": "Jonathan CANU 2"}, inplace=True)

# Normaliser les colonnes textuelles dans df
for col in df.columns:
    if df[col].dtype == "object":  # Appliquer uniquement aux colonnes textuelles
        df[col] = df[col].apply(normalize_string)

# 1) Identifier les colonnes 'passage'
people = ["Eric VEAUTE", "J-C REPIQUET", "Jonathan CANU"]
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
# Normalisation utilisée pour faire correspondre les clés (supprime accents et ponctuation, met en MAJ)
def _normalize_key_for_map(s):
    if pd.isna(s):
        return s
    key = unicodedata.normalize('NFKD', str(s)).encode('ASCII', 'ignore').decode('ASCII')
    # Supprimer la ponctuation (y compris les tirets) pour matcher des variantes comme 'J-C' vs 'J C'
    key = re.sub(r"[^\w\s]", "", key)
    key = re.sub(r"\s+", " ", key).strip().upper()
    return key

ANON_MAP = {
    "ERIC VEAUTE": "1",
    "ERIC VAUTE": "1",
    "J C REPIQUET": "2",
    "JC REPIQUET": "2",
    "JONATHAN CANU": "3",
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
        # Filtrer pour ne garder que les lignes où Date <= aujourd'hui
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
    return "Autre"

long["VisiteType"] = long["Visite"].apply(classify_visite)

# --- 3) Créer 'Passage_ID' comme entier
long.insert(0, "Passage_ID", range(1, len(long) + 1))

# --- 4) Supprimer la colonne 'Passage'
if "Passage" in long.columns:
    long = long.drop(columns=["Passage"])
    if "Compteur" in long.columns:
        long = long.drop(columns=["Compteur"])

# --- 5) Dictionnaire Code -> Nom (ville/libellé)
LIEU_MAP = {
    "CE01": "Rennes", "CE02": "Nantes", "CE03": "Orleans", "CE04": "Bordeaux",
    "CE05": "Toulouse", "CE06": "Montpellier", "CE07": "Corbas", "CE08": "Marseille",
    "CE09": "Clermont", "CE10": "Lille", "CE11": "Nice", "CE12": "Caen",
    "CE13": "Nancy", "CE14": "Dijon", "CE15": "Tremblay", "CE16": "Lisses",
    "CE17": "Ferrieres", "CE18": "Pantin", "CE19": "Chartres", "CE20": "Reims",
    "CE21": "Niort", "CE24": "Strasbourg", "CE26": "Chambery", "CE27": "Rouen",
    "CE28": "Quincieux", "CE40": "WILLEBROEK",
    "H03": "Artenay", "H07": "Moins", "H10": "Brebieres", "H18": "Compans",
    "SIEGE": "", "RTT": "", "CP": "", "FERIE": "", "DSP": "", "TTRAVAIL": "", "FORMATION": "",
    "AT0301": "VICHY - COLIS PRIVE", "AT4201": "ST ETIENNE - COLIS PRIVE",
    "AT2201": "ST BRIEUC - COLIS PRIVE", "AT2901": "BREST - COLIS PRIVE",
    "AT5601": "VANNES - COLIS PRIVE", "AT5602": "LORIENT - COLIS PRIVE",
    "AT1801": "BOURGES - COLIS PRIVE", "AT3601": "CHATEAUROUX - COLIS PRIVE",
    "AT8601": "POITIERS - COLIS PRIVE", "AT8302": "FREJUS - COLIS PRIVE",
    "AT1201": "RODEZ-COLIS PRIVE", "AT1901": "BRIVE-COLIS PRIVE",
    "AT2601": "VALENCE -COLIS PRIVE", "AT4202": "ROANNE-COLIS PRIVE",
    "AT4601": "CAHORS-COLIS PRIVE", "AT5701": "METZ-COLIS PRIVE",
    "AT6001": "COMPIEGNE-COLIS PRIVE", "AT6002": "BEAUVAIS-COLIS PRIVE",
    "AT6101": "ALENCON-COLIS PRIVE", "AT6801": "COLMAR-COLIS PRIVE",
    "AT7401": "ANNECY-COLIS PRIVE", "AT7801": "MAUREPAS.-COLIS PRIVE",
    "AT8001": "AMIENS-COLIS PRIVE", "AT8301": "TOULON-COLIS PRIVE",
    "AT8401": "AVIGNON-COLIS PRIVE", "AT8901": "AUXERRE-COLIS PRIVE",
}

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
    "SIEGE": "Siege", "RTT": "Inconnu", "CP": "Inconnu", "FERIE": "Inconnu",
    "DSP": "Inconnu", "TTRAVAIL": "Inconnu", "FORMATION": "Inconnu", "H10": "NORD", "H03": "OUEST", "H07": "EST", "H18": "National", "CE40": "BeLux"
}

long["Region"] = codes_norm.map(REGION_MAP).fillna("Inconnu")

# --- 9) Filtrer pour exclure VisiteType == 'Autre'
long = long[long["VisiteType"] != "Autre"]

# --- 10) Mettre les colonnes dans un ordre pratique
front = [c for c in ["Passage_ID", "Date", "Mois", "Jour", "Semaine",
                     "Agent", "Visite", "VisiteType", "Region", "Lieu"]
         if c in long.columns]
others = [c for c in long.columns if c not in front]
long = long[front + others]

# --- 11) Convertir les noms des colonnes en minuscules
long.columns = [col.lower() for col in long.columns]

# --- 12) Exporter vers CSV
long.to_csv("planning.csv", index=False)