import pandas as pd
import psycopg2
import toml
from dotenv import load_dotenv
import os

# Fonction pour charger les paramètres de connexion
def load_db_config():
    # Essayer de charger depuis .streamlit/secrets.toml
    secrets_file = os.path.join(".streamlit", "secrets.toml")
    if os.path.exists(secrets_file):
        secrets = toml.load(secrets_file)
        # Supabase fournit les paramètres dans [connections.postgresql]
        if "connections" in secrets and "postgresql" in secrets["connections"]:
            db_config = secrets["connections"]["postgresql"]
            return {
                "DB_USER": db_config.get("username"),
                "DB_PASSWORD": db_config.get("password"),
                "DB_HOST": db_config.get("host"),
                "DB_PORT": db_config.get("port", "5432"),
                "DB_NAME": db_config.get("database", "postgres")
            }
        else:
            print("Aucun bloc [connections.postgresql] trouvé dans secrets.toml, tentative avec .env...")
    
    # Fallback sur .env si secrets.toml n'est pas trouvé ou invalide
    load_dotenv()
    return {
        "DB_USER": os.getenv("DB_USER"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD"),
        "DB_HOST": os.getenv("DB_HOST"),
        "DB_PORT": os.getenv("DB_PORT", "5432"),
        "DB_NAME": os.getenv("DB_NAME", "postgres")
    }

# Charger les paramètres de connexion
db_config = load_db_config()
DB_USER = db_config["DB_USER"]
DB_PASSWORD = db_config["DB_PASSWORD"]
DB_HOST = db_config["DB_HOST"]
DB_PORT = db_config["DB_PORT"]
DB_NAME = db_config["DB_NAME"]

# Vérifier que toutes les variables nécessaires sont définies
if not all([DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME]):
    raise ValueError("Certaines variables de connexion (DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME) sont manquantes.")

# URL de connexion pour psycopg2
DATABASE_URL = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"

# Nom de la table
TABLE_NAME = 'db_planning'

# Lire le fichier CSV
csv_path = 'planning.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas.")

df = pd.read_csv(csv_path)

# Assurer les types de données corrects
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Établir la connexion à Supabase
try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
except Exception as e:
    raise Exception(f"Erreur de connexion à la base de données Supabase : {e}")

# Vérifier si la table existe
cursor.execute("""
    SELECT EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' AND table_name = %s
    );
""", (TABLE_NAME,))
table_exists = cursor.fetchone()[0]

if table_exists:
    # Vérifier les colonnes de la table
    cursor.execute("""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public' AND table_name = %s;
    """, (TABLE_NAME,))
    columns = cursor.fetchall()
    print(f"Colonnes de la table {TABLE_NAME} : {columns}")

    # Vérifier si 'passage_id' existe et est de type integer
    if 'passage_id' not in [col[0] for col in columns]:
        raise ValueError(f"La colonne 'passage_id' n'existe pas dans la table {TABLE_NAME}.")
    passage_id_type = next(col[1] for col in columns if col[0] == 'passage_id')
    if passage_id_type != 'integer':
        print(f"Attention : la colonne 'passage_id' est de type {passage_id_type}, mais le script suppose un type integer.")

# Si la table n'existe pas, la créer avec passage_id comme integer
if not table_exists:
    df.to_sql(TABLE_NAME, f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}", 
              index=False, if_exists='fail', method='multi')
    # Ajouter la contrainte de clé primaire sur passage_id
    cursor.execute(f"""
        ALTER TABLE {TABLE_NAME}
        ADD CONSTRAINT {TABLE_NAME}_pkey PRIMARY KEY (passage_id);
    """)
    conn.commit()
    print(f"Table '{TABLE_NAME}' créée avec passage_id comme clé primaire (type integer) et données insérées.")
else:
    # Créer une table temporaire pour les données du CSV
    temp_table = f'{TABLE_NAME}_temp'
    df.to_sql(temp_table, f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}", 
              index=False, if_exists='replace', method='multi')

    # Supprimer les lignes de db_planning qui ne sont pas dans le CSV
    cursor.execute(f"""
        DELETE FROM {TABLE_NAME}
        WHERE passage_id NOT IN (
            SELECT passage_id::integer FROM {temp_table}
        );
    """)

    # Upsert : insérer ou mettre à jour basé sur passage_id
    cursor.execute(f"""
        INSERT INTO {TABLE_NAME} (
            passage_id, date, mois, jour, semaine, agent, visite, visitetype, lieu, commentaire, region
        )
        SELECT 
            passage_id::integer, date, mois, jour, semaine, agent, visite, visitetype, lieu, commentaire, region
        FROM {temp_table}
        ON CONFLICT (passage_id)
        DO UPDATE SET
            date = EXCLUDED.date,
            mois = EXCLUDED.mois,
            jour = EXCLUDED.jour,
            semaine = EXCLUDED.semaine,
            agent = EXCLUDED.agent,
            visite = EXCLUDED.visite,
            visitetype = EXCLUDED.visitetype,
            lieu = EXCLUDED.lieu,
            commentaire = EXCLUDED.commentaire,
            region = EXCLUDED.region;
    """)
        
    # Supprimer la table temporaire
    cursor.execute(f"DROP TABLE {temp_table};")
    conn.commit()

    print(f"Données mises à jour dans la table '{TABLE_NAME}' : lignes absentes supprimées et nouvelles lignes insérées/mises à jour.")

# Vérification : lire les premières lignes de la table
try:
    df_check = pd.read_sql(f"SELECT * FROM {TABLE_NAME} LIMIT 5;", 
                           f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    print("\nAperçu des 5 premières lignes de la table après mise à jour :")
    print(df_check)
except Exception as e:
    print("Erreur lors de la lecture de la table :", e)

# Fermer la connexion
cursor.close()
conn.close()