import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import os

# Charger les variables d'environnement
load_dotenv()

# Paramètres de connexion à la base de données
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# URL de connexion
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

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

# Vérifier si la table existe
with engine.connect() as conn:
    table_exists = conn.execute(text(f"""
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name = '{TABLE_NAME}'
        );
    """)).scalar()

    if table_exists:
        # Vérifier les colonnes de la table
        columns_query = conn.execute(text(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'public' AND table_name = '{TABLE_NAME}';
        """)).fetchall()
        columns = [(row[0], row[1]) for row in columns_query]
        print(f"Colonnes de la table {TABLE_NAME} : {columns}")

        # Vérifier si 'passage_id' existe et est de type integer
        if 'passage_id' not in [col[0] for col in columns]:
            raise ValueError(f"La colonne 'passage_id' n'existe pas dans la table {TABLE_NAME}.")
        passage_id_type = next(col[1] for col in columns if col[0] == 'passage_id')
        if passage_id_type != 'integer':
            print(f"Attention : la colonne 'passage_id' est de type {passage_id_type}, mais le script suppose un type integer.")

# Si la table n'existe pas, la créer avec passage_id comme integer
if not table_exists:
    df.to_sql(TABLE_NAME, engine, index=False, if_exists='fail', dtype={'passage_id': 'integer'})
    # Ajouter la contrainte de clé primaire sur passage_id
    with engine.connect() as conn:
        conn.execute(text(f"""
            ALTER TABLE {TABLE_NAME}
            ADD CONSTRAINT {TABLE_NAME}_pkey PRIMARY KEY (passage_id);
        """))
        conn.commit()
    print(f"Table '{TABLE_NAME}' créée avec passage_id comme clé primaire (type integer) et données insérées.")
else:
    # Créer une table temporaire pour les données du CSV
    temp_table = f'{TABLE_NAME}_temp'
    df.to_sql(temp_table, engine, index=False, if_exists='replace')

    with engine.connect() as conn:
        # Supprimer les lignes de db_planning qui ne sont pas dans le CSV
        conn.execute(text(f"""
            DELETE FROM {TABLE_NAME}
            WHERE passage_id NOT IN (
                SELECT passage_id::integer FROM {temp_table}
            );
        """))

        # Upsert : insérer ou mettre à jour basé sur passage_id
        conn.execute(text(f"""
            INSERT INTO {TABLE_NAME} (
                passage_id, date, mois, jour, semaine, agent, visite, visitetype, lieu, commentaire
            )
            SELECT 
                passage_id::integer, date, mois, jour, semaine, agent, visite, visitetype, lieu, NULL AS commentaire
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
                commentaire = EXCLUDED.commentaire;
        """))
        conn.execute(text(f"DROP TABLE {temp_table};"))
        conn.commit()

    print(f"Données mises à jour dans la table '{TABLE_NAME}' : lignes absentes supprimées et nouvelles lignes insérées/mises à jour.")

# Vérification : lire les premières lignes de la table
try:
    df_check = pd.read_sql(f"SELECT * FROM {TABLE_NAME} LIMIT 5;", engine)
    print("\nAperçu des 5 premières lignes de la table après mise à jour :")
    print(df_check)
except Exception as e:
    print("Erreur lors de la lecture de la table :", e)