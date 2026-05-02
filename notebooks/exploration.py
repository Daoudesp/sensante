"""
SenSante - Exploration du dataset patients_dakar.csv Lab 1 : Git, Python et Structure Projet
"""
import pandas as pd
# ===== CHARGER LES DONNEES =====
df = pd.read_csv("data/patients_dakar.csv")
# ===== PREMIERS APER US =====
print("=" * 50)
print("SENSANTE - Exploration du dataset")
print("=" * 50)
# Dimensions du dataset
print(f"\nNombre de patients : {len(df)}")
print(f"Nombre de colonnes : {df.shape[1]}")
print(f"Colonnes : {list(df.columns)}")
# Apercu des 5 premieres lignes
print(f"\n--- 5 premiers patients ---")
print(df.head())
# ===== STATISTIQUES DE BASE =====
print(f"\n--- Statistiques descriptives ---")
print(df.describe().round(2))
# ===== REPARTITION DES DIAGNOSTICS =====
print(f"\n--- Repartition des diagnostics ---")
diag_counts = df["diagnostic"].value_counts()
for diag, count in diag_counts.items():
    pct = count / len(df) * 100
    print(f" {diag:12s} : {count:3d} patients ({pct:.1f}%)")
# ===== REPARTITION PAR REGION =====
print(f"\n--- Repartition par region (top 5) ---")
region_counts = df["region"].value_counts().head(5)
for region, count in region_counts.items():
    print(f" {region:15s} : {count:3d} patients")
# ===== TEMPERATURE MOYENNE PAR DIAGNOSTIC =====
print(f"\n--- Temperature moyenne par diagnostic ---")
temp_by_diag = df.groupby("diagnostic")["temperature"].mean()
for diag, temp in temp_by_diag.items():
    print(f" {diag:12s} : {temp:.1f} C")

# ===== EXERCICE 1 : Patients par sexe et diagnostic =====
print(f"\n--- Nombre de patients par sexe et diagnostic ---")
patients_by_sexe_diag = df.groupby(["sexe", "diagnostic"]).size()
print(patients_by_sexe_diag)

# ===== EXERCICE 3 : Réflexion sur les données de santé =====
# Le dataset contient 500 patients actifs.
# 3 difficultés potentielles avec de vraies données de santé au Sénégal :
# 1. Qualité des données :
#    - Erreurs de saisie manuelle, données manquantes ou incomplètes
#    - Formatage inconsistant (ex: dates, unités de mesure)
# 2. Vie privée et conformité :
#    - Protection des données sensibles de santé
#    - Conformité RGPD, anonymisation insuffisante
#    - Consentement des patients et accès autorisé
# 3. Représentativité et biais :
#    - Biais géographique (surreprésentation urbaine)
#    - Biais démographique (couches sociales spécifiques)
#    - Populations rurales et isolées sous-représentées

print(f"\n{'=' * 50}")
print("Exploration terminee !")
print("Prochain lab : entrainer un modele ML")
print(f"{'=' * 50}")