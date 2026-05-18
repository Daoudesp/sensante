# notebooks/exercices_lab5.py
# Lab 5 - Exercice 1 : Prompt engineering (Wolof/Français)
#          Exercice 2 : Effet de la température

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ============================================================
# EXERCICE 1 : PROMPT ENGINEERING — Réponse en Wolof/Français
# ============================================================

SYSTEM_PROMPT_WOLOF = """Tu es un assistant médical sénégalais.
Tu reçois un diagnostic et des données patient.
Explique le résultat en mêlant le français et le wolof simple,
comme un médecin sénégalais parlerait à son patient au quotidien.
Utilise des expressions wolof naturelles comme :
- "yërmande" (s'il vous plaît)
- "dafa wér" (ça va aller)
- "dem fetal dokter bi" (allez voir le médecin)
- "dafa daw" (c'est urgent)
- "baal ma" (pardonnez-moi)
- "xam-xam" (savoir/connaissance)
Sois rassurant mais recommande toujours une consultation médicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-même.
Tu expliques uniquement le diagnostic fourni."""

# Cas de test : plusieurs diagnostics
cas_tests = [
    {
        "label": "Cas 1 — Paludisme (Dakar, Femme 28 ans)",
        "sexe": "F", "age": 28, "region": "Dakar",
        "temperature": 39.5,
        "diagnostic": "paludisme", "probabilite": 0.72
    },
    {
        "label": "Cas 2 — Grippe (Ziguinchor, Homme 45 ans)",
        "sexe": "M", "age": 45, "region": "Ziguinchor",
        "temperature": 38.2,
        "diagnostic": "grippe", "probabilite": 0.65
    },
    {
        "label": "Cas 3 — Typhoïde (Saint-Louis, Femme 17 ans)",
        "sexe": "F", "age": 17, "region": "Saint-Louis",
        "temperature": 39.0,
        "diagnostic": "typhoïde", "probabilite": 0.58
    },
    {
        "label": "Cas 4 — Sain (Thiès, Homme 32 ans)",
        "sexe": "M", "age": 32, "region": "Thiès",
        "temperature": 37.0,
        "diagnostic": "sain", "probabilite": 0.91
    },
]

print("=" * 60)
print("EXERCICE 1 : PROMPT WOLOF/FRANÇAIS")
print("=" * 60)

for cas in cas_tests:
    user_prompt = (
        f"Patient : {cas['sexe']}, {cas['age']} ans, région {cas['region']}\n"
        f"Température : {cas['temperature']}°C\n"
        f"Diagnostic du modèle : {cas['diagnostic']} "
        f"(probabilité {cas['probabilite']:.0%})\n"
        f"Explique ce résultat au patient."
    )

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_WOLOF},
            {"role": "user",   "content": user_prompt}
        ],
        max_tokens=200,
        temperature=0.3
    )

    print(f"\n--- {cas['label']} ---")
    print(response.choices[0].message.content)
    print(f"(tokens : {response.usage.total_tokens})")

print("\n")

# ============================================================
# EXERCICE 2 : EFFET DE LA TEMPÉRATURE (0.0, 0.5, 1.0)
# ============================================================

SYSTEM_PROMPT_FR = """Tu es un assistant médical sénégalais.
Tu reçois un diagnostic et des données patient.
Explique le résultat en français simple,
comme un médecin parlerait à son patient.
Sois rassurant mais recommande toujours une consultation médicale.
Maximum 3 phrases.
Ne fais JAMAIS de diagnostic toi-même.
Tu expliques uniquement le diagnostic fourni."""

USER_PROMPT_TEST = (
    "Patient : F, 28 ans, région Dakar\n"
    "Température : 39.5°C\n"
    "Diagnostic du modèle : paludisme (probabilité 72%)\n"
    "Explique ce résultat au patient."
)

temperatures = [0.0, 0.5, 1.0]

print("=" * 60)
print("EXERCICE 2 : EFFET DE LA TEMPÉRATURE")
print("(même prompt, même cas, 3 températures différentes)")
print("=" * 60)

for temp in temperatures:
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_FR},
            {"role": "user",   "content": USER_PROMPT_TEST}
        ],
        max_tokens=200,
        temperature=temp
    )

    print(f"\n--- temperature={temp} ---")
    print(response.choices[0].message.content)
    print(f"(tokens : {response.usage.total_tokens})")

print("\n")

# ============================================================
# ANALYSE COMPARATIVE
# ============================================================
print("=" * 60)
print("ANALYSE : Que change la température ?")
print("=" * 60)
print("""
temperature=0.0 → Réponse la plus déterministe et répétable.
                   Le modèle choisit toujours le token le plus probable.
                   Idéal pour un contexte médical (précision > créativité).

temperature=0.5 → Équilibre entre précision et variété.
                   Légèrement plus fluide et naturel.
                   Bon compromis pour SenSante.

temperature=1.0 → Réponses plus créatives et variées.
                   Risque de formulations imprécises ou inventées.
                   Déconseillé pour un usage médical.

CONCLUSION : Pour SenSante, temperature=0.3 est le bon choix —
proche de 0.0 (fiable) mais avec un peu de fluidité naturelle.
""")