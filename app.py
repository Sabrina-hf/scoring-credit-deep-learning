"""
Outil conseiller — Scoring de crédit
=====================================
Interface Gradio avec thème custom (reprend le design du prototype validé) pour
un conseiller bancaire : saisie d'un dossier, probabilité de défaut, décision
selon le seuil optimal, et top 5 des variables influentes (SHAP).

Fichiers attendus dans le même dossier (générés par le notebook d'entraînement) :
    modele_scoring_credit.keras
    preprocessor.joblib
    feature_names.joblib
    seuil_optimal.joblib
    shap_background.joblib
    variables_info.joblib
"""
import os
import numpy as np
import pandas as pd
import joblib
import shap
import gradio as gr
from tensorflow import keras

# ----------------------------------------------------------------------------
# 1. Chargement des artefacts entraînés (voir le notebook Scoring_Credit_DeepLearning)
# ----------------------------------------------------------------------------

MODEL_PATH = "modele_scoring_credit.keras"

modele = keras.models.load_model(MODEL_PATH)
preprocessor = joblib.load("preprocessor.joblib")
feature_names = joblib.load("feature_names.joblib")
seuil_optimal = float(joblib.load("seuil_optimal.joblib"))
background = joblib.load("shap_background.joblib")
variables_info = joblib.load("variables_info.joblib")

# Explainer SHAP — nsamples réduit pour rester réactif dans une démo interactive
explainer = shap.KernelExplainer(
    lambda X: modele.predict(X, verbose=0).ravel(), background
)
SHAP_NSAMPLES = 80


def choices(var):
    """Construit les options (libellé lisible -> code numérique) d'une variable."""
    return [(f"{v}", k) for k, v in variables_info[var]["values"].items()]


# ----------------------------------------------------------------------------
# 2. Thème visuel custom — reprend le look du prototype (flat, cartes, jauge)
# ----------------------------------------------------------------------------

CUSTOM_CSS = """
:root {
    --cs-accent: #185FA5;
    --cs-accent-bg: #E6F1FB;
    --cs-success: #3B6D11;
    --cs-success-bg: #EAF3DE;
    --cs-danger: #991F1F;
    --cs-danger-bg: #FCEBEB;
    --cs-text: #1A1A18;
    --cs-text-secondary: #6B6B63;
    --cs-text-muted: #9A9A90;
    --cs-border: #E4E3DC;
    --cs-surface: #FFFFFF;
    --cs-surface-1: #F7F7F4;
}

.gradio-container { font-family: -apple-system, "Segoe UI", Roboto, sans-serif !important; }

#cs-header {
    display: flex; align-items: center; gap: 12px;
    padding: 4px 0 18px 0;
}
#cs-header .cs-icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: var(--cs-accent-bg); color: var(--cs-accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; flex-shrink: 0;
}
#cs-header .cs-title { font-size: 17px; font-weight: 600; color: var(--cs-text); margin: 0; }
#cs-header .cs-subtitle { font-size: 13px; color: var(--cs-text-secondary); margin: 0; }

.cs-card {
    background: var(--cs-surface) !important;
    border: 1px solid var(--cs-border) !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
}
.cs-card-title { font-size: 14px; font-weight: 600; color: var(--cs-text); margin: 0 0 14px 0; }

#cs-evaluer-btn {
    background: var(--cs-accent) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
#cs-evaluer-btn:hover { opacity: 0.92; }

.cs-result-card {
    border-radius: 10px; padding: 16px 18px; background: var(--cs-surface-1);
    margin-bottom: 12px;
}
.cs-result-label { font-size: 12.5px; color: var(--cs-text-secondary); margin: 0 0 6px 0; }
.cs-proba-value { font-size: 26px; font-weight: 700; color: var(--cs-text); margin: 0 0 10px 0; }

.cs-gauge-track { height: 9px; border-radius: 5px; background: #E7E6E0; overflow: hidden; }
.cs-gauge-fill { height: 100%; border-radius: 5px; }
.cs-gauge-scale { display: flex; justify-content: space-between; margin-top: 5px; }
.cs-gauge-scale span { font-size: 11px; color: var(--cs-text-muted); }

.cs-decision-badge {
    border-radius: 10px; padding: 16px 18px; margin-bottom: 12px;
    display: flex; align-items: center; gap: 10px;
}
.cs-decision-badge .cs-decision-icon { font-size: 20px; }
.cs-decision-text { font-size: 16px; font-weight: 700; margin: 0; }
.cs-decision-sub { font-size: 12.5px; margin: 2px 0 0 0; opacity: 0.85; }

.cs-feature-row { margin-bottom: 10px; }
.cs-feature-row:last-child { margin-bottom: 0; }
.cs-feature-top { display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 4px; }
.cs-feature-name { color: var(--cs-text); }
.cs-feature-effect { font-weight: 600; }
.cs-feature-track { height: 6px; border-radius: 3px; background: #E7E6E0; overflow: hidden; }
.cs-feature-fill { height: 100%; border-radius: 3px; }

.cs-placeholder { color: var(--cs-text-muted); font-size: 14px; text-align: center; padding: 30px 0; }
"""


# ----------------------------------------------------------------------------
# 3. Logique métier : prédiction + construction du HTML de résultat
# ----------------------------------------------------------------------------

def resultats_html_defaut():
    """État initial avant toute évaluation."""
    return """
    <div class="cs-card">
        <p class="cs-placeholder">Renseigne le dossier puis clique sur « Évaluer le dossier »</p>
    </div>
    """


def construire_html_resultats(proba, decision_ok, top5):
    pct = round(proba * 100, 1)
    seuil_pct = round(seuil_optimal * 100, 1)
    couleur_jauge = "var(--cs-danger)" if not decision_ok else "var(--cs-success)"

    if decision_ok:
        badge_bg, badge_color, icon, texte = (
            "var(--cs-success-bg)", "var(--cs-success)", "&#10003;", "Accord"
        )
        sous_texte = "Probabilité de défaut sous le seuil de décision."
    else:
        badge_bg, badge_color, icon, texte = (
            "var(--cs-danger-bg)", "var(--cs-danger)", "&#10005;", "Refus"
        )
        sous_texte = "Risque de défaut jugé trop élevé au regard du seuil."

    features_html = ""
    for f in top5:
        couleur = "var(--cs-danger)" if f["effet"] > 0 else "var(--cs-success)"
        fleche = "&#8593;" if f["effet"] > 0 else "&#8595;"
        largeur = min(100, round(abs(f["effet"]) / top5[0]["abs_max"] * 100, 1)) if top5[0]["abs_max"] else 0
        features_html += f"""
        <div class="cs-feature-row">
            <div class="cs-feature-top">
                <span class="cs-feature-name">{f['label']}</span>
                <span class="cs-feature-effect" style="color:{couleur};">{fleche} {abs(f['effet']):.3f}</span>
            </div>
            <div class="cs-feature-track">
                <div class="cs-feature-fill" style="width:{largeur}%; background:{couleur};"></div>
            </div>
        </div>
        """

    return f"""
    <div class="cs-result-card">
        <p class="cs-result-label">Probabilité de défaut</p>
        <p class="cs-proba-value">{pct}%</p>
        <div class="cs-gauge-track">
            <div class="cs-gauge-fill" style="width:{pct}%; background:{couleur_jauge};"></div>
        </div>
        <div class="cs-gauge-scale">
            <span>0%</span><span>seuil {seuil_pct}%</span><span>100%</span>
        </div>
    </div>

    <div class="cs-decision-badge" style="background:{badge_bg}; color:{badge_color};">
        <span class="cs-decision-icon">{icon}</span>
        <div>
            <p class="cs-decision-text" style="color:{badge_color};">{texte}</p>
            <p class="cs-decision-sub" style="color:{badge_color};">{sous_texte}</p>
        </div>
    </div>

    <div class="cs-result-card" style="margin-bottom:0;">
        <p class="cs-result-label" style="margin-bottom:10px;">Top 5 variables influentes</p>
        {features_html}
    </div>
    """


def predire_dossier(statut_compte, duree_mois, historique_credit, objet_credit, montant_credit, epargne, anciennete_emploi, taux_mensualite,
                     statut_familial, garants, anciennete_residence, patrimoine, age, autres_credits, logement, nb_credits_existants,
                     emploi, personnes_charge, telephone, travailleur_etranger):
    dossier = pd.DataFrame([{
        "statut_compte": statut_compte, "duree_mois": duree_mois, "historique_credit": historique_credit, "objet_credit": objet_credit,
        "montant_credit": montant_credit, "epargne": epargne, "anciennete_emploi": anciennete_emploi, "taux_mensualite": taux_mensualite,
        "statut_familial": statut_familial, "garants": garants, "anciennete_residence": anciennete_residence, "patrimoine": patrimoine,
        "age": age, "autres_credits": autres_credits, "logement": logement, "nb_credits_existants": nb_credits_existants,
        "emploi": emploi, "personnes_charge": personnes_charge, "telephone": telephone, "travailleur_etranger": travailleur_etranger,
    }])

    dossier_prep = preprocessor.transform(dossier)
    proba = float(modele.predict(dossier_prep, verbose=0).ravel()[0])
    decision_ok = proba < seuil_optimal

    shap_row = explainer.shap_values(dossier_prep, nsamples=SHAP_NSAMPLES)[0]
    idx_tries = np.argsort(-np.abs(shap_row))[:5]
    abs_max = float(np.abs(shap_row[idx_tries[0]])) if len(idx_tries) else 1.0

    top5 = []
    for idx in idx_tries:
        nom_technique = feature_names[idx]
        nom_base = nom_technique.split("_")[0]
        label = variables_info.get(nom_base, {}).get("label", nom_technique)
        top5.append({"label": label, "effet": float(shap_row[idx]), "abs_max": abs_max})

    return construire_html_resultats(proba, decision_ok, top5)


# ----------------------------------------------------------------------------
# 4. Construction de l'interface
# ----------------------------------------------------------------------------

with gr.Blocks(css=CUSTOM_CSS, title="Scoring de crédit — Outil conseiller") as demo:

    gr.HTML("""
    <div id="cs-header">
        <div class="cs-icon">&#127974;</div>
        <div>
            <p class="cs-title">Scoring de crédit</p>
            <p class="cs-subtitle">Outil conseiller — pré-décision d'octroi</p>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=6):
            with gr.Group(elem_classes="cs-card"):
                gr.HTML('<p class="cs-card-title">Dossier client</p>')

                with gr.Row():
                    statut_compte = gr.Dropdown(choices("statut_compte"), label=variables_info["statut_compte"]["label"], value=2)
                    historique_credit = gr.Dropdown(choices("historique_credit"), label=variables_info["historique_credit"]["label"], value=2)

                with gr.Row():
                    duree_mois = gr.Slider(4, 72, value=24, step=1, label=variables_info["duree_mois"]["label"])
                    montant_credit = gr.Slider(250, 20000, value=4000, step=50, label=variables_info["montant_credit"]["label"])

                with gr.Row():
                    epargne = gr.Dropdown(choices("epargne"), label=variables_info["epargne"]["label"], value=1)
                    anciennete_emploi = gr.Dropdown(choices("anciennete_emploi"), label=variables_info["anciennete_emploi"]["label"], value=3)

                with gr.Row():
                    taux_mensualite = gr.Dropdown(choices("taux_mensualite"), label=variables_info["taux_mensualite"]["label"], value=3)
                    statut_familial = gr.Dropdown(choices("statut_familial"), label=variables_info["statut_familial"]["label"], value=2)

                with gr.Row():
                    garants = gr.Dropdown(choices("garants"), label=variables_info["garants"]["label"], value=1)
                    anciennete_residence = gr.Dropdown(choices("anciennete_residence"), label=variables_info["anciennete_residence"]["label"], value=2)

                with gr.Row():
                    patrimoine = gr.Dropdown(choices("patrimoine"), label=variables_info["patrimoine"]["label"], value=2)
                    age = gr.Slider(18, 80, value=30, step=1, label=variables_info["age"]["label"])

                with gr.Row():
                    autres_credits = gr.Dropdown(choices("autres_credits"), label=variables_info["autres_credits"]["label"], value=3)
                    logement = gr.Dropdown(choices("logement"), label=variables_info["logement"]["label"], value=2)

                with gr.Row():
                    nb_credits_existants = gr.Dropdown(choices("nb_credits_existants"), label=variables_info["nb_credits_existants"]["label"], value=1)
                    emploi = gr.Dropdown(choices("emploi"), label=variables_info["emploi"]["label"], value=3)

                with gr.Row():
                    personnes_charge = gr.Dropdown(choices("personnes_charge"), label=variables_info["personnes_charge"]["label"], value=2)
                    telephone = gr.Dropdown(choices("telephone"), label=variables_info["telephone"]["label"], value=2)

                with gr.Row():
                    objet_credit = gr.Dropdown(choices("objet_credit"), label=variables_info["objet_credit"]["label"], value=3)
                    travailleur_etranger = gr.Dropdown(choices("travailleur_etranger"), label=variables_info["travailleur_etranger"]["label"], value=2)

                bouton = gr.Button("Évaluer le dossier", elem_id="cs-evaluer-btn")

        with gr.Column(scale=5):
            resultats = gr.HTML(resultats_html_defaut())

    bouton.click(
        fn=predire_dossier,
        inputs=[statut_compte, duree_mois, historique_credit, objet_credit, montant_credit, epargne, anciennete_emploi, taux_mensualite,
                statut_familial, garants, anciennete_residence, patrimoine, age, autres_credits, logement, nb_credits_existants,
                emploi, personnes_charge, telephone, travailleur_etranger],
        outputs=resultats,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

