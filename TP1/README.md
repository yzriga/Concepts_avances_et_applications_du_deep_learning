# TP1 - Segmentation Interactive avec SAM

## Description

Ce projet implémente une application de segmentation interactive d'images utilisant le modèle SAM (Segment Anything Model) de Meta.

## Structure du projet

```
TP1/
├── data/images/          # Images d'entrée
├── src/                  # Code source
│   ├── app.py           # Application Streamlit principale
│   ├── sam_utils.py     # Utilitaires SAM
│   ├── geom_utils.py    # Utilitaires géométriques
│   └── viz_utils.py     # Utilitaires de visualisation
├── outputs/
│   ├── overlays/        # Images avec masques superposés
│   └── logs/            # Logs d'exécution
├── report/              # Rapport du TP
├── requirements.txt     # Dépendances Python
└── README.md           # Ce fichier
```

## Installation

1. Cloner le dépôt
2. Installer les dépendances : `pip install -r requirements.txt`
3. Lancer l'application : `streamlit run src/app.py`

## Usage

L'application Streamlit permet de :
- Charger des images
- Sélectionner des zones avec des bounding boxes
- Générer des masques de segmentation avec SAM
- Visualiser les résultats
- Calculer des métriques sur les masques

## Auteur

[Votre nom] - TP CI Modern Computer Vision - Janvier 2026
