# TP2 - CI : Génération d'image

## Exercice 1: Mise en place & smoke test (GPU + Diffusers)

**Image générée :**
![alt text](img/image-1.png)

- Aucun problème OOM ou CUDA rencontré

---
## Exercice 2: Factoriser le chargement du pipeline (text2img/img2img)

**Image générée :**
![alt text](img/image-2.png)

**Configuration baseline :**
![alt text](img/image-3.png)

---
## Exercice 3: Text2Img : 6 expériences contrôlées (paramètres steps, guidance, scheduler)

**Grille de résultats :**

| Run01 - Baseline (EulerA, 30 steps, guidance 7.5) | Run02 - Steps15 (15 steps) |
|---|---|
| ![Run01](../outputs/t2i_run01_baseline.png) | ![Run02](../outputs/t2i_run02_steps15.png) |

| Run03 - Steps50 (50 steps) | Run04 - Guidance4 (guidance 4.0) |
|---|---|
| ![Run03](../outputs/t2i_run03_steps50.png) | ![Run04](../outputs/t2i_run04_guid4.png) |

| Run05 - Guidance12 (guidance 12.0) | Run06 - DDIM (scheduler DDIM) |
|---|---|
| ![Run05](../outputs/t2i_run05_guid12.png) | ![Run06](../outputs/t2i_run06_ddim.png) |

**Observations qualitatives :**

**Effet des steps (15 vs 30 vs 50) :**
- **15 steps :** Image plus granuleuse, détails moins affinés, convergence incomplète
- **30 steps :** Bon compromis qualité/temps, détails nets, textures réalistes  
- **50 steps :** Raffinement marginal, sur-optimisation possible, temps 2x plus long

**Effet du guidance (4.0 vs 7.5 vs 12.0) :**
- **Guidance 4.0 :** Interprétation plus libre du prompt, créativité accrue, moins de contraste
- **Guidance 7.5 :** Équilibre optimal fidélité/naturalisme, respect du prompt
- **Guidance 12.0 :** Sur-saturation des couleurs, artefacts possibles, rigidité excessive

**Effet du scheduler (EulerA vs DDIM) :**
- **EulerA :** Rendu plus naturel, transitions fluides, meilleur pour produits e-commerce
- **DDIM :** Style légèrement différent, peut produire des variations intéressantes
- **Performance :** Vitesse similaire (~7.8 it/s) pour 30 steps