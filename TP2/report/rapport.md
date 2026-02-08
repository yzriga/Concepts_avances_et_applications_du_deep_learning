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

---
## Exercice 4: Img2Img : 3 expériences contrôlées (strength faible/moyen/élevé)

**Image source vs transformations :**

| Image Source (baseline handbag) | Strength 0.35 (faible) |
|---|---|
| ![Source](../inputs/source_handbag.png) | ![Run07](../outputs/i2i_run07_strength035.png) |

| Strength 0.60 (moyen) | Strength 0.85 (élevé) |
|---|---|
| ![Run08](../outputs/i2i_run08_strength060.png) | ![Run09](../outputs/i2i_run09_strength085.png) |

**Analyse :**

**Ce qui est conservé :**
- **Strength 0.35 :** Structure générale handbag, forme principale, position globale, cadrage identique
- **Strength 0.60 :** Silhouette reconnaissable, concept produit préservé, certaines proportions
- **Strength 0.85 :** Notion de "sac" reste, mais forme et détails très altérés

**Ce qui change :**
- **Strength 0.35 :** Textures affinées, éclairage légèrement modifié, détails surface handbag
- **Strength 0.60 :** Matériaux transformés (cuir -> autres textures), arrière-plan modifié, style général
- **Strength 0.85 :** Transformation majeure : couleur, matière, forme, éclairage complètement repensés

**Utilisabilité e-commerce :**
- **Strength 0.35 :** **Optimal** - Améliore qualité sans dénaturer, garde identité produit
- **Strength 0.60 :** **Modéré** - Variations intéressantes mais peut s'éloigner trop du produit original
- **Strength 0.85 :** **Risqué** - Transformation trop importante, produit non reconnaissable

---
