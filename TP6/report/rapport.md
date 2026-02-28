# TP6 - CI : IA Explicable et Interprétable

## Exercice 1 : Mise en place, Inférence et Grad-CAM

### Visualisations Grad-CAM

**normal_1.jpeg :**

![Grad-CAM normal_1](../gradcam_normal_1.png)

**normal_2.jpeg :**

![Grad-CAM normal_2](../gradcam_normal_2.png)

**pneumo_1.jpeg :**

![Grad-CAM pneumo_1](../gradcam_pneumo_1.png)

**pneumo_2.jpeg :**

![Grad-CAM pneumo_2](../gradcam_pneumo_2.png)

### Analyse des Faux Positifs et effet Clever Hans

Le modèle ne produit aucun faux positif sur ces 4 images : les deux radiographies saines sont correctement étiquetées NORMAL et les deux pneumonies correctement étiquetées PNEUMONIA. Sur les cas de pneumonie, la carte Grad-CAM devrait mettre en évidence les zones pulmonaires inférieures et centrales, ce qui correspond aux infiltrats et consolidations caractéristiques observés par les radiologues, signe que le modèle regarde les bonnes zones anatomiques.

L'effet **Clever Hans** serait détecté si la zone activée sur une erreur pointait vers un artefact non médical : texte de bordereau, marqueur radio-opaque, position du patient, etc. Pour le provoquer, il faudrait tester le modèle sur une image ne contenant que ces artefacts (ex. une radiographie dont les poumons ont été masqués) et vérifier si le modèle prédit quand même PNEUMONIA.

### Granularité de Grad-CAM

Les cartes générées présentent des blocs flous de basse résolution, même après interpolation bilinéaire à la taille de l'image d'entrée. Cette perte de résolution spatiale est structurelle : dans un ResNet, chaque bloc de convolution applique un *stride* ou un *max pooling* qui réduit progressivement la taille des feature maps. La dernière couche convolutive (stage[-1].layers[-1]) produit des feature maps de taille **7×7** (pour une entrée 224×224), soit un facteur de réduction de ×32. Grad-CAM calcule la moyenne pondérée de ces 49 valeurs par canal, ce qui donne intrinsèquement une carte de chaleur très basse résolution. L'upsampling qui suit ne peut que lisser cette information, il ne crée pas de détails fins.

---
## Exercice 2 : Integrated Gradients et SmoothGrad

### Visualisations IG / SmoothGrad

**normal_1.jpeg :**

![IG SmoothGrad normal_1](../ig_smooth_normal_1.png)

**pneumo_1.jpeg :**

![IG SmoothGrad pneumo_1](../ig_smooth_pneumo_1.png)

### Temps de calcul et faisabilité temps réel

| Méthode | normal_1.jpeg | pneumo_1.jpeg |
|---------|--------------|--------------|
| Inférence seule | 0.0140 s | 0.0194 s |
| Integrated Gradients (50 steps) | 0.7364 s | 0.5435 s |
| SmoothGrad (IG × 100 samples) | 14.1684 s | 14.2219 s |

SmoothGrad est environ **×1000 plus lent** que l'inférence seule (~14 s vs ~0.015 s). Un déploiement synchrone (bloquer l'interface le temps du calcul) est donc difficilement envisageable en production médicale où la réactivité est critique. Une architecture adaptée serait de traiter l'explication de manière **asynchrone via une file de messages**: le médecin reçoit immédiatement le résultat de l'inférence, et l'explication XAI est calculée en tâche de fond et poussée sur l'interface dès qu'elle est prête.

### Avantage mathématique des attributions signées

Grad-CAM applique un filtre ReLU sur ses attributions : il ne conserve que les valeurs positives (pixels qui poussent vers la classe prédite) et ignore les valeurs négatives (pixels qui s'opposent à la prédiction). Cette troncature perd de l'information : un pixel fortement négatif, qui indique que cette zone plaide contre la pneumonie, est tout aussi informatif cliniquement.

Integrated Gradients produit des attributions signées: les valeurs positives signalent les pixels qui ont augmenté le score de la classe prédite, les valeurs négatives ceux qui l'ont diminué. En prenant la valeur absolue (comme fait dans ce script), on visualise l'importance totale de chaque pixel quelle que soit sa direction d'influence. En gardant le signe, on pourrait distinguer les zones pro-pneumonie des zones anti-pneumonie, ce qui offre une carte d'explication bien plus riche et diagnostiquement utile qu'une simple heatmap positive.

---
## Exercice 3 : Modélisation Intrinsèquement Interprétable (Glass-box) sur Données Tabulaires

### Visualisation des coefficients

Accuracy de la régression logistique : **0.9737** (97.4% sur le jeu de test).

![Coefficients Régression Logistique](../glassbox_coefficients.png)

### Feature la plus importante pour la classe "Maligne"

La feature avec le coefficient négatif le plus élevé en valeur absolue est **`worst texture`** (β = -1.35), suivie de `radius error` (β = -1.27) et `worst symmetry` (β = -1.21). Ces trois variables poussent le modèle vers la classe 0 (Maligne) lorsqu'elles sont élevées. D'un point de vue médical, une texture irrégulière et une symétrie cellulaire dégradée sont effectivement des indicateurs histologiques classiques de malignité, le modèle a appris une logique cohérente avec la littérature clinique.

### Avantage du modèle Glass-box vs méthodes post-hoc

Contrairement aux méthodes post-hoc (Grad-CAM, Integrated Gradients) qui génèrent une explication après coup et de manière approximative, un modèle intrinsèquement interprétable comme la régression logistique offre une explication exacte, garantie et disponible instantanément : les coefficients sont l'explication, il n'y a pas de surcoût de calcul ni de risque d'incohérence entre la décision et son explication.

---

## Exercice 4 : Explicabilité Post-Hoc avec SHAP sur un Modèle Complexe

### Visualisations SHAP

**Summary Plot (importance globale) :**

![SHAP Summary Plot](../shap_summary.png)

**Waterfall Plot - Patient 0 (importance locale) :**

![SHAP Waterfall Plot](../shap_waterfall.png)

### Explicabilité Globale : comparaison RF (SHAP) vs Régression Logistique

| Rang | Régression Logistique (coeff. négatifs) | Random Forest (SHAP) |
|------|----------------------------------------|----------------------|
| 1 | `worst texture` (β = -1.35) | `worst area` |
| 2 | `radius error` (β = -1.27) | `worst concave points` |
| 3 | `worst symmetry` (β = -1.21) | `mean concave points` |

Les deux modèles atteignent des performances similaires (~97% et ~96%) mais n'identifient **pas les mêmes biomarqueurs**. La régression logistique, contrainte par sa linéarité, s'appuie sur la texture et la symétrie cellulaire. Le Random Forest, lui, exploite des interactions non-linéaires et fait émerger la taille (`worst area`) et la concavité des cellules comme signaux dominants. Ces deux ensembles de features sont médicalement cohérents (taille tumorale et irrégularité de contour sont toutes deux des indicateurs de malignité), mais aucun n'est universellement "le bon", cela illustre que l'importance d'un biomarqueur dépend du modèle qui l'utilise, d'où la nécessité d'auditer plusieurs modèles avant de valider un pipeline médical.

### Explicabilité Locale : analyse du Patient 0

Pour le patient 0, le modèle prédit **BÉNIN avec une probabilité de 0.97** (contre une moyenne de 0.632 sur le jeu de test). La feature ayant le plus contribué à cette prédiction est **`worst area`** avec une valeur de **677.9** et une contribution SHAP de **+0.07**, la plus grande barre du waterfall. Toutes les contributions sont positives (toutes les barres poussent vers la droite, vers la classe Bénigne), ce qui signifie que l'ensemble des caractéristiques de ce patient concordent dans le sens d'une tumeur bénigne. La valeur `worst area = 677.9` est relativement modérée (les tumeurs malignes ont généralement des aires plus élevées), ce qui explique pourquoi cette feature pousse vers Bénigne.

