# TP5 - CI : Deep Reinforcement Learning

## Exercice 1 : Comprendre la Matrice et Instrumenter l'Environnement (Exploration de Gymnasium)

### Agent aléatoire - rapport de vol

![Agent aléatoire](../random_agent.gif)

![alt text](img/image-1.png)

**Distance au seuil de résolution :** L'agent aléatoire obtient **-124.90 points** sur ce vol, contre un seuil de résolution fixé à **+200 points** (moyenne sur 100 épisodes). L'écart est donc d'environ **325 points**. L'agent n'a aucune stratégie : il tire des actions uniformément, ce qui déclenche les propulseurs de façon incohérente (34 allumages latéraux pour seulement 13 allumages principaux) et conduit inévitablement au crash en 63 frames.

---
## Exercice 2 : Entraînement et Évaluation de l'Agent PPO (Stable Baselines3)

### Évolution de ep_rew_mean et rapport de vol PPO

L'agent progresse significativement par rapport à l'aléatoire mais ne dépasse pas ~45 de récompense moyenne sur cette fenêtre finale : 500 000 timesteps sont insuffisants pour converger complètement sur LunarLander-v3.

![Agent PPO entraîné](../trained_ppo_agent.gif)

![alt text](img/image-2.png)

**Comparaison agent aléatoire vs PPO :**

| Métrique | Agent aléatoire | Agent PPO |
|----------|----------------|-----------|
| Issue du vol | CRASH | ATTERRISSAGE RÉUSSI |
| Récompense totale | -124.90 pts | +111.95 pts |
| Allumages moteur principal | 13 | 394 |
| Allumages moteurs latéraux | 34 | 453 |
| Durée du vol | 63 frames | 971 frames |

**Analyse :** L'agent PPO réussit l'atterrissage là où l'agent aléatoire crashait, avec un gain de **+237 pts** sur un seul épisode. Il utilise bien plus les propulseurs (394 allumages principaux vs 13) car il contrôle activement sa descente au lieu d'agir au hasard, le vol dure plus longtemps (971 frames vs 63). Cependant, avec une `ep_rew_mean` finale de ~45, l'agent n'a pas encore atteint le seuil de résolution de +200 en moyenne.

---
## Exercice 3 : L'Art du Reward Engineering (Wrappers et Hacking)

### Rapport de vol de l'agent radin et analyse

![Agent PPO Hacked](../hacked_agent.gif)

![alt text](img/image-3.png)

**Description de la stratégie adoptée :**

L'agent radin a appris une politique remarquablement simple : **ne jamais allumer le moteur principal** (0 allumage sur 61 frames). Il se contente de 5 très rares allumages latéraux, laissant le module se laisser tomber librement sous la gravité lunaire. C'est un crash garanti, mais c'est la stratégie que la fonction de récompense modifiée désigne comme optimale.

**Explication mathématique et logique du Reward Hacking :**

L'objectif de l'agent PPO est de maximiser le retour cumulé espéré :

$$G_t = \sum_{k=0}^{T} \gamma^k r_{t+k}$$

Dans l'environnement modifié, la fonction de récompense devient :

$$r'(s, a) = r(s, a) - 50 \cdot \mathbf{1}[a = 2]$$

Comparons les deux stratégies possibles sous cette récompense :

- **Atterrir proprement** (comme le PPO normal) : ~394 allumages principaux × (-50) = **-19 700 pts** de pénalité carburant + récompense d'atterrissage → bilan catastrophique.
- **Se laisser tomber** (0 allumage principal) : pénalité carburant = **0**, coût du crash = -100 pts → bilan = ~**-99 pts**.

L'agent apprend donc que $\arg\max_{a} \mathbb{E}[G_t]$ sous $r'$ correspond à ne jamais utiliser $a=2$, même si cela garantit un crash. La pénalité de -50 par allumage est tellement disproportionnée que **2 allumages principaux coûtent déjà plus cher que le crash lui-même** (-100 pts). La solution est rationnellement optimale vis-à-vis de $r'$ mais totalement aberrante vis-à-vis de l'objectif réel. C'est la définition exacte du **Reward Hacking** : l'agent exploite une faille dans la spécification de la récompense plutôt que d'apprendre le comportement désiré.

---
## Exercice 4 : Robustesse et Changement de Physique (Généralisation OOD)

