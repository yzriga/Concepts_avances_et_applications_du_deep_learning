# TP3 - CI : Deep learning pour audio

## Exercice 1: Initialisation du TP3 et vérification de l'environnement

### Résultats obtenus
![alt text](img/image-1.png)

---
## Exercice 2: Constituer un mini-jeu de données : enregistrement d’un “appel” (anglais) + vérification audio

### Vérification des métadonnées

![alt text](img/image-2.png)

### Résultats d'inspect_audio.py

![alt text](img/image-3.png)

---
## Exercice 3: VAD (Voice Activity Detection) : segmenter la parole et mesurer speech/silence

### Commande utilisée

```bash
pip install silero-vad
```

### Exécution du VAD

![alt text](img/image-4.png)

### Analyse speech/silence

Le ratio de parole de **0.847** est cohérent avec une lecture continue à voix claire. Les 10 segments reflètent bien la structure naturelle du texte : les grands blocs (ex. 0.51->7.45s, 11.14->18.40s, 23.68->30.43s) correspondent aux phrases longues, tandis que les 3 courts segments en fin d'enregistrement (≈0.60s chacun) correspondent aux chiffres du numéro de téléphone « 5 5 5 0 1 9 9 » prononcés séparément. Le VAD est bien calibré : pas de micro-segments parasites, les pauses naturelles entre phrases sont correctement détectées comme silence.

### Ajustement du seuil min_dur_s

Pour un filtrage plus strict :

```python
min_dur_s = 0.60
```

En passant de 0.30 à 0.60, `num_segments` reste à 10 et `speech_ratio` est inchangé : tous les segments détectés ont déjà une durée ≥ 0.604s. Le filtrage à 0.30s était déjà suffisamment sélectif sur cet enregistrement.

---
## Exercice 4: ASR avec Whisper : transcription segmentée + mesure de latence

### Exécution — model_id, elapsed_s, rtf

![alt text](img/image-5.png)

### Extrait JSON — segments et full_text

5 premiers segments :

```json
{ "segment_id": 0, "start_s": 0.514,  "end_s": 7.454,  "text": "Hello, thank you for calling customer support. My name is Alex and I will help you today." },
{ "segment_id": 1, "start_s": 8.034,  "end_s": 10.654, "text": "I'm calling about an order that arrived damaged." },
{ "segment_id": 2, "start_s": 11.138, "end_s": 18.398, "text": "The package was delivered yesterday, but the screen is cracked. I would like refund or replacement as soon as possible." },
{ "segment_id": 3, "start_s": 18.69,  "end_s": 20.35,  "text": "The order number is A." },
{ "segment_id": 4, "start_s": 20.514, "end_s": 21.758, "text": "X19." }
```

full_text :

![alt text](img/image-6.png)

### Analyse VAD + transcription

La segmentation VAD aide globalement la transcription : les segments longs (0–3) sont très bien transcrits, avec ponctuation et majuscules correctes. En revanche, elle nuit sur les éléments épelés : le numéro de commande "AX19735" a été découpé en trois segments courts (3, 4, 5 -> "A.", "X19.", "75"), empêchant Whisper de reconstituer le token complet. Même constat pour le numéro de téléphone (segments 7–8 -> "0-1.", "by nine." au lieu de "0199"). Le segment 6 montre aussi une erreur sur l'email : "john.mit and example.com" au lieu de "john dot smith at example dot com".

---
## Exercice 5: Call center analytics : redaction PII + intention + fiche appel