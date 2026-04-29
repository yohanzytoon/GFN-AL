# SCRIPT DE PRÉSENTATION
## GFlowNets et apprentissage actif pour la découverte sous budget
### Durée : ~10 minutes | 3 présentateurs
### IFT 3710 / IFT 6759 — Hiver 2026

---

> **Instructions :** Lisez à voix haute à un rythme naturel de présentation (~120–130 mots/min).
> Avancez la diapositive au moment indiqué entre crochets.
> Les transitions entre présentateurs sont indiquées clairement.

---

---
## YOHAN — Slides 1 à 5 [~3:10]
---

**[Avancer : Slide 1 — Titre]**

Bonjour à tous. Notre projet s'appelle « GFlowNets et apprentissage actif pour la découverte sous budget ».
Je m'appelle Yohan, et je vais commencer par vous présenter le problème qu'on a cherché à résoudre.
Ensuite, Charbel vous expliquera les méthodes qu'on a utilisées, et Yann-Olivier présentera nos résultats et nos conclusions.

---

**[Avancer : Slide 2 — Le problème en une phrase]**

La question centrale de notre projet est simple : comment découvrir rapidement de bons mots valides dans un espace très grand, quand chaque évaluation exacte est coûteuse ?

Pour tester nos méthodes, nous avons construit un environnement inspiré du jeu de Scrabble.
Le principe est le suivant : un candidat est une séquence de lettres, avec une longueur maximale de sept.
Un oracle — imaginez un dictionnaire avec une calculatrice intégrée — vérifie si la séquence est un vrai mot anglais.
Si oui, il retourne le score Scrabble du mot.
Sinon, le score est zéro.
Et chaque méthode qu'on compare dispose exactement du même budget de requêtes à cet oracle.

---

**[Avancer : Slide 3 — Pourquoi ce problème est intéressant]**

Vous allez peut-être vous demander : pourquoi s'intéresser au Scrabble ?
Eh bien, ce cadre est en réalité une version jouet d'un problème très sérieux en recherche scientifique.
Pensez à la découverte de nouveaux médicaments, de matériaux innovants, ou de séquences protéiques :
dans tous ces cas, l'espace de recherche est énorme et combinatoire, la majorité des candidats sont mauvais ou invalides,
et surtout, les évaluations exactes — expériences de laboratoire, simulations lourdes — sont rares et coûteuses.
On ne peut pas tout tester.

Le vrai défi n'est donc pas seulement de trouver un bon candidat.
C'est de trouver de bons candidats **en dépensant le moins d'évaluations possible**.

---

**[Avancer : Slide 4 — Définition de la tâche]**

Techniquement, chaque état du problème est un mot de longueur maximale sept.
La récompense vaut le score Scrabble si le mot est un vrai mot anglais, et zéro sinon.
Dans toutes nos expériences, les méthodes comparées ont exactement le même budget oracle.
C'est ce qui rend la comparaison équitable : chacun dispose du même nombre de questions à poser à l'oracle.

---

**[Avancer : Slide 5 — Pourquoi c'est difficile]**

Alors, où se trouve la difficulté concrète ? Ce n'est pas tant la taille de l'espace de recherche.
C'est ce qu'on appelle la **parcimonie de la récompense** — en anglais, *sparse reward*.

La grande majorité des combinaisons aléatoires de lettres ne forment pas de vrais mots.
Donc la plupart des requêtes oracle retournent zéro.
Et quand un modèle reçoit presque uniquement des zéros comme signal d'apprentissage, il apprend très difficilement où chercher.
Cela vaut pour un prédicteur classique, pour un générateur, et même pour une exploration naïve.

C'est ce problème précis — comment apprendre efficacement avec un signal très rare — qui est au cœur de notre projet.

Je laisse maintenant la parole à Charbel.

---

---
## CHARBEL — Slides 6 à 9 [~3:10]
---

**[Avancer : Slide 6 — Qu'est-ce que l'apprentissage actif ?]**

Merci Yohan. Pour répondre à ce défi, on a utilisé deux outils principaux. Commençons par le premier : l'apprentissage actif.

L'idée de base est la suivante : au lieu d'évaluer des candidats au hasard, on choisit intelligemment lesquels évaluer.
La boucle se déroule en six étapes.

On part d'un petit ensemble de mots déjà évalués par l'oracle.
On entraîne ce qu'on appelle un **modèle substitut** — un modèle léger qui essaie de prédire le score d'un mot sans avoir à appeler l'oracle.
On génère un grand pool de candidats potentiels.
Une **fonction d'acquisition** choisit dans ce pool les candidats les plus utiles à évaluer — ceux qui sont à la fois prometteurs et informatifs.
On envoie uniquement ce sous-ensemble à l'oracle réel.
Et on réentraîne le substitut avec les nouvelles données, puis on recommence.

L'intuition clé : au lieu de gaspiller notre budget sur des candidats peu intéressants, on guide chaque requête oracle vers ce qui nous apprendra le plus.

---

**[Avancer : Slide 7 — Qu'est-ce qu'un GFlowNet ?]**

Le deuxième outil est le GFlowNet, ou Generative Flow Network. C'est un type de modèle génératif, c'est-à-dire un modèle dont le rôle est de **générer** des objets. Sa propriété fondamentale : il apprend à générer des objets **proportionnellement à leur récompense**.

Concrètement dans notre contexte : le GFlowNet construit un mot lettre par lettre, en suivant une politique qu'il a apprise.
À convergence, les mots à forte récompense sont générés plus souvent que les mauvais mots.
Mais contrairement à un algorithme d'optimisation classique qui converge vers un seul optimum,
le GFlowNet maintient une certaine **diversité** dans ce qu'il produit.

Et ça, c'est exactement ce qu'on veut.
On ne cherche pas le meilleur mot unique.
On cherche un ensemble varié de bons candidats, pour couvrir l'espace et ne pas passer à côté de bonnes solutions.

---

**[Avancer : Slide 8 — Les méthodes comparées]**

Nous comparons cinq méthodes au total. Les quatre premières ont le même budget oracle — c'est la comparaison équitable.

La méthode **Supervisée** : on entraîne une seule fois un prédicteur sur un dataset fixe, et on classe un grand pool de candidats par score prédit.
Simple, mais sans boucle de feedback — on ne s'adapte pas à ce qu'on découvre.

La méthode **Active** : c'est la boucle d'apprentissage actif classique avec un substitut et une acquisition, réentraîné à chaque ronde.
C'est notre baseline principale, et elle est déjà très solide.

Le **GFlowNet direct** : on entraîne un GFlowNet directement sur la vraie récompense oracle, sans passer par un substitut.
Il doit donc apprendre avec le signal sparse qu'on a décrit.

Et notre méthode principale, **l'Hybride** : on combine les deux.
D'abord un substitut, puis un GFlowNet entraîné sur ce substitut, puis une acquisition pour sélectionner ce qui va à l'oracle.

Enfin, la **Borne Supérieure** utilise un budget beaucoup plus grand — elle sert de plafond théorique, pas de comparaison équitable.

---

**[Avancer : Slide 9 — Vue d'ensemble des pipelines]**

Pour bien lire les résultats qu'on va vous montrer, il faut comprendre ce que chaque méthode fait vraiment.

Le Supervisé entraîne une fois et classe.
L'Actif réentraîne son substitut à chaque ronde et choisit activement quoi évaluer.
Le GFlowNet direct génère des candidats en utilisant directement la vraie récompense oracle.
L'Hybride enchaîne : substitut d'abord, puis génération par GFlowNet, puis sélection par acquisition.

On compare donc à la fois des approches prédictives, des approches génératives, et leur combinaison.

Je passe maintenant la parole à Yann-Olivier.

---

---
## YANN-OLIVIER — Slides 12, 16, 17, 18, 20, 22 [~3:40]
---

**[Avancer : Slide 12 — Comment fonctionne la méthode hybride]**

Merci Charbel. Avant de passer aux chiffres, je veux vous expliquer précisément pourquoi notre méthode hybride est construite comme elle l'est.

Dans le GFlowNet direct, le modèle doit apprendre depuis un signal presque toujours nul — les mots invalides retournent zéro, et avec un budget limité, il n'y a pas assez d'exemples positifs pour que le modèle apprenne correctement où chercher dans l'espace.

L'hybride résout ce problème en **découplant les rôles**.
Le substitut — ici un Gaussian Process — reçoit les vraies évaluations oracle et construit une représentation plus dense et plus régulière de la récompense.
Le GFlowNet s'entraîne sur la récompense **prédite par ce substitut**, pas sur l'oracle direct.
Il reçoit donc un signal beaucoup plus riche pour apprendre où se trouvent les bons candidats.
Une fois qu'il a généré des candidats divers et prometteurs, la fonction d'acquisition filtre et n'envoie que les meilleurs à l'oracle réel.

En une phrase : le **GFlowNet propose**, l'**apprentissage actif sélectionne**.

---

**[Avancer : Slide 16 — Ce qu'on mesure]**

Avant les chiffres, je veux vous expliquer ce qu'on mesure exactement, parce que ça conditionne toute l'interprétation.

On utilise cinq métriques.
**Best** : le meilleur score trouvé par la méthode, tout confondu.
**Top-10** : la moyenne des dix meilleurs scores découverts — ça mesure si la méthode trouve plusieurs bonnes solutions, pas juste une par chance.
**Score moyen** : la moyenne des scores de tous les candidats évalués — ça mesure l'efficacité globale du budget.
**Valides (%)** : la proportion de vrais mots anglais parmi toutes les requêtes oracle — ça mesure si la méthode cherche dans les bonnes zones de l'espace.
Et **Q90** : combien de requêtes sont nécessaires pour atteindre 90% du meilleur score final — plus ce nombre est petit, plus la méthode est rapide.

Ces métriques permettent de distinguer une méthode qui a eu un seul coup de chance d'une méthode qui construit vraiment une bonne collection de solutions.

---

**[Avancer : Slide 17 — Résultats principaux]**

Voici le tableau de résultats. Je vais vous le lire et vous expliquer pourquoi, malgré les apparences, c'est l'Hybride qui gagne.

Le **Supervisé** a le meilleur Best : 15,6.
À première vue, on pourrait croire que c'est la meilleure méthode. Mais ce serait une erreur.
Ce Best élevé vient d'un très grand pool de candidats classés une seule fois.
Ça ne dit pas que la méthode est efficace — juste qu'elle a correctement prédit un ou deux bons candidats parmi un grand nombre.

L'**Actif** est une baseline solide et équilibrée : Best 14,4, Top-10 de 13,1, Q90 de 297 requêtes.

Le **GFlowNet direct** souffre clairement : Best de seulement 12,6, taux de validité de 6,3%, Q90 de 377.
Le signal sparse l'empêche d'apprendre efficacement sous budget limité.

L'**Hybride**, notre méthode : Best de 14,2 — légèrement inférieur au Supervisé.
Mais regardez ce qui compte vraiment.
Top-10 de 13,2 : le meilleur parmi les méthodes budget-matchées.
Score moyen de 1,216 : le meilleur.
Taux de validité de 14,6% : le meilleur.
Et Q90 de seulement **148 requêtes** : presque deux fois plus rapide que l'Actif, et plus de trois fois plus rapide que le Supervisé.

---

**[Avancer : Slide 18 — Lecture des résultats]**

Pour résumer l'interprétation : si notre objectif était de trouver un seul bon mot coûte que coûte, le Supervisé gagnerait.
Mais ce n'est pas notre objectif.
Notre objectif est de trouver **rapidement beaucoup de bons mots valides**, avec un budget limité.

Sur ce critère réel, l'Hybride est clairement le meilleur.
Il construit la meilleure frontière de solutions, il est le plus rapide à atteindre des scores élevés,
et il dépense son budget sur des candidats qui sont vraiment des mots valides.
Ce n'est pas un coup de chance — c'est une méthode efficace.

---

**[Avancer : Slide 20 — Pourquoi l'hybride marche mieux]**

L'explication tient en trois points.
Le substitut donne au GFlowNet un signal dense et exploitable, là où l'oracle donnait presque uniquement des zéros.
Le GFlowNet génère des candidats divers et bien ciblés dans l'espace des bons mots.
Et l'acquisition ne gaspille pas le budget sur des candidats redondants ou peu prometteurs.

C'est la **complémentarité** de ces trois composants qui produit ces résultats.
Pas une seule idée brillante isolée, mais une architecture où chaque pièce résout le problème de la pièce précédente.

---

**[Avancer : Slide 22 — Conclusion]**

Pour conclure, les quatre leçons principales de notre projet.

Premièrement, la **parcimonie de la récompense** est l'obstacle central dans ce type de problème de découverte.
La plupart des méthodes échouent principalement à cause de ça.

Deuxièmement, l'**apprentissage actif** est une stratégie très puissante pour allouer intelligemment un budget rare —
même une baseline simple comme la nôtre est déjà très solide.

Troisièmement, le **GFlowNet direct** est trop fragilisé par le signal sparse quand le budget est contraint.
Ce n'est pas que l'idée des GFlowNets est mauvaise — c'est qu'on lui demande de trop dans ce régime.

Et quatrièmement, notre méthode **hybrid_gfn_only** donne le meilleur compromis global :
elle combine la diversité et l'expressivité du GFlowNet avec la sélectivité de l'apprentissage actif,
et c'est cette combinaison qui lui permet de trouver le plus de bons candidats valides
en dépensant le budget le plus intelligemment.

Merci pour votre attention. On est disponibles pour vos questions.

---

> **Temps estimé par section :**
> - Yohan (slides 1–5) : ~3:10
> - Charbel (slides 6–9) : ~3:10
> - Yann-Olivier (slides 12, 16–18, 20, 22) : ~3:40
> - **Total : ~10:00**
>
> **Note :** Les slides très techniques (10, 11, 13, 14, 15, 19, 21) sont à l'écran mais ne sont pas lues
> en détail pour tenir dans le temps. Si une question porte sur ces détails, les chiffres précis sont dans les slides.
