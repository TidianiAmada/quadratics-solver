# Résolveur de fonction quadratique par la méthode de Newton Multi-Variable avec Matrice Hessiennne

Ce projet est une application Flask qui implémente la **méthode de Newton pour plusieurs variables**, avec une gestion explicite du **Hessien**. Elle permet de résoudre des systèmes d'équations non linéaires en utilisant des approximations locales pour trouver une solution proche des points critiques (racines).  

## Méthode : `newton_method_with_hessian`

### Contexte
La méthode de Newton pour plusieurs variables est une extension de la méthode de Newton classique à une dimension. Elle est utilisée pour résoudre des systèmes d'équations non linéaires sous la forme :

\[
\mathbf{F}(\mathbf{x}) = 0
\]

où :
- \( \mathbf{F} \) est un vecteur de fonctions non linéaires,
- \( \mathbf{x} \) est le vecteur des variables inconnues.  

La méthode repose sur une série d'itérations où la solution est mise à jour à l'aide de la matrice jacobienne (les dérivées partielles de \( \mathbf{F} \)) et, dans ce cas, du **Hessien**, qui représente les dérivées secondes.  

### Fonctionnement de `newton_method_with_hessian`

#### Paramètres :
1. **`func_str`** : La chaîne de caractères représentant la fonction ou le système d'équations à résoudre (par exemple, `"x**2 + y**2 - 4"`).
2. **`variables`** : Une liste des noms des variables dans la fonction (par exemple, `["x", "y"]`).
3. **`initial_guesses`** : Les valeurs initiales des variables (par exemple, `[1.0, 1.0]`).
4. **`tolerance`** : La précision désirée pour l'arrêt des itérations (par défaut \( 10^{-7} \)).
5. **`max_iterations`** : Le nombre maximal d'itérations autorisées pour éviter les boucles infinies.

#### Étapes principales :
1. **Initialisation** :
   - Les variables sont symbolisées à l'aide de `sympy`.
   - La fonction, le gradient (vecteur des dérivées premières), et le Hessien (matrice des dérivées secondes) sont calculés symboliquement.

2. **Itérations** :
   - Calcul de la valeur actuelle de la fonction \( f(x) \), du gradient, et du Hessien.
   - Résolution du système linéaire \( H \Delta x = -\nabla f(x) \) pour déterminer le pas de mise à jour \( \Delta x \).
   - Mise à jour des variables : \( x_{\text{nouveau}} = x_{\text{actuel}} + \Delta x \).
   - Vérification de la convergence : arrêt si la norme du gradient ou le changement relatif des variables est inférieur à la tolérance.

3. **Résultats** :
   - Si la convergence est atteinte, la solution finale est retournée.
   - Sinon, un message d'erreur est généré si le nombre maximal d'itérations est dépassé.

#### Exemple d'appel :
```python
result = newton_method_multi_variable(
    func_str="x**2 + y**2 - 4",
    variables=["x", "y"],
    initial_guesses=[1.0, 1.0],
    tolerance=1e-7,
    max_iterations=100
)
```

#### Exemple de résultat :
```json
{
    "status": "success",
    "solution": [1.414213, 1.414213],
    "iterations": [
        {
            "iteration": 1,
            "x": [1.0, 1.0],
            "f(x)": -2.0,
            "gradient": [2.0, 2.0],
            "x_new": [1.414213, 1.414213]
        }
    ]
}
```

## Application Flask

L'application utilise la méthode `newton_method_with_hessian` pour permettre aux utilisateurs de résoudre des systèmes d'équations à 1, 2 ou 3 variables via une interface web interactive.

### Fonctionnalités :
- **Formulaire utilisateur** :
  - Entrée de la fonction cible.
  - Déclaration des variables et de leurs valeurs initiales.
  - Paramètres de tolérance et nombre maximal d'itérations.
- **Résultats dynamiques** :
  - Visualisation de chaque étape des itérations.
  - Solution finale ou message d'erreur en cas de non-convergence.
- **Interface utilisateur moderne** grâce à **Bootstrap 5**.

### Fichiers principaux :
- **`controller.py`** : Fichier Flask principal gérant les routes et l'intégration avec le solveur.
- **`templates/index.html`** : Interface web pour les utilisateurs.
- **`newton_solver.py`** : Module contenant l'implémentation de `newton_method_with_hessian`.

## Installation et Exécution

### Pré-requis :
- Python 3.8+
- Bibliothèques : Flask, SymPy

### Étapes :
1. Clonez le dépôt :
   
2. Installez les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Lancez l'application Flask :
   ```bash
   python -m controller
   ```
4. Ouvrez un navigateur et accédez à : `http://127.0.0.1:5000`


## Limites
- La méthode nécessite que la fonction soit différentiable.
- Peut ne pas converger si l'estimation initiale est trop éloignée de la solution.
