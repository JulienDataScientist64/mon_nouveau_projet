%cd /content


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# Supprimer les configurations Git globales
!rm -f ~/.gitconfig

# Supprimer le dossier SSH
!rm -rf ~/.ssh


from google.colab import drive
drive.mount('/content/drive')


!mkdir -p ~/.ssh
!cp /content/drive/MyDrive/ssh_keys/id_rsa ~/.ssh/id_rsa
!cp /content/drive/MyDrive/ssh_keys/id_rsa.pub ~/.ssh/id_rsa.pub
!chmod 600 ~/.ssh/id_rsa

# Ajouter GitHub aux hôtes connus pour éviter les avertissements
!ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts

# Vérifier la connexion SSH
!ssh -T git@github.com


!git config --global user.name "JulienDataScientist64"
!git config --global user.email "jcantalapiedra1@gmail.com"


import os

new_project_path = '/content/drive/MyDrive/test'
os.makedirs(new_project_path, exist_ok=True)
%cd {new_project_path}


with open("README.md", "w") as f:
    f.write("# Test\n\nDescription de mon projet.")


!git init


with open(".gitignore", "w") as gitignore:
    gitignore.write("""\
# Ignorer les fichiers CSV et volumineux
*.csv
*.pkl
*.npz

# Ignorer les checkpoints Jupyter Notebook
.ipynb_checkpoints/

# Ignorer les caches Python
__pycache__/
*.py[cod]

# Ignorer les fichiers temporaires ou de sauvegarde
*.tmp
*.log
*.bak
*.swp
~*

# Ignorer les fichiers système
.DS_Store

# Ignorer le dossier data/
data/
""")


!git add .
!git commit -m "Initial commit: ajout du README et configuration initiale"


# Vérifie si un remote 'origin' existe déjà
!git remote -v

# Si un remote 'origin' existe déjà et est incorrect, supprime-le
!git remote remove origin

# Ajouter le remote correct
!git remote add origin git@github.com:JulienDataScientist64/mon_nouveau_projet.git


!git branch -M main


!git push -u origin main


!pip freeze > requirements.txt


!git add requirements.txt
!git commit -m "Add requirements.txt for reproducibility"
!git push


!pip install mlflow


import mlflow
import mlflow.sklearn


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Importing fetch_california_housing instead of load_boston
from sklearn.datasets import fetch_california_housing
import mlflow

# Charger les données using fetch_california_housing
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Démarrer une nouvelle run MLflow
with mlflow.start_run():
    # Initialiser et entraîner le modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions
    predictions = model.predict(X_test)

    # Calculer une métrique
    mse = ((predictions - y_test) ** 2).mean()

    # Enregistrer les métriques et le modèle
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print(f"Mean Squared Error: {mse}")

!echo ".mlruns/" >> .gitignore
!git add .gitignore
!git commit -m "Update .gitignore to exclude MLflow runs"
!git push


os.makedirs('notebooks', exist_ok=True)
os.makedirs('src', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)


with open("README.md", "a") as f:
    f.write("""
## Installation

Clone le dépôt et installe les dépendances :

```bash
git clone git@github.com:JulienDataScientist64/test.git
cd test
pip install -r requirements.txt
""")

!git add .
!git commit -m "Description de la modification"
!git push


# Le notebook actuel est 'mon_notebook_colab.ipynb' dans le drive G:/Mon Drive/Colab Notebooks
!mv "/test/notebooks/mon_notebook_colab.ipynb" notebooks/



!ls "/content/drive/My Drive/Colab Notebooks/"


# Affiche le répertoire actuel
!pwd

# Liste les fichiers dans le répertoire actuel
!ls



# Déplacer le fichier
!mv "/content/drive/My Drive/Colab Notebooks/mon_notebook_colab.ipynb" "/content/drive/My Drive/Colab Notebooks/notebooks/"

!pip install gensim

!pip install pipreqs

with open("test_imports.py", "w") as f:
    f.write("""\
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
""")


!pipreqs /content/drive/MyDrive/test --force
!cat /content/drive/MyDrive/test/requirements.txt


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# Importing fetch_california_housing instead of load_boston
from sklearn.datasets import fetch_california_housing
import mlflow

# Charger les données using fetch_california_housing
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Démarrer une nouvelle run MLflow
with mlflow.start_run():
    # Initialiser et entraîner le modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédictions
    predictions = model.predict(X_test)

    # Calculer une métrique
    mse = ((predictions - y_test) ** 2).mean()

    # Enregistrer les métriques et le modèle
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")

    print(f"Mean Squared Error: {mse}")


!pip install gensim

import mlflow
import gensim


!jupyter nbconvert --to script /content/drive/MyDrive/test/notebooks/mon_notebook_colab.ipynb


!pipreqs /content/drive/MyDrive/test --force


!cat /content/drive/MyDrive/test/requirements.txt













# Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Configurer SSH
!mkdir -p ~/.ssh
!cp /content/drive/MyDrive/ssh_keys/id_rsa ~/.ssh/id_rsa
!cp /content/drive/MyDrive/ssh_keys/id_rsa.pub ~/.ssh/id_rsa.pub
!chmod 600 ~/.ssh/id_rsa
!ssh-keyscan -t rsa github.com >> ~/.ssh/known_hosts
!ssh -T git@github.com

# Configurer Git (Nom et email)
!git config --global user.name "JulienDataScientist64"
!git config --global user.email "jcantalapiedra1@gmail.com"

# Naviguer dans le répertoire de ton projet (cloné ou existant dans Colab)
%cd /content/drive/MyDrive/test

# Synchroniser avec les changements distants
!git pull origin main --allow-unrelated-histories

# Vérifier l'état des modifications
!git status

# Ajouter toutes les modifications
!git add .

# Créer un commit avec une description claire
!git commit -m "Description des changements du jour"

# Pousser les modifications vers GitHub
!git push origin main

