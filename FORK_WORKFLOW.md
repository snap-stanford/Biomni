# 🔄 Workflow Fork Biomni - gpt-5-mini support

Ce document explique comment gérer ce fork avec les modifications pour gpt-5-mini tout en profitant des mises à jour upstream.

## 📊 Architecture du fork

```
upstream/main (snap-stanford/Biomni)
    ↓ sync
fork/main (mickaelleclercq/Biomni) ← toujours identique à upstream
    ↓ rebase
fork/feat/gpt5-mini-support ← VOS modifications pour gpt-5-mini
```

## 🚀 Usage quotidien

### Utiliser Biomni avec gpt-5-mini

```bash
cd /home/mickael/biomni
git use-gpt5  # Alias pour: git checkout feat/gpt5-mini-support
conda activate biomni_e1
python run_biomni.py --source OpenAI "votre prompt"
```

### Vérifier sur quelle branche vous êtes

```bash
git branch  # L'étoile montre la branche active
```

## 🔄 Synchronisation avec upstream

### Option 1: Alias rapide (recommandé)

```bash
# Synchroniser main (1x par semaine/mois)
git sync-main

# Mettre à jour votre branche de feature
git checkout feat/gpt5-mini-support
git update-feature
```

### Option 2: Commandes complètes

```bash
# Synchroniser main avec upstream
git checkout main
git fetch upstream
git reset --hard upstream/main
git push fork main --force

# Mettre à jour votre branche de feature
git checkout feat/gpt5-mini-support
git rebase main
# Si conflits, résoudre puis:
git rebase --continue
git push fork feat/gpt5-mini-support --force-with-lease
```

## 📝 Modifications apportées

### Commit 1: Support OpenAI Responses API
**Fichier**: `biomni/llm.py`
- Classe `_ChatOpenAIResponsesNoStop` pour gpt-5
- Suppression du paramètre `stop` (incompatible avec Responses API)
- Force `use_responses_api=True` et `output_version="v0"`

### Commit 2: Normalisation retriever
**Fichier**: `biomni/model/retriever.py`
- Gestion des listes de content blocks (Responses API)
- Conversion en string avant parsing regex

### Commit 3: Normalisation agent A1
**Fichier**: `biomni/agent/a1.py`
- Flatten des content blocks dans `generate()`
- Extraction propre des blocs "text" avant parsing `<execute>`/`<solution>`

## 🔧 Alias Git configurés

| Alias | Commande | Description |
|-------|----------|-------------|
| `git sync-main` | Sync main avec upstream | Synchronise votre main avec upstream/main |
| `git update-feature` | Rebase sur upstream | Met à jour la branche courante avec upstream/main |
| `git use-gpt5` | Checkout feat/gpt5-mini-support | Bascule sur votre branche de travail |

## ⚠️ Gestion des conflits

Si lors d'un `git update-feature` vous avez des conflits :

```bash
# 1. Voir les fichiers en conflit
git status

# 2. Éditer les fichiers et résoudre les conflits
# Chercher les marqueurs: <<<<<<<, =======, >>>>>>>

# 3. Marquer comme résolu et continuer
git add <fichier-résolu>
git rebase --continue

# Si trop complexe, annuler et demander de l'aide
git rebase --abort
```

### Fichiers à risque de conflit

- `biomni/llm.py` - Si upstream ajoute d'autres modèles
- `biomni/agent/a1.py` - Si upstream modifie la fonction `generate()`
- `biomni/model/retriever.py` - Si upstream modifie le parsing

## 🎯 Contribuer à upstream

Si vous voulez proposer vos modifications à snap-stanford :

1. **Via GitHub UI** :
   - Aller sur https://github.com/mickaelleclercq/Biomni
   - Cliquer sur "Compare & pull request"
   - Base: `snap-stanford/Biomni:main`
   - Compare: `mickaelleclercq/Biomni:feat/gpt5-mini-support`

2. **Si la PR est acceptée** :
   ```bash
   git sync-main  # Récupérer vos changements depuis upstream
   git branch -D feat/gpt5-mini-support  # Plus besoin de la branche!
   ```

## 📦 Structure des remotes

```bash
git remote -v
# fork      https://github.com/mickaelleclercq/Biomni.git
# origin    https://github.com/snap-stanford/Biomni.git
# upstream  https://github.com/snap-stanford/Biomni.git
```

- **upstream**: Dépôt officiel snap-stanford (lecture seule pour sync)
- **fork**: Votre fork GitHub (push vos modifications)
- **origin**: Pointer vers snap-stanford (legacy, peu utilisé)

## 🔍 Commandes de diagnostic

```bash
# Voir les différences entre votre branche et upstream
git diff upstream/main feat/gpt5-mini-support

# Voir l'historique de vos commits
git log upstream/main..feat/gpt5-mini-support --oneline

# Vérifier l'état du dépôt
git status
```

## 📅 Routine recommandée

**Hebdomadaire** :
```bash
git sync-main
git checkout feat/gpt5-mini-support
git update-feature
```

**Avant chaque utilisation** :
```bash
git use-gpt5  # ou: git checkout feat/gpt5-mini-support
conda activate biomni_e1
```

---

**Créé le**: 21 octobre 2025  
**Auteur**: Mickael Leclercq  
**Base upstream**: snap-stanford/Biomni @ 8fd7b21
