#!/bin/bash
# Script pour synchroniser le fork avec upstream
# Usage: ./sync_fork.sh

set -e  # Exit on error

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}🔄 Synchronisation du fork avec upstream${NC}"
echo ""

# Sauvegarder la branche courante
current_branch=$(git branch --show-current)
echo -e "${BLUE}📍 Branche courante: ${current_branch}${NC}"

# Synchroniser main
echo -e "${YELLOW}1️⃣  Synchronisation de main avec upstream/main${NC}"
git checkout main
git fetch upstream
git reset --hard upstream/main
git push fork main --force
echo -e "${GREEN}✅ main synchronisé${NC}"
echo ""

# Mettre à jour la branche de feature si elle existe
if git show-ref --verify --quiet refs/heads/feat/gpt5-mini-support; then
    echo -e "${YELLOW}2️⃣  Mise à jour de feat/gpt5-mini-support${NC}"
    git checkout feat/gpt5-mini-support

    # Tenter le rebase
    if git rebase main; then
        echo -e "${GREEN}✅ Rebase réussi${NC}"
        echo -e "${YELLOW}   Push vers fork (force-with-lease)...${NC}"
        git push fork feat/gpt5-mini-support --force-with-lease
        echo -e "${GREEN}✅ feat/gpt5-mini-support mis à jour${NC}"
    else
        echo -e "${RED}❌ Conflits détectés pendant le rebase${NC}"
        echo -e "${YELLOW}   Résolvez les conflits puis exécutez:${NC}"
        echo -e "   git add <fichiers-résolus>"
        echo -e "   git rebase --continue"
        echo -e "   git push fork feat/gpt5-mini-support --force-with-lease"
        echo ""
        echo -e "${YELLOW}   Ou annulez avec: git rebase --abort${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠️  Branche feat/gpt5-mini-support non trouvée (ignorée)${NC}"
fi

# Retourner sur la branche d'origine
if [ "$current_branch" != "$(git branch --show-current)" ]; then
    echo ""
    echo -e "${BLUE}↩️  Retour sur ${current_branch}${NC}"
    git checkout "$current_branch"
fi

echo ""
echo -e "${GREEN}✅ Synchronisation terminée!${NC}"
echo ""
echo -e "${BLUE}📊 État actuel:${NC}"
git log --oneline --graph --all --decorate -n 5
