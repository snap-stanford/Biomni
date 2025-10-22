#!/bin/bash
# Script pour utiliser Biomni avec gpt-5-mini
# Usage: ./run_gpt5.sh "votre prompt"

set -e  # Exit on error

# Couleurs pour output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Biomni avec gpt-5-mini${NC}"
echo ""

# Vérifier qu'on est sur la bonne branche
current_branch=$(git branch --show-current)
if [ "$current_branch" != "feat/gpt5-mini-support" ]; then
    echo -e "${YELLOW}⚠️  Basculement sur feat/gpt5-mini-support${NC}"
    git checkout feat/gpt5-mini-support
fi

# Activer l'environnement conda
echo -e "${GREEN}✅ Activation de l'environnement biomni_e1${NC}"
eval "$(conda shell.bash hook)"
conda activate biomni_e1

# Lancer Biomni avec le prompt
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: $0 \"votre prompt\"${NC}"
    echo "Exemple: $0 \"create mock gene expression data...\""
    exit 1
fi

echo -e "${GREEN}✅ Lancement de Biomni avec OpenAI gpt-5-mini${NC}"
echo ""
python run_biomni.py --source OpenAI "$1"

echo ""
echo -e "${GREEN}✅ Terminé!${NC}"
