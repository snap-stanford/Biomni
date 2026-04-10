#!/bin/bash

# Biomni Environment Setup Script (Simplified)

ENV_NAME="biomni_flow"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}=== Biomni Environment Setup ===${NC}"

# Check for conda or micromamba
if command -v conda &> /dev/null; then
    PKG_MANAGER="conda"
elif command -v micromamba &> /dev/null; then
    PKG_MANAGER="micromamba"
else
    echo -e "${RED}Error: Neither conda nor micromamba is installed.${NC}"
    exit 1
fi

echo -e "${YELLOW}Using package manager: $PKG_MANAGER${NC}"

# Activate shell
if [ "$PKG_MANAGER" = "micromamba" ]; then
    eval "$($MAMBA_EXE shell hook --shell bash)"
else
    eval "$(conda shell.bash hook)"
fi

# Step 1: Create base environment
echo -e "\n${YELLOW}Creating environment: $ENV_NAME${NC}"
$PKG_MANAGER env create -n "$ENV_NAME" -f environment.yml
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to create environment.${NC}"
    exit 1
fi

# Step 2: Activate environment
echo -e "\n${YELLOW}Activating environment...${NC}"
$PKG_MANAGER activate "$ENV_NAME"

# Step 3: List of YAML files to install
YML_FILES=(
    "r_packages.yml"
)

# Step 4: Install all YAMLs
for file in "${YML_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "\n${YELLOW}Installing from $file...${NC}"
        $PKG_MANAGER env update -n "$ENV_NAME" -f "$file"

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install $file${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}Warning: $file not found, skipping.${NC}"
    fi
done

# Step 5: Install additional R packages
if [ -f "install_r_packages.R" ]; then
    echo -e "\n${YELLOW}Installing additional R packages...${NC}"
    Rscript install_r_packages.R
else
    echo -e "${YELLOW}No install_r_packages.R found, skipping.${NC}"
fi

# Done
echo -e "\n${GREEN}=== Setup Completed Successfully! ===${NC}"
echo -e "Activate with: ${YELLOW}$PKG_MANAGER activate $ENV_NAME${NC}"