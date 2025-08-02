#!/bin/bash

# BioAgentOS - Biomni Environment Setup Script
# This script sets up a comprehensive bioinformatics environment with various tools and packages

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default tools directory is the current directory
DEFAULT_TOOLS_DIR="$(pwd)/biomni_tools"
TOOLS_DIR=""

echo -e "${YELLOW}=== Biomni Environment Setup ===${NC}"
echo -e "${BLUE}This script will set up a comprehensive bioinformatics environment with various tools and packages.${NC}"

# Check if conda is installed
if ! command -v conda &> /dev/null && ! command -v micromamba &> /dev/null; then
    echo -e "${RED}Error: Conda is not installed or not in PATH.${NC}"
    echo "Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# redirect to micromamba if needed
if ! command -v conda &> /dev/null && command -v micromamba &> /dev/null; then
    conda() {
        micromamba "$@"
    }
    export -f conda
fi

# Function to handle errors
handle_error() {
    local exit_code=$1
    local error_message=$2
    local optional=${3:-false}

    if [ $exit_code -ne 0 ]; then
        echo -e "${RED}Error: $error_message${NC}"
        if [ "$optional" = true ]; then
            echo -e "${YELLOW}Continuing with setup as this component is optional.${NC}"
            return 0
        else
            if [ -z "$NON_INTERACTIVE" ]; then
                read -p "Continue with setup? (y/n) " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo -e "${RED}Setup aborted.${NC}"
                    exit 1
                fi
            else
                echo -e "${YELLOW}Non-interactive mode: continuing despite error.${NC}"
            fi
        fi
    fi
    return $exit_code
}

# Function to install a specific environment file
install_env_file() {
    local env_file=$1
    local description=$2
    local optional=${3:-false}

    echo -e "\n${BLUE}=== Installing $description ===${NC}"

    if [ "$optional" = true ]; then
        if [ -z "$NON_INTERACTIVE" ]; then
            read -p "Do you want to install $description? (y/n) " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                echo -e "${YELLOW}Skipping $description installation.${NC}"
                return 0
            fi
        else
            echo -e "${YELLOW}Non-interactive mode: automatically installing $description.${NC}"
        fi
    fi

    echo -e "${YELLOW}Installing $description from $env_file...${NC}"
    conda env update -f $env_file
    handle_error $? "Failed to install $description." $optional

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully installed $description!${NC}"
    fi
}

# Function to install CLI tools
install_cli_tools() {
    echo -e "\n${BLUE}=== Installing Command-Line Bioinformatics Tools ===${NC}"

    # Ask user for the directory to install CLI tools
    if [ -z "$NON_INTERACTIVE" ]; then
        echo -e "${YELLOW}Where would you like to install the command-line tools?${NC}"
        echo -e "${BLUE}Default: $DEFAULT_TOOLS_DIR${NC}"
        read -p "Enter directory path (or press Enter for default): " user_tools_dir
    else
        user_tools_dir=""
        echo -e "${YELLOW}Non-interactive mode: using default directory $DEFAULT_TOOLS_DIR for CLI tools.${NC}"
    fi

    if [ -z "$user_tools_dir" ]; then
        TOOLS_DIR="$DEFAULT_TOOLS_DIR"
    else
        TOOLS_DIR="$user_tools_dir"
    fi

    # Export the tools directory for the CLI tools installer
    export BIOMNI_TOOLS_DIR="$TOOLS_DIR"

    echo -e "${YELLOW}Installing command-line tools (PLINK, IQ-TREE, GCTA, etc.) to $TOOLS_DIR...${NC}"

    # Set environment variable to skip prompts in the CLI tools installer
    export BIOMNI_AUTO_INSTALL=1

    # Run the CLI tools installer
    bash install_cli_tools.sh
    handle_error $? "Failed to install CLI tools." true

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully installed command-line tools!${NC}"

        # Create a setup_path.sh file in the current directory
        echo "#!/bin/bash" > setup_path.sh
        echo "# Added by biomni setup" >> setup_path.sh
        echo "# Remove any old paths first to avoid duplicates" >> setup_path.sh
        echo "PATH=\$(echo \$PATH | tr ':' '\n' | grep -v \"biomni_tools/bin\" | tr '\n' ':' | sed 's/:$//')" >> setup_path.sh
        echo "export PATH=\"$TOOLS_DIR/bin:\$PATH\"" >> setup_path.sh
        chmod +x setup_path.sh

        echo -e "${GREEN}Created setup_path.sh in the current directory.${NC}"
        echo -e "${YELLOW}You can add the tools to your PATH by running:${NC}"
        echo -e "${GREEN}source $(pwd)/setup_path.sh${NC}"

        # Also add to the current session
        # Remove any old paths first to avoid duplicates
        PATH=$(echo $PATH | tr ':' '\n' | grep -v "biomni_tools/bin" | tr '\n' ':' | sed 's/:$//')
        export PATH="$TOOLS_DIR/bin:$PATH"
    fi

    # Unset the environment variables
    unset BIOMNI_AUTO_INSTALL
    unset BIOMNI_TOOLS_DIR
}

# Main installation process
main() {
    # Step 1: Create base conda environment
    echo -e "\n${YELLOW}Step 1: Creating base environment from environment.yml...${NC}"
    conda env create -n biomni_e1 -f environment.yml
    handle_error $? "Failed to create base conda environment."

    # Step 2: Activate the environment
    echo -e "\n${YELLOW}Step 2: Activating conda environment...${NC}"
    if command -v micromamba &> /dev/null; then
        eval "$("$MAMBA_EXE" shell hook --shell bash)"
        micromamba activate biomni_e1
    else
        eval "$(conda shell.bash hook)"
        conda activate biomni_e1
    fi
    handle_error $? "Failed to activate biomni_e1 environment."

    # Step 3: Install core bioinformatics tools (including QIIME2)
    echo -e "\n${YELLOW}Step 3: Installing core bioinformatics tools (including QIIME2)...${NC}"
    install_env_file "bio_env.yml" "core bioinformatics tools"

    # Step 4: Install R packages
    echo -e "\n${YELLOW}Step 4: Installing R packages...${NC}"
    install_env_file "r_packages.yml" "core R packages"

    # Step 5: Install additional R packages through R's package manager
    echo -e "\n${YELLOW}Step 5: Installing additional R packages through R's package manager...${NC}"
    Rscript install_r_packages.R
    handle_error $? "Failed to install additional R packages." true

    # Step 6: Install CLI tools
    echo -e "\n${YELLOW}Step 6: Installing command-line bioinformatics tools...${NC}"
    install_cli_tools


    # Setup completed
    echo -e "\n${GREEN}=== Biomni Environment Setup Completed! ===${NC}"
    echo -e "You can now run the example analysis with: ${YELLOW}python bio_analysis_example.py${NC}"
    echo -e "To activate this environment in the future, run: ${YELLOW}conda activate biomni_e1${NC}"
    echo -e "To use BioAgentOS, navigate to the BioAgentOS directory and follow the instructions in the README."

    # Display CLI tools setup instructions
    if [ -n "$TOOLS_DIR" ]; then
        echo -e "\n${BLUE}=== Command-Line Tools Setup ===${NC}"
        echo -e "The command-line tools are installed in: ${YELLOW}$TOOLS_DIR${NC}"
        echo -e "To add these tools to your PATH, run: ${YELLOW}source $(pwd)/setup_path.sh${NC}"
        echo -e "You can also add this line to your shell profile for permanent access:"
        echo -e "${GREEN}export PATH=\"$TOOLS_DIR/bin:\$PATH\"${NC}"

        # Test if tools are accessible
        echo -e "\n${BLUE}=== Testing CLI Tools ===${NC}"
        if command -v plink2 &> /dev/null; then
            echo -e "${GREEN}PLINK2 is accessible in the current PATH${NC}"
            echo -e "PLINK2 location: $(which plink2)"
        else
            echo -e "${RED}PLINK2 is not accessible in the current PATH${NC}"
            echo -e "Please run: ${YELLOW}source $(pwd)/setup_path.sh${NC} to update your PATH"
        fi

        if command -v gcta64 &> /dev/null; then
            echo -e "${GREEN}GCTA is accessible in the current PATH${NC}"
            echo -e "GCTA location: $(which gcta64)"
        else
            echo -e "${RED}GCTA is not accessible in the current PATH${NC}"
            echo -e "Please run: ${YELLOW}source $(pwd)/setup_path.sh${NC} to update your PATH"
        fi

        if command -v iqtree2 &> /dev/null; then
            echo -e "${GREEN}IQ-TREE is accessible in the current PATH${NC}"
            echo -e "IQ-TREE location: $(which iqtree2)"
        else
            echo -e "${RED}IQ-TREE is not accessible in the current PATH${NC}"
            echo -e "Please run: ${YELLOW}source $(pwd)/setup_path.sh${NC} to update your PATH"
        fi
    fi

    PATH=$(echo $PATH | tr ':' '\n' | grep -v "biomni_tools/bin" | tr '\n' ':' | sed 's/:$//')
    export PATH="$(pwd)/biomni_tools/bin:$PATH"
}

# Run the main installation process
main
