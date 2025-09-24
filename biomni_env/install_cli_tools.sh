#!/bin/bash

# Script to install command-line bioinformatics tools

# Set up colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check for wget or fallback to curl
if command -v wget &> /dev/null; then
  DOWNLOADER="wget -v -O"
elif command -v curl &> /dev/null; then
  DOWNLOADER="curl -L -o"
  echo -e "${YELLOW}Warning: wget not found, using curl instead.${NC}"
else
  echo -e "${RED}Error: Neither wget nor curl is installed.${NC}"
  echo -e "${YELLOW}Please install one of them (e.g., brew install wget on macOS).${NC}"
  exit 1
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}jq is not installed. Installing jq for JSON parsing...${NC}"
    # Use conda to install jq (no sudo required)
    conda install -y -c conda-forge jq

    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to install jq with conda. Please install it manually.${NC}"
        echo "Visit: https://stedolan.github.io/jq/download/"
        exit 1
    fi
fi

# Function to clean up old installations
cleanup_old_installations() {
    echo -e "${YELLOW}Checking for old installations...${NC}"

    # Check for old installation in AFS
    local old_afs_dir="/afs/cs.stanford.edu/u/$(whoami)/biomni_tools"
    if [ -d "$old_afs_dir" ]; then
        echo -e "${YELLOW}Found old installation in $old_afs_dir${NC}"
        echo -e "${YELLOW}Removing old bin directory...${NC}"
        rm -rf "$old_afs_dir/bin"
        echo -e "${GREEN}Old bin directory removed.${NC}"
    fi

    # Check for old installation in HOME
    local old_home_dir="$HOME/biomni_tools"
    if [ -d "$old_home_dir" ] && [ "$old_home_dir" != "$TOOLS_DIR" ]; then
        echo -e "${YELLOW}Found old installation in $old_home_dir${NC}"
        echo -e "${YELLOW}Removing old bin directory...${NC}"
        rm -rf "$old_home_dir/bin"
        echo -e "${GREEN}Old bin directory removed.${NC}"
    fi

    # Clean up PATH
    echo -e "${YELLOW}Cleaning up PATH...${NC}"
    PATH=$(echo $PATH | tr ':' '\n' | grep -v "biomni_tools/bin" | tr '\n' ':' | sed 's/:$//')
    export PATH="$TOOLS_DIR/bin:$PATH"
    echo -e "${GREEN}PATH cleaned up.${NC}"
}

# Create a directory for CLI tools if it doesn't exist
if [ -n "$BIOMNI_TOOLS_DIR" ]; then
    TOOLS_DIR="$BIOMNI_TOOLS_DIR"
else
    TOOLS_DIR="$(pwd)/biomni_tools"
fi

# Clean up any existing installation if it exists
if [ -d "$TOOLS_DIR" ]; then
    echo -e "${YELLOW}Cleaning up existing installation in $TOOLS_DIR...${NC}"
    # Remove bin directory to clean up symlinks
    rm -rf "$TOOLS_DIR/bin"
fi

# Clean up old installations
cleanup_old_installations

# Create fresh directories
mkdir -p "$TOOLS_DIR"
mkdir -p "$TOOLS_DIR/bin"

# Add the tools bin directory to PATH in the current session
# Remove any old paths first to avoid duplicates
PATH=$(echo $PATH | tr ':' '\n' | grep -v "biomni_tools/bin" | tr '\n' ':' | sed 's/:$//')
export PATH="$TOOLS_DIR/bin:$PATH"

# Clear the shell's command hash table to force it to re-search the PATH
hash -r 2>/dev/null || rehash 2>/dev/null || true

# Create a setup_path.sh file in the tools directory
echo "#!/bin/bash" > "$TOOLS_DIR/setup_path.sh"
echo "# Added by biomni setup" >> "$TOOLS_DIR/setup_path.sh"
echo "# Remove any old paths first to avoid duplicates" >> "$TOOLS_DIR/setup_path.sh"
echo "PATH=\$(echo \$PATH | tr ':' '\n' | grep -v \"biomni_tools/bin\" | tr '\n' ':' | sed 's/:$//')" >> "$TOOLS_DIR/setup_path.sh"
echo "export PATH=\"$TOOLS_DIR/bin:\$PATH\"" >> "$TOOLS_DIR/setup_path.sh"
echo "# Clear the shell's command hash table to force it to re-search the PATH" >> "$TOOLS_DIR/setup_path.sh"
echo "hash -r 2>/dev/null || rehash 2>/dev/null || true" >> "$TOOLS_DIR/setup_path.sh"
chmod +x "$TOOLS_DIR/setup_path.sh"

# Config file path
CONFIG_FILE="cli_tools_config.json"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Configuration file $CONFIG_FILE not found.${NC}"
    exit 1
fi

# Function to download and install a tool
install_tool() {
    local tool_name=$1
    local download_url=$2
    local binary_path=$3
    local version_cmd=$4
    local tool_dir_name=$(echo "$tool_name" | tr '[:upper:]' '[:lower:]' | tr ' .' '_')
    local binary_name=$(basename "$binary_path")

    echo -e "\n${BLUE}=== Installing $tool_name ===${NC}"

    # Check if the tool is already installed
    if [ -f "$TOOLS_DIR/bin/$binary_name" ]; then
        echo -e "${GREEN}$tool_name is already installed at $TOOLS_DIR/bin/$binary_name${NC}"
        echo -e "${YELLOW}Testing installation...${NC}"

        if [ -n "$version_cmd" ]; then
            echo -e "${YELLOW}Running version command: $TOOLS_DIR/bin/$binary_name $version_cmd${NC}"
            $TOOLS_DIR/bin/$binary_name $version_cmd
        fi

        echo -e "${GREEN}Skipping download and installation.${NC}"
        return 0
    fi

    # Create directory for the tool
    mkdir -p "$TOOLS_DIR/$tool_dir_name"

    # Special handling for HOMER
    if [ "$tool_name" = "HOMER" ]; then
        echo -e "${YELLOW}Installing HOMER via Perl script...${NC}"

        # Download the configuration script directly to the bin directory
        $DOWNLOADER "$TOOLS_DIR/bin/configureHomer.pl" "$download_url"

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download HOMER configuration script from $download_url${NC}"
            return 1
        fi

        # Make the script executable
        chmod +x "$TOOLS_DIR/bin/configureHomer.pl"

        # Create a HOMER installation directory
        mkdir -p "$TOOLS_DIR/$tool_dir_name/homer"

        # Run the configuration script
        echo -e "${YELLOW}Running HOMER configuration script...${NC}"
        echo -e "${YELLOW}This will install HOMER to $TOOLS_DIR/$tool_dir_name/homer${NC}"

        # Install HOMER with batch mode ("-b" flag for basic installation)
        # Use -local to specify the absolute installation directory
        "$TOOLS_DIR/bin/configureHomer.pl" -install -local "$TOOLS_DIR/$tool_dir_name/homer" -b

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to install HOMER.${NC}"
            return 1
        fi

        # Create symlinks to HOMER binaries
        echo -e "${YELLOW}Creating symlinks to HOMER binaries...${NC}"
        for homer_bin in "$TOOLS_DIR/$tool_dir_name/homer/bin/"*; do
            if [ -f "$homer_bin" ] && [ -x "$homer_bin" ]; then
                ln -sf "$homer_bin" "$TOOLS_DIR/bin/$(basename "$homer_bin")"
            fi
        done

        # Create a sourceable environment setup file
        echo "#!/bin/bash" > "$TOOLS_DIR/$tool_dir_name/homer_env.sh"
        echo "export PATH=\"$TOOLS_DIR/$tool_dir_name/homer/bin:\$PATH\"" >> "$TOOLS_DIR/$tool_dir_name/homer_env.sh"
        echo "export HOMER=\"$TOOLS_DIR/$tool_dir_name/homer\"" >> "$TOOLS_DIR/$tool_dir_name/homer_env.sh"
        chmod +x "$TOOLS_DIR/$tool_dir_name/homer_env.sh"

        # Add the HOMER environment to the global setup_path.sh
        echo "# HOMER environment" >> "$TOOLS_DIR/setup_path.sh"
        echo "export PATH=\"$TOOLS_DIR/$tool_dir_name/homer/bin:\$PATH\"" >> "$TOOLS_DIR/setup_path.sh"
        echo "export HOMER=\"$TOOLS_DIR/$tool_dir_name/homer\"" >> "$TOOLS_DIR/setup_path.sh"

        echo -e "${GREEN}HOMER installed successfully!${NC}"
        echo -e "${YELLOW}To use HOMER, you may need to source the environment file:${NC}"
        echo -e "${GREEN}source $TOOLS_DIR/$tool_dir_name/homer_env.sh${NC}"
        echo -e "${YELLOW}Or you can run configureHomer.pl directly from:${NC}"
        echo -e "${GREEN}$TOOLS_DIR/bin/configureHomer.pl${NC}"
        return 0

    # Special handling for FastTree (requires compilation)
    elif [ "$tool_name" = "FastTree" ]; then
        echo -e "${YELLOW}Installing FastTree from source...${NC}"

        # Download the source
        $DOWNLOADER "$TOOLS_DIR/$tool_dir_name/FastTree.c" "$download_url"

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download FastTree source from $download_url${NC}"
            return 1
        fi

        # Compile FastTree
        echo -e "${YELLOW}Compiling FastTree...${NC}"
        echo -e "${YELLOW}This may take a few minutes.${NC}"

        # Try to compile with SSE support first
        (cd "$TOOLS_DIR/$tool_dir_name" && gcc -O3 -finline-functions -funroll-loops -Wall -o FastTree FastTree.c -lm)

        # If compilation fails, try without SSE support
        if [ $? -ne 0 ]; then
            echo -e "${YELLOW}Compilation with SSE failed, trying without SSE...${NC}"
            (cd "$TOOLS_DIR/$tool_dir_name" && gcc -DNO_SSE -O3 -finline-functions -funroll-loops -Wall -o FastTree FastTree.c -lm)

            if [ $? -ne 0 ]; then
                echo -e "${RED}Failed to compile FastTree.${NC}"
                return 1
            fi
        fi

        # Create symlink
        ln -sf "$TOOLS_DIR/$tool_dir_name/FastTree" "$TOOLS_DIR/bin/FastTree"

        echo -e "${GREEN}FastTree installed successfully!${NC}"

        # Test installation
        if [ -f "$TOOLS_DIR/bin/FastTree" ]; then
            echo -e "${GREEN}FastTree installed successfully!${NC}"
            if [ -n "$version_cmd" ]; then
                echo -e "${YELLOW}Running version command: $TOOLS_DIR/bin/FastTree $version_cmd | head -n 5${NC}"
                "$TOOLS_DIR/bin/FastTree" $version_cmd | head -n 5
            fi
        else
            echo -e "${RED}FastTree installation failed.${NC}"
            echo -e "${YELLOW}Binary not found at: $TOOLS_DIR/bin/FastTree${NC}"
            return 1
        fi

        return 0

    # Special handling for BWA (requires compilation)
    elif [ "$tool_name" = "BWA" ]; then
        echo -e "${YELLOW}Installing BWA...${NC}"

        # Check if git is installed
        if ! command -v git &> /dev/null; then
            echo -e "${RED}Git is not installed. Please install git to continue.${NC}"
            return 1
        fi

        # Clone the repository
        echo -e "${YELLOW}Cloning BWA repository from $download_url...${NC}"
        mkdir -p "$TOOLS_DIR/$tool_dir_name"
        git clone "$download_url" "$TOOLS_DIR/$tool_dir_name"

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to clone BWA repository from $download_url${NC}"
            return 1
        fi

        # Compile BWA
        echo -e "${YELLOW}Compiling BWA...${NC}"
        (cd "$TOOLS_DIR/$tool_dir_name" && make)

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to compile BWA.${NC}"
            return 1
        fi

        # Create symlink
        ln -sf "$TOOLS_DIR/$tool_dir_name/bwa" "$TOOLS_DIR/bin/bwa"

        # Test installation
        if [ -f "$TOOLS_DIR/bin/bwa" ]; then
            echo -e "${GREEN}BWA installed successfully!${NC}"
            echo -e "${YELLOW}Running version check:${NC}"
            "$TOOLS_DIR/bin/bwa" 2>&1 | head -n 3
        else
            echo -e "${RED}BWA installation failed.${NC}"
            echo -e "${YELLOW}Binary not found at: $TOOLS_DIR/bin/bwa${NC}"
            return 1
        fi

        return 0
    fi

    # Download the tool
    echo -e "${YELLOW}Downloading $tool_name from: $download_url${NC}"

    # Determine file extension
    if [[ "$download_url" == *".zip" ]]; then
        # Use verbose output to help diagnose issues
        $DOWNLOADER "$TOOLS_DIR/$tool_dir_name.zip" "$download_url"

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download $tool_name from $download_url${NC}"
            echo -e "${YELLOW}Please check your internet connection and try again.${NC}"
            echo -e "${YELLOW}If the problem persists, the download URL may be incorrect or the server may be down.${NC}"
            return 1
        fi

        echo -e "${YELLOW}Extracting $tool_name...${NC}"
        unzip -q -o "$TOOLS_DIR/$tool_dir_name.zip" -d "$TOOLS_DIR/$tool_dir_name"

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to extract $tool_name.${NC}"
            return 1
        fi

        # Clean up
        rm "$TOOLS_DIR/$tool_dir_name.zip"
    elif [[ "$download_url" == *".tar.gz" ]]; then
        # Use verbose output to help diagnose issues
        $DOWNLOADER "$TOOLS_DIR/$tool_dir_name.tar.gz" "$download_url"

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download $tool_name from $download_url${NC}"
            echo -e "${YELLOW}Please check your internet connection and try again.${NC}"
            echo -e "${YELLOW}If the problem persists, the download URL may be incorrect or the server may be down.${NC}"
            return 1
        fi

        echo -e "${YELLOW}Extracting $tool_name...${NC}"
        mkdir -p "$TOOLS_DIR/$tool_dir_name"
        tar -xzf "$TOOLS_DIR/$tool_dir_name.tar.gz" -C "$TOOLS_DIR/$tool_dir_name" --strip-components=1

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to extract $tool_name.${NC}"
            return 1
        fi

        # Clean up
        rm "$TOOLS_DIR/$tool_dir_name.tar.gz"
    # Handle executable files directly (for MUSCLE)
    elif [[ "$tool_name" = "MUSCLE" ]]; then
        echo -e "${YELLOW}Downloading $tool_name binary...${NC}"
        $DOWNLOADER "$TOOLS_DIR/$tool_dir_name/$binary_name" "$download_url"

        if [ $? -ne 0 ]; then
            echo -e "${RED}Failed to download $tool_name from $download_url${NC}"
            return 1
        fi

        # Make executable
        chmod +x "$TOOLS_DIR/$tool_dir_name/$binary_name"

        # Create symlink
        ln -sf "$TOOLS_DIR/$tool_dir_name/$binary_name" "$TOOLS_DIR/bin/$binary_name"

        # Test installation
        if [ -f "$TOOLS_DIR/bin/$binary_name" ]; then
            echo -e "${GREEN}$tool_name installed successfully!${NC}"
            if [ -n "$version_cmd" ]; then
                echo -e "${YELLOW}Running version command: $TOOLS_DIR/bin/$binary_name $version_cmd${NC}"
                "$TOOLS_DIR/bin/$binary_name" $version_cmd
            fi
        else
            echo -e "${RED}$tool_name installation failed.${NC}"
            echo -e "${YELLOW}Binary not found at: $TOOLS_DIR/bin/$binary_name${NC}"
            return 1
        fi

        return 0
    else
        echo -e "${RED}Unsupported file format for $tool_name.${NC}"
        return 1
    fi

    # Find the binary path
    local full_binary_path=$(find "$TOOLS_DIR/$tool_dir_name" -name "$(basename "$binary_path")" | head -n 1)

    if [ -z "$full_binary_path" ]; then
        echo -e "${RED}Could not find binary for $tool_name.${NC}"
        echo -e "${YELLOW}Looking for binary at expected path: $TOOLS_DIR/$tool_dir_name/$binary_path${NC}"
        full_binary_path="$TOOLS_DIR/$tool_dir_name/$binary_path"
    fi

    # Make the binary executable
    chmod +x "$full_binary_path"

    # Create symlink in bin directory
    ln -sf "$full_binary_path" "$TOOLS_DIR/bin/$binary_name"

    # Test installation
    if [ -f "$TOOLS_DIR/bin/$binary_name" ]; then
        echo -e "${GREEN}$tool_name installed successfully!${NC}"
        if [ -n "$version_cmd" ]; then
            echo -e "${YELLOW}Running version command: $TOOLS_DIR/bin/$binary_name $version_cmd${NC}"
            "$TOOLS_DIR/bin/$binary_name" $version_cmd
        fi
    else
        echo -e "${RED}$tool_name installation failed.${NC}"
        echo -e "${YELLOW}Binary not found at: $TOOLS_DIR/bin/$binary_name${NC}"
        return 1
    fi

    return 0
}

# Function to install a tool from the config
install_tool_from_config() {
    local tool_index=$1
    local auto_install=${2:-0}

    # Get tool information from config
    local tool_name=$(jq -r ".tools[$tool_index].name" "$CONFIG_FILE")
    local tool_desc=$(jq -r ".tools[$tool_index].description" "$CONFIG_FILE")

    # Determine the appropriate download URL based on the system
    local download_url=""
    if [[ "$(uname)" == "Darwin" ]]; then
        if [[ "$(uname -m)" == "arm64" ]]; then
            # macOS M1
            download_url=$(jq -r ".tools[$tool_index].downloads.macos_arm64" "$CONFIG_FILE")
        else
            # macOS Intel
            download_url=$(jq -r ".tools[$tool_index].downloads.macos_intel" "$CONFIG_FILE")
        fi
    else
        # Linux
        download_url=$(jq -r ".tools[$tool_index].downloads.linux" "$CONFIG_FILE")
    fi

    local binary_path=$(jq -r ".tools[$tool_index].binary_path" "$CONFIG_FILE")
    local version_cmd=$(jq -r ".tools[$tool_index].version_command" "$CONFIG_FILE")

    # Install the tool
    echo -e "\n${YELLOW}$tool_name: $tool_desc${NC}"

    # If auto_install is enabled, install without asking
    if [ "$auto_install" -eq 1 ]; then
        install_tool "$tool_name" "$download_url" "$binary_path" "$version_cmd"
        return $?
    fi

    # Otherwise, ask for confirmation
    read -p "Install $tool_name? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_tool "$tool_name" "$download_url" "$binary_path" "$version_cmd"
        return $?
    fi

    return 0
}

# Function to add a new tool to the config
add_new_tool() {
    echo -e "\n${BLUE}=== Add a New Tool ===${NC}"

    read -p "Tool name: " tool_name
    read -p "Tool description: " tool_desc
    read -p "Tool website: " tool_website
    read -p "Linux download URL: " linux_url
    read -p "macOS Intel download URL: " macos_intel_url
    read -p "macOS ARM64 download URL: " macos_arm64_url
    read -p "Binary path (relative to extraction): " binary_path
    read -p "Version command (e.g., --version): " version_cmd

    # Generate function name
    function_name="install_$(echo "$tool_name" | tr '[:upper:]' '[:lower:]' | tr ' .' '_')"

    # Create new tool JSON
    new_tool=$(cat <<EOF
{
  "name": "$tool_name",
  "function_name": "$function_name",
  "description": "$tool_desc",
  "website": "$tool_website",
  "downloads": {
    "linux": "$linux_url",
    "macos_intel": "$macos_intel_url",
    "macos_arm64": "$macos_arm64_url"
  },
  "binary_path": "$binary_path",
  "version_command": "$version_cmd"
}
EOF
)

    # Add to config file
    jq ".tools += [$new_tool]" "$CONFIG_FILE" > "$CONFIG_FILE.tmp" && mv "$CONFIG_FILE.tmp" "$CONFIG_FILE"

    echo -e "${GREEN}Tool added to configuration!${NC}"

    # Ask if user wants to install the tool now
    read -p "Install $tool_name now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        local tool_index=$(jq '.tools | length - 1' "$CONFIG_FILE")
        install_tool_from_config "$tool_index"
    fi
}

# Function to install all tools
install_all_tools() {
    local auto_install=${1:-0}
    local num_tools=$(jq '.tools | length' "$CONFIG_FILE")

    echo -e "\n${BLUE}Installing all command-line tools...${NC}"

    for (( i=0; i<$num_tools; i++ )); do
        install_tool_from_config "$i" "$auto_install"
    done

    return 0
}

# Function to add PATH to shell profile
add_path_to_profile() {
    local force_profile=${1:-""}

    # Try to detect the shell profile file
    local profile_file=""
    local shell_name=$(basename "$SHELL")

    # Create a sourceable file in the tools directory with PATH cleanup
    echo "#!/bin/bash" > "$TOOLS_DIR/setup_path.sh"
    echo "# Added by biomni setup" >> "$TOOLS_DIR/setup_path.sh"
    echo "# Remove any old paths first to avoid duplicates" >> "$TOOLS_DIR/setup_path.sh"
    echo "PATH=\$(echo \$PATH | tr ':' '\n' | grep -v \"biomni_tools/bin\" | tr '\n' ':' | sed 's/:$//')" >> "$TOOLS_DIR/setup_path.sh"
    echo "export PATH=\"$TOOLS_DIR/bin:\$PATH\"" >> "$TOOLS_DIR/setup_path.sh"
    echo "# Clear the shell's command hash table to force it to re-search the PATH" >> "$TOOLS_DIR/setup_path.sh"
    echo "hash -r 2>/dev/null || rehash 2>/dev/null || true" >> "$TOOLS_DIR/setup_path.sh"
    chmod +x "$TOOLS_DIR/setup_path.sh"
    echo -e "${GREEN}Created sourceable file at $TOOLS_DIR/setup_path.sh${NC}"
    echo -e "${YELLOW}You can add this to your PATH by running:${NC}"
    echo -e "${GREEN}source $TOOLS_DIR/setup_path.sh${NC}"

    # If a specific profile is forced, use that
    if [ -n "$force_profile" ]; then
        profile_file="$HOME/$force_profile"
        echo -e "${YELLOW}Using specified profile file: $profile_file${NC}"
    else
        # Auto-detect based on current shell
        case "$shell_name" in
            bash)
                if [ -f "$HOME/.bash_profile" ]; then
                    profile_file="$HOME/.bash_profile"
                elif [ -f "$HOME/.profile" ]; then
                    profile_file="$HOME/.profile"
                elif [ -f "$HOME/.bashrc" ]; then
                    profile_file="$HOME/.bashrc"
                fi
                ;;
            zsh)
                profile_file="$HOME/.zshrc"
                ;;
            fish)
                # Fish has a different configuration structure
                mkdir -p "$HOME/.config/fish"
                profile_file="$HOME/.config/fish/config.fish"
                ;;
            *)
                # For other shells, try common profile files
                if [ -f "$HOME/.profile" ]; then
                    profile_file="$HOME/.profile"
                elif [ -f "$HOME/.bashrc" ]; then
                    profile_file="$HOME/.bashrc"
                elif [ -f "$HOME/.zshrc" ]; then
                    profile_file="$HOME/.zshrc"
                fi
                ;;
        esac
    fi

    # If we found a profile file and it's writable, add the PATH
    if [ -n "$profile_file" ]; then
        # Create the file if it doesn't exist
        if [ ! -f "$profile_file" ]; then
            echo -e "${YELLOW}Creating new profile file: $profile_file${NC}"
            touch "$profile_file"
        fi

        if [ -w "$profile_file" ]; then
            # Check if the PATH is already in the profile
            if ! grep -q "export PATH=\"$TOOLS_DIR/bin:\$PATH\"" "$profile_file"; then
                # Remove any old biomni_tools paths first
                if grep -q "biomni_tools/bin" "$profile_file"; then
                    echo -e "${YELLOW}Removing old biomni_tools paths from $profile_file...${NC}"
                    sed -i '/biomni_tools\/bin/d' "$profile_file"
                fi

                echo "" >> "$profile_file"
                echo "# Added by biomni setup" >> "$profile_file"
                echo "# Remove any old paths first to avoid duplicates" >> "$profile_file"
                echo "PATH=\$(echo \$PATH | tr ':' '\n' | grep -v \"biomni_tools/bin\" | tr '\n' ':' | sed 's/:$//')" >> "$profile_file"

                # Use the appropriate syntax for the shell
                if [ "$shell_name" = "fish" ]; then
                    echo "set -gx PATH $TOOLS_DIR/bin \$PATH" >> "$profile_file"
                else
                    echo "export PATH=\"$TOOLS_DIR/bin:\$PATH\"" >> "$profile_file"
                fi

                echo -e "${GREEN}Added tools directory to PATH in $profile_file${NC}"
                echo -e "${YELLOW}Note: You may need to restart your shell or run 'source $profile_file' for changes to take effect.${NC}"
            else
                echo -e "${GREEN}PATH already configured in $profile_file${NC}"
            fi
        else
            echo -e "${RED}Profile file $profile_file is not writable.${NC}"
            echo -e "${YELLOW}Please add the following line to your shell profile manually:${NC}"

            if [ "$shell_name" = "fish" ]; then
                echo -e "${GREEN}set -gx PATH $TOOLS_DIR/bin \$PATH${NC}"
            else
                echo -e "${GREEN}export PATH=\"$TOOLS_DIR/bin:\$PATH\"${NC}"
            fi

            echo -e "${YELLOW}Or source the setup file we created:${NC}"
            echo -e "${GREEN}source $TOOLS_DIR/setup_path.sh${NC}"
        fi
    else
        # If we couldn't find a profile file, just print instructions
        echo -e "${YELLOW}Could not determine appropriate shell profile file.${NC}"
        echo -e "${YELLOW}Please add the following line to your shell profile manually:${NC}"

        if [ "$shell_name" = "fish" ]; then
            echo -e "${GREEN}set -gx PATH $TOOLS_DIR/bin \$PATH${NC}"
        else
            echo -e "${GREEN}export PATH=\"$TOOLS_DIR/bin:\$PATH\"${NC}"
        fi

        echo -e "${YELLOW}Or source the setup file we created:${NC}"
        echo -e "${GREEN}source $TOOLS_DIR/setup_path.sh${NC}"
    fi

    # Always export PATH for the current session
    # Remove any old paths first to avoid duplicates
    PATH=$(echo $PATH | tr ':' '\n' | grep -v "biomni_tools/bin" | tr '\n' ':' | sed 's/:$//')
    export PATH="$TOOLS_DIR/bin:$PATH"

    # Clear the shell's command hash table to force it to re-search the PATH
    hash -r 2>/dev/null || rehash 2>/dev/null || true

    # Create a test script to verify the tools are in the PATH
    echo "#!/bin/bash" > "$TOOLS_DIR/test_tools.sh"
    echo "echo \"Testing if tools are in the PATH...\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "echo \"Current PATH: \$PATH\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "echo \"\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "echo \"Looking for tools in: $TOOLS_DIR/bin\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "ls -la \"$TOOLS_DIR/bin\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "echo \"\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "echo \"Checking for path caching issues...\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "for tool in \$(ls \"$TOOLS_DIR/bin\"); do" >> "$TOOLS_DIR/test_tools.sh"
    echo "  which \$tool 2>/dev/null | grep -q \"/afs/cs.stanford.edu\" && {" >> "$TOOLS_DIR/test_tools.sh"
    echo "    echo \"WARNING: \$tool is still pointing to the old AFS location!\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "    echo \"Run 'hash -r' (bash) or 'rehash' (zsh) to clear the command cache.\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "    break" >> "$TOOLS_DIR/test_tools.sh"
    echo "  }" >> "$TOOLS_DIR/test_tools.sh"
    echo "done" >> "$TOOLS_DIR/test_tools.sh"
    echo "echo \"\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "for tool in \$(ls \"$TOOLS_DIR/bin\"); do" >> "$TOOLS_DIR/test_tools.sh"
    echo "  if command -v \$tool &> /dev/null; then" >> "$TOOLS_DIR/test_tools.sh"
    echo "    echo \"\$tool: \$(which \$tool)\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "  else" >> "$TOOLS_DIR/test_tools.sh"
    echo "    echo \"\$tool: NOT FOUND IN PATH\"" >> "$TOOLS_DIR/test_tools.sh"
    echo "  fi" >> "$TOOLS_DIR/test_tools.sh"
    echo "done" >> "$TOOLS_DIR/test_tools.sh"
    chmod +x "$TOOLS_DIR/test_tools.sh"

    echo -e "${YELLOW}Created test script at $TOOLS_DIR/test_tools.sh${NC}"
    echo -e "${YELLOW}You can run it to verify the tools are in your PATH:${NC}"
    echo -e "${GREEN}$TOOLS_DIR/test_tools.sh${NC}"

    # Create a quick fix script for path caching issues
    echo "#!/bin/bash" > "$TOOLS_DIR/fix_path.sh"
    echo "# Script to fix path caching issues" >> "$TOOLS_DIR/fix_path.sh"
    echo "echo \"Fixing path caching issues...\"" >> "$TOOLS_DIR/fix_path.sh"
    echo "# Remove any old paths first to avoid duplicates" >> "$TOOLS_DIR/fix_path.sh"
    echo "PATH=\$(echo \$PATH | tr ':' '\n' | grep -v \"biomni_tools/bin\" | tr '\n' ':' | sed 's/:$//')" >> "$TOOLS_DIR/fix_path.sh"
    echo "export PATH=\"$TOOLS_DIR/bin:\$PATH\"" >> "$TOOLS_DIR/fix_path.sh"
    echo "# Clear the shell's command hash table to force it to re-search the PATH" >> "$TOOLS_DIR/fix_path.sh"
    echo "hash -r 2>/dev/null || rehash 2>/dev/null || true" >> "$TOOLS_DIR/fix_path.sh"
    echo "echo \"Path fixed. Try running your command again.\"" >> "$TOOLS_DIR/fix_path.sh"
    chmod +x "$TOOLS_DIR/fix_path.sh"

    echo -e "${YELLOW}Created fix script at $TOOLS_DIR/fix_path.sh${NC}"
    echo -e "${YELLOW}If you encounter 'No such file or directory' errors, run:${NC}"
    echo -e "${GREEN}source $TOOLS_DIR/fix_path.sh${NC}"
}

# Function to verify installation
verify_installation() {
    echo -e "\n${BLUE}=== Verifying Installation ===${NC}"
    echo -e "${YELLOW}Tools directory: $TOOLS_DIR${NC}"
    echo -e "${YELLOW}Bin directory: $TOOLS_DIR/bin${NC}"

    # Check if bin directory exists
    if [ ! -d "$TOOLS_DIR/bin" ]; then
        echo -e "${RED}Bin directory does not exist!${NC}"
        return 1
    fi

    # Check if there are any tools in the bin directory
    local tool_count=$(ls -1 "$TOOLS_DIR/bin" 2>/dev/null | wc -l)
    if [ "$tool_count" -eq 0 ]; then
        echo -e "${RED}No tools found in bin directory!${NC}"
        return 1
    fi

    echo -e "${GREEN}Found $tool_count tools in bin directory.${NC}"

    # List all tools
    echo -e "${YELLOW}Installed tools:${NC}"
    ls -la "$TOOLS_DIR/bin"

    # Check if tools are in PATH
    echo -e "\n${YELLOW}Checking if tools are in PATH...${NC}"
    echo -e "${YELLOW}Current PATH: $PATH${NC}"

    # Check if TOOLS_DIR/bin is in PATH
    if [[ "$PATH" == *"$TOOLS_DIR/bin"* ]]; then
        echo -e "${GREEN}Tools directory is in PATH.${NC}"
    else
        echo -e "${RED}Tools directory is NOT in PATH!${NC}"
        echo -e "${YELLOW}Please run: source $TOOLS_DIR/setup_path.sh${NC}"
        return 1
    fi

    # Check if each tool is accessible
    echo -e "\n${YELLOW}Testing tool accessibility:${NC}"
    for tool in $(ls "$TOOLS_DIR/bin"); do
        if command -v "$tool" &> /dev/null; then
            echo -e "${GREEN}$tool: $(which $tool)${NC}"
        else
            echo -e "${RED}$tool: NOT FOUND IN PATH${NC}"
        fi
    done

    echo -e "\n${GREEN}Installation verification completed.${NC}"
    echo -e "${YELLOW}If you encounter any issues, please run:${NC}"
    echo -e "${GREEN}source $TOOLS_DIR/setup_path.sh${NC}"
    echo -e "${YELLOW}And then run the test script:${NC}"
    echo -e "${GREEN}$TOOLS_DIR/test_tools.sh${NC}"

    echo -e "\n${YELLOW}IMPORTANT: If you see 'No such file or directory' errors when running tools,${NC}"
    echo -e "${YELLOW}your shell may be using cached paths to old tool locations.${NC}"
    echo -e "${YELLOW}To fix this, run:${NC}"
    echo -e "${GREEN}source $TOOLS_DIR/fix_path.sh${NC}"
    echo -e "${YELLOW}Or run one of these commands:${NC}"
    echo -e "${GREEN}hash -r${NC} (for bash/sh)"
    echo -e "${GREEN}rehash${NC} (for zsh/csh)"
    echo -e "${YELLOW}Or simply start a new terminal session.${NC}"

    return 0
}

# Main installation function
install_cli_tools() {
    echo -e "${YELLOW}=== Installing Command-Line Bioinformatics Tools ===${NC}"

    # Try to add the tools bin directory to PATH
    add_path_to_profile

    # Check if auto-install mode is enabled
    if [ -n "$BIOMNI_AUTO_INSTALL" ]; then
        install_all_tools 1

        echo -e "\n${GREEN}CLI tools installation completed!${NC}"
        echo -e "The tools are installed in: ${YELLOW}$TOOLS_DIR${NC}"
        echo -e "Binaries are symlinked in: ${YELLOW}$TOOLS_DIR/bin${NC}"
        echo -e "Tools have been added to your PATH for the current session."

        # Verify installation
        verify_installation

        return 0
    fi

    # Get the number of tools in the config
    local num_tools=$(jq '.tools | length' "$CONFIG_FILE")

    # Display menu
    echo -e "\n${BLUE}Available tools:${NC}"
    for (( i=0; i<$num_tools; i++ )); do
        local tool_name=$(jq -r ".tools[$i].name" "$CONFIG_FILE")
        local tool_desc=$(jq -r ".tools[$i].description" "$CONFIG_FILE")
        echo -e "${YELLOW}$((i+1)). $tool_name${NC} - $tool_desc"
    done
    echo -e "${YELLOW}$((num_tools+1)). Add a new tool${NC}"
    echo -e "${YELLOW}$((num_tools+2)). Install all tools${NC}"
    echo -e "${YELLOW}$((num_tools+3)). Add tools directory to shell profile${NC}"
    echo -e "${YELLOW}0. Exit${NC}"

    # Get user choice
    read -p "Enter your choice (0-$((num_tools+3))): " choice

    if [[ "$choice" -eq 0 ]]; then
        echo -e "${YELLOW}Exiting...${NC}"
        return 0
    elif [[ "$choice" -eq $((num_tools+1)) ]]; then
        add_new_tool
    elif [[ "$choice" -eq $((num_tools+2)) ]]; then
        # Install all tools
        install_all_tools
    elif [[ "$choice" -eq $((num_tools+3)) ]]; then
        # Add to shell profile
        echo -e "\n${BLUE}=== Adding to Shell Profile ===${NC}"
        echo -e "1. Auto-detect profile (recommended)"
        echo -e "2. Specify profile file"
        read -p "Enter your choice (1-2): " profile_choice

        if [[ "$profile_choice" -eq 1 ]]; then
            add_path_to_profile
        elif [[ "$profile_choice" -eq 2 ]]; then
            echo -e "\nCommon profile files:"
            echo -e "- .bash_profile (for Bash on macOS/login shells)"
            echo -e "- .bashrc (for Bash on Linux/non-login shells)"
            echo -e "- .zshrc (for Zsh)"
            echo -e "- .config/fish/config.fish (for Fish)"
            echo -e "- .profile (generic)"
            read -p "Enter profile filename (without $HOME/ prefix): " profile_file
            add_path_to_profile "$profile_file"
        else
            echo -e "${RED}Invalid choice.${NC}"
        fi
    elif [[ "$choice" -ge 1 ]] && [[ "$choice" -le "$num_tools" ]]; then
        # Install selected tool
        install_tool_from_config "$((choice-1))"
    else
        echo -e "${RED}Invalid choice.${NC}"
        return 1
    fi

    echo -e "\n${GREEN}CLI tools installation completed!${NC}"
    echo -e "The tools are installed in: ${YELLOW}$TOOLS_DIR${NC}"
    echo -e "Binaries are symlinked in: ${YELLOW}$TOOLS_DIR/bin${NC}"
    echo -e "Tools have been added to your PATH for the current session."

    # Ask if user wants to install more tools or configure PATH
    read -p "Install more tools or configure PATH? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_cli_tools
    fi
}

# Function to display help message
show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Install command-line bioinformatics tools for BioAgentOS."
    echo
    echo "Options:"
    echo "  --auto                 Automatically install all tools without prompting"
    echo "  --profile PROFILE      Add tools directory to the specified shell profile file"
    echo "                         Example: $0 --profile .zshrc"
    echo "  --help                 Display this help message and exit"
    echo
    echo "Without options, the script runs in interactive mode."
}

# Check command-line arguments
if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    show_help
    exit 0
elif [ "$1" = "--auto" ]; then
    # Set BIOMNI_AUTO_INSTALL if it's not already set
    export BIOMNI_AUTO_INSTALL=1

    # Run in auto-install mode without showing the menu
    install_all_tools 1

    # Add to profile automatically
    add_path_to_profile

    # Verify installation
    verify_installation

    echo -e "\n${GREEN}CLI tools installation completed!${NC}"
    echo -e "The tools are installed in: ${YELLOW}$TOOLS_DIR${NC}"
    echo -e "Binaries are symlinked in: ${YELLOW}$TOOLS_DIR/bin${NC}"
    echo -e "Tools have been added to your PATH for the current session."

    exit 0
elif [ "$1" = "--profile" ] && [ -n "$2" ]; then
    # Add to a specific profile file
    add_path_to_profile "$2"
    exit 0
elif [ -n "$BIOMNI_AUTO_INSTALL" ]; then
    # Run in auto-install mode without showing the menu
    install_all_tools 1

    # Add to profile automatically
    add_path_to_profile

    # Verify installation
    verify_installation

    echo -e "\n${GREEN}CLI tools installation completed!${NC}"
    echo -e "The tools are installed in: ${YELLOW}$TOOLS_DIR${NC}"
    echo -e "Binaries are symlinked in: ${YELLOW}$TOOLS_DIR/bin${NC}"
    echo -e "Tools have been added to your PATH for the current session."

    exit 0
elif [ -n "$1" ]; then
    echo -e "${RED}Unknown option: $1${NC}"
    show_help
    exit 1
else
    # Run the interactive installation
    install_cli_tools
fi
