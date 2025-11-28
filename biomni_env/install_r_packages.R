#!/usr/bin/env Rscript

# Script to install R packages that might be missing from the conda environment
# Run this script after creating the conda environment if you encounter issues with R packages

# Set repository
options(repos = c(CRAN = "https://cran.rstudio.com/"))

# Function to install a package if it's not already installed
install_if_missing <- function(package_name, bioconductor = FALSE) {
  if (!require(package_name, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("Installing package: %s\n", package_name))

    if (bioconductor) {
      if (!require("BiocManager", quietly = TRUE)) {
        install.packages("BiocManager", dependencies = TRUE)
      }
      BiocManager::install(package_name, update = FALSE, ask = FALSE, dependencies = TRUE)
    } else {
      install.packages(package_name, dependencies = TRUE)
    }

    # Check if installation was successful
    if (require(package_name, character.only = TRUE, quietly = TRUE)) {
      cat(sprintf("✓ Successfully installed %s\n", package_name))
    } else {
      cat(sprintf("✗ Failed to install %s\n", package_name))
    }
  } else {
    cat(sprintf("✓ Package %s is already installed\n", package_name))
  }
}

# Install BiocManager first with dependencies
cat("Installing BiocManager...\n")
if (!require("BiocManager", quietly = TRUE)) {
  install.packages("BiocManager", dependencies = TRUE)
}

# Make sure BiocManager is up to date
BiocManager::install(version = BiocManager::version(), update = TRUE, ask = FALSE)

# Install system dependencies for Bioconductor packages
cat("\nInstalling system dependencies for Bioconductor packages...\n")

# Install CRAN packages
cran_packages <- c(
  "ggplot2",
  "lme4",
  "dplyr",
  "tidyr",
  "readr",
  "stringr",
  "Matrix",  # Required for Seurat
  "Rcpp",    # Required for many packages
  "devtools", # Useful for package installation
  "remotes",  # Required for GitHub installations
  "harmony"
)

# Install Bioconductor packages
bioc_packages <- c(
  "DESeq2",
  "dada2",
  "xcms",
  "flowCore",
  "edgeR",
  "limma"
)

# Install CRAN packages
cat("\nInstalling CRAN packages...\n")
for (pkg in cran_packages) {
  install_if_missing(pkg)
}

# Install Bioconductor packages one by one with specific handling
cat("\nInstalling Bioconductor packages...\n")

# Install DESeq2 with dependencies
cat("\nInstalling DESeq2 and dependencies...\n")
if (!require("DESeq2", quietly = TRUE)) {
  BiocManager::install("DESeq2", dependencies = TRUE, update = FALSE, ask = FALSE)
}

# Install WGCNA (from CRAN, not Bioconductor)
# Following official instructions from https://cran.r-project.org/web/packages/WGCNA/index.html
cat("\nInstalling WGCNA and dependencies...\n")
if (!require("WGCNA", quietly = TRUE)) {
  # Install WGCNA dependencies first
  wgcna_deps <- c("dynamicTreeCut", "fastcluster", "matrixStats", "Hmisc", "foreach", "doParallel")
  for (dep in wgcna_deps) {
    install_if_missing(dep)
  }

  # Install Bioconductor dependencies for WGCNA
  bioc_deps <- c("impute", "preprocessCore", "GO.db", "AnnotationDbi")
  for (dep in bioc_deps) {
    if (!require(dep, character.only = TRUE, quietly = TRUE)) {
      BiocManager::install(dep, update = FALSE, ask = FALSE)
    }
  }

  # Install WGCNA from CRAN
  install.packages("WGCNA", dependencies = TRUE)
}

# Install clusterProfiler
cat("\nInstalling clusterProfiler and dependencies...\n")
if (!require("clusterProfiler", quietly = TRUE)) {
  BiocManager::install("clusterProfiler", dependencies = TRUE, update = FALSE, ask = FALSE)
}

# Install remaining Bioconductor packages
remaining_bioc <- setdiff(bioc_packages, c("DESeq2", "clusterProfiler"))
for (pkg in remaining_bioc) {
  install_if_missing(pkg, bioconductor = TRUE)
}

# Verify installations
cat("\nVerifying installations...\n")
all_packages <- c(cran_packages, bioc_packages, "WGCNA")
for (pkg in unique(all_packages)) {
  if (require(pkg, character.only = TRUE, quietly = TRUE)) {
    cat(sprintf("✓ Package %s is successfully installed\n", pkg))
  } else {
    cat(sprintf("✗ Package %s is NOT installed\n", pkg))
  }
}

cat("\nPackage installation completed!\n")
cat("If you still encounter issues with specific packages, please install them manually.\n")
