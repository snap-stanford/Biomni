#!/usr/bin/env Rscript

# Script to install R packages for Biomni HITS Docker image

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

# Install hgu133plus2.db
cat("\nInstalling hgu133plus2.db and dependencies...\n")
if (!require("hgu133plus2.db", character.only = TRUE, quietly = TRUE)) {
  BiocManager::install("hgu133plus2.db", dependencies = TRUE, update = FALSE, ask = FALSE)
  if (require("hgu133plus2.db", character.only = TRUE, quietly = TRUE)) {
    cat("✓ Successfully installed hgu133plus2.db\n")
  } else {
    cat("✗ Failed to install hgu133plus2.db\n")
  }
} else {
  cat("✓ Package hgu133plus2.db is already installed\n")
}

# Install DESeq2 with dependencies
cat("\nInstalling DESeq2 and dependencies...\n")
if (!require("DESeq2", quietly = TRUE)) {
  BiocManager::install("DESeq2", dependencies = TRUE, update = FALSE, ask = FALSE)
}

cat("\nInstalling edgeR and dependencies...\n")
if (!require("edgeR", quietly = TRUE)) {
  BiocManager::install("edgeR", dependencies = TRUE, update = FALSE, ask = FALSE)
}

cat("\nInstalling glmnet and dependencies via BiocManager...\n")
if (!require("glmnet", quietly = TRUE)) {
  BiocManager::install("glmnet", dependencies = TRUE, update = FALSE, ask = FALSE)
}

cat("\nInstalling survival and dependencies via BiocManager...\n")
if (!require("survival", quietly = TRUE)) {
  BiocManager::install("survival", dependencies = TRUE, update = FALSE, ask = FALSE)
  BiocManager::install("survivalROC", dependencies = TRUE, update = FALSE, ask = FALSE)
  BiocManager::install("survminer", dependencies = TRUE, update = FALSE, ask = FALSE)
}

cat("\nInstalling clusterProfiler and dependencies...\n")
if (!require("clusterProfiler", quietly = TRUE)) {
  BiocManager::install("clusterProfiler", dependencies = TRUE, update = FALSE, ask = FALSE)
}

cat ("Installing GEOquery and dependencies...\n")
if (!require("GEOquery", quietly = TRUE)) {
  BiocManager::install("GEOquery", dependencies = TRUE, update = FALSE, ask = FALSE)
}

cat("\nInstalling org.Hs.eg.db and dependencies...\n")
BiocManager::install("org.Hs.eg.db")

install.packages(c("survival"))
install.packages(c("timeROC"))

cat("\nPackage installation completed!\n")
cat("If you still encounter issues with specific packages, please install them manually.\n")

