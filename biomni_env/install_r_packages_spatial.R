#!/usr/bin/env Rscript

# Install R packages required by Biomni spatial transcriptomics tools:
#   - SPOTlight (spot deconvolution)
#   - CellChat v2 (cell-cell communication)
#   - Seurat, SingleCellExperiment, SpatialExperiment (object containers)
#   - scater, scran (marker genes / HVGs for SPOTlight)
#
# Usage: Rscript install_r_packages_spatial.R
# Requires a working compiler toolchain (gcc/g++) in the R environment.

options(repos = c(CRAN = "https://cloud.r-project.org/"))

install_if_missing <- function(pkg, bioc = FALSE) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    cat(sprintf("  OK  %s (already installed)\n", pkg))
    return(invisible(TRUE))
  }
  cat(sprintf("  Installing %s ...\n", pkg))
  ok <- tryCatch({
    if (bioc) {
      if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")
      BiocManager::install(pkg, update = FALSE, ask = FALSE, dependencies = TRUE)
    } else {
      install.packages(pkg, dependencies = TRUE)
    }
    requireNamespace(pkg, quietly = TRUE)
  }, error = function(e) {
    cat(sprintf("  ERROR installing %s: %s\n", pkg, conditionMessage(e)))
    FALSE
  })
  if (ok) cat(sprintf("  OK  %s\n", pkg)) else cat(sprintf("  FAILED  %s\n", pkg))
  invisible(ok)
}

cat("== Installing BiocManager ==\n")
if (!requireNamespace("BiocManager", quietly = TRUE)) install.packages("BiocManager")

cat("\n== CRAN packages ==\n")
cran_pkgs <- c(
  "Seurat",            # single-cell object / CellChat input
  "Matrix",
  "Rcpp",
  "future",
  "future.apply",
  "pbapply",
  "irlba",
  "NMF",
  "ggalluvial",
  "stringr",
  "svglite",
  "ggrepel",
  "circlize",
  "RColorBrewer",
  "cowplot",
  "RSpectra",
  "reticulate",
  "scales",
  "sna",
  "reshape2",
  "FNN",
  "shape",
  "magrittr",
  "patchwork",
  "colorspace",
  "plyr",
  "ggpubr",
  "ggnetwork",
  "ggraph",
  "collapse",
  "wordcloud",
  "presto",
  "tidyverse",
  "tidyr",
  "rlang",
  "purrr"
)
for (p in cran_pkgs) install_if_missing(p)

cat("\n== Bioconductor packages ==\n")
bioc_pkgs <- c(
  "SingleCellExperiment",
  "SpatialExperiment",
  "scater",
  "scran",
  "ComplexHeatmap",
  "BiocGenerics"
)
for (p in bioc_pkgs) install_if_missing(p, bioc = TRUE)

cat("\n== SPOTlight (Bioconductor) ==\n")
install_if_missing("SPOTlight", bioc = TRUE)

cat("\n== CellChat v2 (from local source or GitHub) ==\n")
if (!requireNamespace("presto", quietly = TRUE)) {
  # presto: fast Wilcoxon used by CellChat::identifyOverExpressedGenes (official default)
  cat("Installing presto from GitHub...\n")
  if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
  remotes::install_github("immunogenomics/presto", upgrade = "never")
}
if (!requireNamespace("CellChat", quietly = TRUE)) {
  # Try local source first (biomni repo tools_code/CellChat), then GitHub
  local_src <- "tools_code/CellChat"
  if (dir.exists(local_src)) {
    if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
    remotes::install_local(local_src, dependencies = FALSE, upgrade = "never")
  } else {
    if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
    remotes::install_github("jinworks/CellChat", dependencies = FALSE, upgrade = "never")
  }
}
cat("  CellChat installed:", requireNamespace("CellChat", quietly = TRUE), "\n")

cat("\n== Verification ==\n")
need <- c("SPOTlight", "CellChat", "Seurat", "SingleCellExperiment",
          "SpatialExperiment", "scater", "scran", "Matrix")
for (p in need) {
  cat(sprintf("  %s: %s\n", p, requireNamespace(p, quietly = TRUE)))
}

cat("\nDone. If CellChat failed, install it manually:\n")
cat("  remotes::install_github('jinworks/CellChat')\n")
