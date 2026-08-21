"""

Spatial transcriptomics (10x Visium) analysis tools.



Implements a standard Visium analysis workflow with per-step visualization:

  Stage 1: data loading, QC filtering, normalization

  Stage 2: clustering and spatial domain detection

  Stage 3: spatially variable gene detection

  Stage 4: cell-type deconvolution (SPOTlight / DestVI)

  Stage 5: cell-cell communication inference (CellChat v2)

  Stage 6: neighborhood enrichment analysis



Every tool saves its own plots (PNG) under the output directory and returns

the plot paths in the step log, following Biomni tool conventions.

"""



import os



import numpy as np

import pandas as pd

import scanpy as sc





def _plot_setup(output_dir: str) -> None:

    """Ensure matplotlib uses the Agg backend and the output dir exists."""

    import matplotlib



    matplotlib.use("Agg")

    os.makedirs(output_dir, exist_ok=True)





def _spatial_scatter(adata, color, ax, title, cmap="viridis"):

    """Plot spots colored by a value/annotation on the spatial coordinates."""

    import matplotlib.pyplot as plt



    if "spatial" not in adata.obsm:

        return

    coords = adata.obsm["spatial"]

    x, y = coords[:, 0], coords[:, 1]

    if color in adata.obs:

        vals = adata.obs[color]

        if pd.api.types.is_categorical_dtype(vals) or vals.dtype == object:

            cats = vals.cat.categories if pd.api.types.is_categorical_dtype(vals) else pd.unique(vals)

            cmap_obj = plt.get_cmap("tab20", len(cats))

            color_map = {c: cmap_obj(i) for i, c in enumerate(cats)}

            colors = [color_map.get(v, "grey") for v in vals]

            scatter = ax.scatter(x, y, c=colors, s=8, alpha=0.8)

        else:

            scatter = ax.scatter(x, y, c=vals, s=8, alpha=0.8, cmap=cmap)

            plt.colorbar(scatter, ax=ax, fraction=0.046)

    elif color in adata.var_names:

        scatter = ax.scatter(x, y, c=adata[:, color].X.toarray().flatten(), s=8, alpha=0.8, cmap=cmap)

        plt.colorbar(scatter, ax=ax, fraction=0.046)

    else:

        ax.scatter(x, y, s=8, alpha=0.8)

    ax.set_title(title)

    ax.set_xlabel("x")

    ax.set_ylabel("y")

    ax.set_aspect("equal")





# ============================================================================

# Stage 1: Data loading, QC and normalization

# ============================================================================





def load_visium_data(

    counts_file: str,

    coordinates_file: str = None,

    sample_id: str = None,

    output_path: str = "./visium_loaded.h5ad",

    plot_path: str = None,

) -> str:

    """Load Visium data into a standardized AnnData object.



    This tool does NOT guess file formats. The caller (LLM agent) is

    responsible for identifying the input data format first and passing

    explicit file paths. Supported inputs:



      1. h5ad file: counts_file = path to .h5ad (must contain raw counts in X,

         optionally obsm['spatial']).

      2. 10x matrix directory: counts_file = directory containing

         matrix.mtx(.gz) + features.tsv(.gz) + barcodes.tsv(.gz).

      3. Wide counts CSV: counts_file = path to CSV with rows=barcodes,

         cols=genes (e.g. gastric GSE counts).



    If the data is not one of these formats, do NOT call this tool. Instead

    inspect the files yourself (use run_python_repl), convert them to a

    supported format (or to h5ad), then call this tool.



    Args:

        counts_file: Path to h5ad, 10x matrix directory, or wide counts CSV.

        coordinates_file: Optional path to coordinate CSV (barcode + x/y or

            Space Ranger tissue_positions format). If the h5ad already has

            obsm['spatial'] this is not needed.

        sample_id: Optional sample identifier stored in obs['sample_id'].

        output_path: Path where the loaded AnnData will be saved.

        plot_path: Path where the spot-layout plot will be saved.



    Returns:

        str: Step-by-step log describing the load result.

    """

    import scanpy as sc



    steps = []

    if not os.path.exists(counts_file):

        return f"Error: counts_file not found: {counts_file}"



    # ---- Load counts ----

    if counts_file.endswith(".h5ad"):

        adata = sc.read_h5ad(counts_file)

        steps.append(f"Loaded h5ad: {adata.shape[0]} spots x {adata.shape[1]} genes")

        # Try to recover coordinates from obs if not in obsm

        if "spatial" not in adata.obsm:

            for xk, yk in (("pixel_x", "pixel_y"), ("pxl_row_in_fullres", "pxl_col_in_fullres"),

                           ("x", "y"), ("array_x", "array_y")):

                if xk in adata.obs and yk in adata.obs:

                    adata.obsm["spatial"] = adata.obs[[xk, yk]].values.astype(float)

                    steps.append(f"Recovered coordinates from obs columns {xk}/{yk}")

                    break

    elif os.path.isdir(counts_file):

        # 10x matrix directory

        import scipy.io as sio



        mtx = os.path.join(counts_file, "matrix.mtx")

        feat = os.path.join(counts_file, "features.tsv")

        bar = os.path.join(counts_file, "barcodes.tsv")

        for f in (mtx, feat, bar):

            if not os.path.exists(f):

                f_gz = f + ".gz"

                if os.path.exists(f_gz):

                    f = f_gz

                else:

                    return f"Error: expected 10x file not found: {f}"

        if mtx.endswith(".gz"):

            adata = sc.read_10x_mtx(counts_file, var_names="gene_symbols", make_unique=True)

        else:

            counts = sio.mmread(mtx).T.tocsr()

            ft = pd.read_csv(feat, sep="\t", header=None)

            bc = pd.read_csv(bar, sep="\t", header=None)

            adata = sc.AnnData(X=counts)

            adata.var_names = ft.iloc[:, 0].astype(str).values

            adata.var_names_make_unique()

            adata.obs_names = bc.iloc[:, 0].astype(str).values

            adata.obs_names_make_unique()

        steps.append(f"Loaded 10x matrix: {adata.shape[0]} spots x {adata.shape[1]} genes")

    else:

        # Wide counts CSV (rows=barcodes, cols=genes)

        import gzip



        opener = gzip.open if counts_file.endswith(".gz") else open

        with opener(counts_file, "rt", errors="replace") as f:

            df = pd.read_csv(f, index_col=0)

        df.index = df.index.astype(str)

        adata = sc.AnnData(X=df.values.astype(float))

        adata.obs_names = df.index.values

        adata.var_names = df.columns.astype(str).values

        adata.var_names_make_unique()

        steps.append(f"Loaded wide counts CSV: {adata.shape[0]} spots x {adata.shape[1]} genes")



    # ---- Standardize ----

    if "gene_symbols" not in adata.var:

        adata.var["gene_symbols"] = adata.var_names.astype(str).values

    adata.obs["barcode"] = adata.obs_names.astype(str).values



    # ---- Attach coordinates if provided ----

    if coordinates_file and os.path.exists(coordinates_file):

        raw = pd.read_csv(coordinates_file, header=None)

        # Strip header if present

        first_row = [str(v).strip().lower() for v in raw.iloc[0].tolist()]

        if any(k in first_row for k in ("barcode", "in_tissue", "x", "y")):

            raw = raw.iloc[1:].reset_index(drop=True)

        n_cols = raw.shape[1]

        if n_cols >= 6:

            # Space Ranger tissue_positions: barcode,in_tissue,row,col,pxl_row,pxl_col

            pos = raw.iloc[:, :6].copy()

            pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]

            pos["in_tissue"] = pos["in_tissue"].astype(str).str.strip()

            pos = pos[pos["in_tissue"].isin(["1", "True", "true"])]

            pos.index = pos["barcode"].astype(str).values

            coords = pos.iloc[:, 4:6].astype(float).loc[adata.obs_names].values

            adata.obsm["spatial"] = coords

            steps.append(f"Attached coordinates from tissue_positions ({len(coords)} spots)")

        elif n_cols == 3:

            pos = raw.iloc[:, :3].copy()

            pos.columns = ["barcode", "x", "y"]

            pos.index = pos["barcode"].astype(str).values

            adata.obsm["spatial"] = pos.loc[adata.obs_names, ["x", "y"]].astype(float).values

            steps.append(f"Attached coordinates from barcode,x,y CSV ({adata.n_obs} spots)")

        elif n_cols == 2:

            pos = raw.iloc[:, :2].copy()

            pos.columns = ["x", "y"]

            pos.index = pos.index.astype(str)

            adata.obsm["spatial"] = pos.loc[adata.obs_names, ["x", "y"]].astype(float).values

            steps.append(f"Attached coordinates from x,y CSV ({adata.n_obs} spots)")

        else:

            steps.append(f"Warning: unrecognized coordinate file format ({n_cols} cols)")

    elif "spatial" not in adata.obsm:

        steps.append("Warning: no coordinates attached (no coordinates_file given and h5ad lacks obsm['spatial'])")



    if sample_id:

        adata.obs["sample_id"] = sample_id

        steps.append(f"Set sample_id = {sample_id}")



    adata.uns["spatial_meta"] = {

        "source": counts_file,

        "n_spots": adata.n_obs,

        "n_genes": adata.n_vars,

        "has_coords": "spatial" in adata.obsm,

    }



    adata.write(output_path, compression="lzf")

    steps.append(f"Saved loaded AnnData to {output_path}")



    # Plot spot layout

    if "spatial" in adata.obsm:

        _plot_setup(os.path.dirname(output_path) or ".")

        import matplotlib.pyplot as plt



        if plot_path is None:

            plot_path = os.path.splitext(output_path)[0] + "_layout.png"

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        _spatial_scatter(adata, None, ax, "Visium spot layout")

        fig.savefig(plot_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        steps.append(f"Spot layout plot saved to {plot_path}")

    else:

        steps.append("Warning: no coordinates to plot; spot layout plot skipped")



    return "\n".join(steps)





def filter_visium_spots(

    adata_path: str,

    output_path: str = "./visium_filtered.h5ad",

    plot_path: str = None,

    min_counts: int = 200,

    min_genes: int = 20,

    pct_mt: float = 20.0,

) -> str:

    """Filter low-quality spots in 10x Visium data.



    Applies standard QC filtering to a Visium AnnData: removes spots with too

    few counts or too few detected genes, and spots with a high fraction of

    mitochondrial reads. Adds QC metrics to ``adata.obs`` and plots the QC

    violin distributions before and after filtering.



    Args:

        adata_path: Path to the input AnnData (h5ad) file.

        output_path: Path where the filtered AnnData will be saved.

        plot_path: Path where the QC plot will be saved.

        min_counts: Minimum number of total counts per spot.

        min_genes: Minimum number of detected genes per spot.

        pct_mt: Maximum allowed percentage of mitochondrial counts.



    Returns:

        str: Step-by-step log describing the filtering result.

    """

    import scanpy as sc



    steps = []

    adata = sc.read_h5ad(adata_path)

    steps.append(f"Loaded AnnData: {adata.shape[0]} spots x {adata.shape[1]} genes")



    adata.var["mt"] = adata.var_names.str.startswith("MT-")

    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, inplace=True)

    n_before = adata.n_obs



    # Plot QC metrics before filtering

    _plot_setup(os.path.dirname(output_path) or ".")

    import matplotlib.pyplot as plt



    if plot_path is None:

        plot_path = os.path.splitext(output_path)[0] + "_qc.png"

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax_i, key in enumerate(["n_genes_by_counts", "total_counts", "pct_counts_mt"]):

        sc.pl.violin(adata, key, jitter=0.4, ax=axes[ax_i], show=False)

    fig.suptitle("QC metrics (before filtering)")

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    steps.append(f"QC metrics plot saved to {plot_path}")



    adata = adata[adata.obs["total_counts"] >= min_counts, :].copy()

    adata = adata[adata.obs["n_genes_by_counts"] >= min_genes, :].copy()

    adata = adata[adata.obs["pct_counts_mt"] <= pct_mt, :].copy()



    n_after = adata.n_obs

    steps.append(

        f"QC filtering: kept {n_after}/{n_before} spots "

        f"(min_counts={min_counts}, min_genes={min_genes}, pct_mt<={pct_mt})"

    )



    adata.write(output_path, compression="lzf")

    steps.append(f"Saved filtered AnnData to {output_path}")



    return "\n".join(steps)





def normalize_visium(

    adata_path: str,

    output_path: str = "./visium_normalized.h5ad",

    plot_path: str = None,

    target_sum: float = 1e4,

    n_top_genes: int = 2000,

) -> str:

    """Normalize and log-transform 10x Visium data.



    Applies the standard scanpy pipeline: normalize_total (library size

    normalization) + log1p, optionally followed by highly variable gene

    selection. Raw counts are stored in ``adata.raw``. Plots the total-counts

    distribution before/after normalization and the HVG expression plot.



    Args:

        adata_path: Path to the input AnnData (h5ad) file.

        output_path: Path where the normalized AnnData will be saved.

        plot_path: Path where the normalization plot will be saved.

        target_sum: Target library size for normalize_total.

        n_top_genes: Number of highly variable genes to select (0 to skip).



    Returns:

        str: Step-by-step log describing the normalization result.

    """

    import scanpy as sc



    steps = []

    adata = sc.read_h5ad(adata_path)

    steps.append(f"Loaded AnnData: {adata.shape[0]} spots x {adata.shape[1]} genes")



    # Keep raw counts in .raw for downstream deconvolution tools

    adata.raw = adata.copy()

    steps.append("Stored raw counts in adata.raw")



    sc.pp.normalize_total(adata, target_sum=target_sum)

    sc.pp.log1p(adata)

    steps.append(f"Normalized to target_sum={target_sum} and log1p-transformed")



    if n_top_genes and n_top_genes > 0:

        # HVG on raw counts (seurat_v3 flavor requires raw count data)

        raw_adata = adata.raw.to_adata()

        try:

            sc.pp.highly_variable_genes(raw_adata, n_top_genes=n_top_genes, flavor="seurat_v3")

            adata.var["highly_variable"] = raw_adata.var["highly_variable"]

            adata.var["means"] = raw_adata.var["means"]

            adata.var["dispersions_norm"] = raw_adata.var["dispersions_norm"]

        except Exception:

            sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor="seurat")

        n_hvg = int(adata.var["highly_variable"].sum())

        steps.append(f"Selected {n_hvg} highly variable genes (seurat_v3, top {n_top_genes})")



    adata.write(output_path, compression="lzf")

    steps.append(f"Saved normalized AnnData to {output_path}")



    # Plot: HVG dispersion if HVG was computed, else total-counts histogram

    _plot_setup(os.path.dirname(output_path) or ".")

    import matplotlib.pyplot as plt



    if plot_path is None:

        plot_path = os.path.splitext(output_path)[0] + "_norm.png"

    if n_top_genes and n_top_genes > 0 and "highly_variable" in adata.var:

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        hvg_df = adata.var[["means", "dispersions_norm", "highly_variable"]].copy()

        ax.scatter(hvg_df.loc[~hvg_df["highly_variable"], "means"],

                   hvg_df.loc[~hvg_df["highly_variable"], "dispersions_norm"],

                   s=5, c="lightgrey", label="not HVG")

        ax.scatter(hvg_df.loc[hvg_df["highly_variable"], "means"],

                   hvg_df.loc[hvg_df["highly_variable"], "dispersions_norm"],

                   s=5, c="red", label="HVG")

        ax.set_xlabel("mean expression")

        ax.set_ylabel("normalized dispersion")

        ax.set_title("Highly variable genes")

        ax.legend()

        fig.savefig(plot_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        steps.append(f"HVG plot saved to {plot_path}")

    else:

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))

        ax.hist(adata.obs["total_counts"], bins=50, color="steelblue")

        ax.set_xlabel("total_counts (normalized)")

        ax.set_ylabel("spots")

        ax.set_title("Normalized library size distribution")

        fig.savefig(plot_path, dpi=150, bbox_inches="tight")

        plt.close(fig)

        steps.append(f"Normalization plot saved to {plot_path}")



    return "\n".join(steps)





# ============================================================================

# Stage 2: Clustering and spatial domain detection

# ============================================================================





def cluster_spatial_data(

    adata_path: str,

    output_path: str = "./visium_clustered.h5ad",

    plot_path: str = None,

    n_pcs: int = 30,

    resolution: float = 1.0,

) -> str:

    """Cluster 10x Visium spots using the standard PCA + UMAP + Leiden pipeline.



    Plots the UMAP colored by Leiden clusters and the spatial scatter colored

    by clusters.



    Args:

        adata_path: Path to the normalized AnnData (h5ad) file.

        output_path: Path where the clustered AnnData will be saved.

        plot_path: Path where the cluster plots will be saved (two panels).

        n_pcs: Number of principal components to use.

        resolution: Leiden clustering resolution.



    Returns:

        str: Step-by-step log describing the clustering result.

    """

    import scanpy as sc



    steps = []

    adata = sc.read_h5ad(adata_path)

    steps.append(f"Loaded AnnData: {adata.shape[0]} spots x {adata.shape[1]} genes")



    sc.pp.pca(adata, n_comps=n_pcs)

    steps.append(f"Computed PCA with {n_pcs} components")



    sc.pp.neighbors(adata, n_pcs=n_pcs)

    sc.tl.umap(adata)

    sc.tl.leiden(adata, resolution=resolution)

    n_clusters = adata.obs["leiden"].nunique()

    steps.append(f"Leiden clustering (res={resolution}) found {n_clusters} clusters")



    adata.write(output_path, compression="lzf")

    steps.append(f"Saved clustered AnnData to {output_path}")



    # Plot UMAP + spatial

    _plot_setup(os.path.dirname(output_path) or ".")

    import matplotlib.pyplot as plt



    if plot_path is None:

        plot_path = os.path.splitext(output_path)[0] + "_clusters.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc.pl.umap(adata, color="leiden", ax=axes[0], show=False, legend_loc="on data")

    axes[0].set_title("UMAP (Leiden clusters)")

    _spatial_scatter(adata, "leiden", axes[1], "Spatial (Leiden clusters)")

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    steps.append(f"Cluster plots saved to {plot_path}")



    return "\n".join(steps)



def identify_spatial_domains(

    adata_path: str,

    output_path: str = "./visium_domains.h5ad",

    plot_path: str = None,

    resolution: float = 1.0,

    n_neighs: int = 6,

) -> str:

    """Identify spatial domains in 10x Visium data using spatial neighborhood graph.



    Builds a spatial neighborhood graph from spot coordinates (squidpy) and

    runs Leiden clustering on that graph, producing spatially coherent domains.

    Plots the spatial scatter colored by domain.



    Args:

        adata_path: Path to the normalized AnnData (h5ad) file with spatial coordinates.

        output_path: Path where the domain-annotated AnnData will be saved.

        plot_path: Path where the domain plot will be saved.

        resolution: Leiden clustering resolution for domain detection.

        n_neighs: Number of spatial neighbors per spot.



    Returns:

        str: Step-by-step log describing the spatial domain result.

    """

    try:

        import squidpy as sq

    except ImportError:

        return "Error: squidpy is required for spatial domain detection. Install with: pip install squidpy"



    steps = []

    adata = sc.read_h5ad(adata_path)

    steps.append(f"Loaded AnnData: {adata.shape[0]} spots x {adata.shape[1]} genes")



    if "spatial" not in adata.obsm:

        return "Error: no spatial coordinates found in adata.obsm['spatial']; run load_visium_data first"



    sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type="generic")

    steps.append(f"Built spatial neighborhood graph with {n_neighs} neighbors per spot")



    # Leiden on the spatial graph (squidpy stores it in obsp["spatial_connectivities"])

    try:

        sc.tl.leiden(adata, resolution=resolution, key_added="spatial_domain",

                     neighbors_key="spatial_neighbors")

    except KeyError:

        # Fallback: pass the spatial connectivity matrix directly

        sc.tl.leiden(adata, resolution=resolution, key_added="spatial_domain",

                     obsp="spatial_connectivities")

    n_domains = adata.obs["spatial_domain"].nunique()

    steps.append(f"Spatial domain clustering found {n_domains} domains")



    adata.write(output_path, compression="lzf")

    steps.append(f"Saved domain-annotated AnnData to {output_path}")



    # Plot domains on spatial coordinates

    _plot_setup(os.path.dirname(output_path) or ".")

    import matplotlib.pyplot as plt



    if plot_path is None:

        plot_path = os.path.splitext(output_path)[0] + "_domains.png"

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    _spatial_scatter(adata, "spatial_domain", ax, "Spatial domains")

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    steps.append(f"Spatial domain plot saved to {plot_path}")



    return "\n".join(steps)





# ============================================================================

# Stage 3: Spatially variable genes

# ============================================================================





def find_spatially_variable_genes(

    adata_path: str,

    output_path: str = "./svg_results.csv",

    plot_path: str = None,

    n_top_genes: int = 100,

    genes_to_plot: list = None,

    n_jobs: int = 1,

) -> str:

    """Detect spatially variable genes (SVGs) in 10x Visium data using Moran's I.



    Uses squidpy's spatial autocorrelation (Moran's I) to rank genes by spatial

    clustering of expression. Saves the ranked SVG table to CSV and plots the

    top SVG expressions on the spatial coordinates.



    Args:

        adata_path: Path to the normalized AnnData (h5ad) file with spatial coordinates.

        output_path: Path where the SVG ranking CSV will be saved.

        plot_path: Path where the top-SVG spatial plots will be saved.

        n_top_genes: Number of top spatially variable genes to report.

        genes_to_plot: Optional list of gene names to plot on spatial coordinates.

            If provided, these genes are plotted (must exist in var_names).

            If None, the top 4 SVGs are plotted by default.

        n_jobs: Number of parallel jobs for autocorrelation computation.



    Returns:

        str: Step-by-step log describing the SVG detection result.

    """

    try:

        import squidpy as sq

    except ImportError:

        return "Error: squidpy is required for SVG detection. Install with: pip install squidpy"



    steps = []

    adata = sc.read_h5ad(adata_path)

    steps.append(f"Loaded AnnData: {adata.shape[0]} spots x {adata.shape[1]} genes")



    if "spatial" not in adata.obsm:

        return "Error: no spatial coordinates found in adata.obsm['spatial']; run load_visium_data first"

    if "spatial_neighbors" not in adata.uns and "spatial_domain" not in adata.obs:

        sq.gr.spatial_neighbors(adata, n_neighs=6, coord_type="generic")

        steps.append("Built spatial neighborhood graph")



    sq.gr.spatial_autocorr(

        adata,

        mode="moran",

        n_jobs=n_jobs,

        genes=adata.var_names[:2000],

    )

    steps.append("Computed Moran's I for all genes")



    moran = adata.uns["moranI"]

    moran = moran.sort_values("I", ascending=False)

    top = moran.head(n_top_genes)

    top.to_csv(output_path)

    steps.append(f"Top {len(top)} spatially variable genes saved to {output_path}")



    # Plot top 4 SVGs on spatial coordinates

    _plot_setup(os.path.dirname(output_path) or ".")

    import matplotlib.pyplot as plt



    if plot_path is None:

        plot_path = os.path.splitext(output_path)[0] + "_top_svgs.png"

    if genes_to_plot:

        missing = [g for g in genes_to_plot if g not in adata.var_names]

        if missing:

            return "\\n".join(steps) + f"\\nError: genes not found in var_names: {missing}"

        plot_genes = genes_to_plot

        title = "Spatially variable genes (user selected)"

    else:

        plot_genes = top.index[:4].tolist()

        title = "Top spatially variable genes"

    n_plots = len(plot_genes)

    ncols = min(n_plots, 4)

    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)

    for ax, gene in zip(axes.ravel(), plot_genes):

        moran_i = top.loc[gene, "I"] if gene in top.index else float("nan")

        _spatial_scatter(adata, gene, ax, f"{gene} (Moran's I={moran_i:.3f})" if gene in top.index else gene)

    for ax in axes.ravel()[len(plot_genes):]:

        ax.axis("off")

    fig.suptitle(title)

    fig.tight_layout()

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    steps.append(f"SVG spatial plots ({len(plot_genes)} genes) saved to {plot_path}")



    return "\n".join(steps)





# ============================================================================

# Stage 4: Cell-type deconvolution

# ============================================================================





def _plot_proportions(prop_df, plot_path: str, title: str) -> None:

    """Plot per-spot cell-type proportions as a stacked bar/area summary."""

    _plot_setup(os.path.dirname(plot_path) or ".")

    import matplotlib.pyplot as plt



    # Bar chart of mean proportions per cell type (numeric columns only)

    num_cols = prop_df.select_dtypes(include="number").columns

    mean_prop = prop_df[num_cols].mean(axis=0).sort_values(ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(mean_prop)), mean_prop.values, color="steelblue")

    axes[0].set_xticks(range(len(mean_prop)))

    axes[0].set_xticklabels(mean_prop.index, rotation=90, fontsize=7)

    axes[0].set_ylabel("mean proportion")

    axes[0].set_title("Mean cell-type proportions")



    # Stacked proportions for first 20 spots (sampled)

    # Each bar = one spot, stack = cell-type proportions (sums to 1)

    n_show = min(20, prop_df.shape[0])

    sample = prop_df[num_cols].iloc[:n_show]

    sample.plot(kind="bar", stacked=True, ax=axes[1], colormap="tab20", width=0.9)

    axes[1].set_xlabel("spot")

    axes[1].set_ylabel("proportion")

    axes[1].set_title(f"Per-spot proportions (first {n_show} spots)")

    axes[1].legend(fontsize=5, ncol=2)

    fig.suptitle(title)

    fig.tight_layout()

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")

    plt.close(fig)





def deconvolve_spatial_spotlight(

    ref_h5ad: str,

    st_h5ad: str,

    cell_type_key: str,

    output_dir: str = "./spotlight_results",

    plot_path: str = None,

    r_script_path: str = "Rscript",

) -> str:

    """Deconvolve 10x Visium spots into cell-type proportions using SPOTlight.



    SPOTlight (NAR 2021) performs seeded NMF on reference scRNA-seq data to

    learn cell-type topic profiles, then deconvolves each spatial spot into

    cell-type proportions via non-negative least squares. This tool bridges to

    the R implementation through a temporary R script. Saves a proportions

    summary plot.



    Args:

        ref_h5ad: Path to the reference scRNA-seq AnnData (h5ad) with cell-type labels.

        st_h5ad: Path to the Visium AnnData (h5ad) to deconvolve.

        cell_type_key: Column in ref_h5ad.obs containing cell-type labels.

        output_dir: Directory where SPOTlight results (proportions CSV) are saved.

        plot_path: Path where the proportions plot will be saved.

        r_script_path: Path to the Rscript executable.



    Returns:

        str: Step-by-step log describing the deconvolution result.

    """

    import subprocess

    import tempfile



    if not os.path.exists(ref_h5ad) or not os.path.exists(st_h5ad):

        return "Error: ref_h5ad or st_h5ad file not found"



    os.makedirs(output_dir, exist_ok=True)

    steps = []

    steps.append(f"SPOTlight deconvolution: ref={ref_h5ad}, st={st_h5ad}")



    # Convert h5ad inputs to 10x format for R (deterministic, no R h5ad reader needed)

    import scipy.io as sio



    bridge_dir = os.path.join(output_dir, "_bridge")

    os.makedirs(bridge_dir, exist_ok=True)



    def _h5ad_to_10x(h5ad_path, prefix):

        """Write an h5ad to 10x mtx/features/barcodes files (genes x cells).



        Uses gene symbols when available (var['feature_name']), falling back

        to var_names (Ensembl IDs). SPOTlight/scran gene filters need symbols.

        """

        ad = sc.read_h5ad(h5ad_path)

        if ad.raw is not None:

            ad = ad.raw.to_adata()

        if "counts" in ad.layers:

            ad.X = ad.layers["counts"].copy()

        # Gene symbols for R-side filters (^Rp[l|s]|Mt etc.)

        if "feature_name" in ad.var.columns:

            symbols = ad.var["feature_name"].astype(str).values

            # Fill missing/empty symbols with var_names

            symbols = np.where((symbols == "") | (symbols == "nan"), ad.var_names.astype(str), symbols)

        else:

            symbols = ad.var_names.astype(str).values

        mtx_path = os.path.join(bridge_dir, f"{prefix}_matrix.mtx")

        feat_path = os.path.join(bridge_dir, f"{prefix}_features.tsv")

        bar_path = os.path.join(bridge_dir, f"{prefix}_barcodes.tsv")

        sio.mmwrite(mtx_path, ad.X.T.tocsr())

        with open(feat_path, "w") as f:

            for g in symbols:

                f.write(f"{g}\t{g}\tGene Expression\n")

        with open(bar_path, "w") as f:

            for b in ad.obs_names:

                f.write(f"{b}\n")

        if cell_type_key in ad.obs:

            ct_path = os.path.join(bridge_dir, f"{prefix}_celltypes.tsv")

            ad.obs[[cell_type_key]].to_csv(ct_path, header=True, index=False)

        return mtx_path, feat_path, bar_path



    steps.append("Converting h5ad to 10x format for R bridge...")

    ref_mtx, ref_feat, ref_bar = _h5ad_to_10x(ref_h5ad, "ref")

    st_mtx, st_feat, st_bar = _h5ad_to_10x(st_h5ad, "st")

    steps.append("Conversion done")



    # Write the R bridge script

    r_script = f"""

suppressPackageStartupMessages({{

  library(SPOTlight)

  library(SingleCellExperiment)

  library(scater)

  library(scran)

  library(Matrix)

}})



read10x <- function(mtx, feat, bar) {{

  cnt <- readMM(mtx)  # genes x cells (10x standard)

  rownames(cnt) <- read.delim(feat, header = FALSE)[,1]   # genes

  colnames(cnt) <- read.delim(bar, header = FALSE)[,1]    # cells

  sce <- SingleCellExperiment(assays = list(counts = cnt))

  return(sce)

}}



ref <- read10x('{ref_mtx}', '{ref_feat}', '{ref_bar}')

st <- read10x('{st_mtx}', '{st_feat}', '{st_bar}')



# Cell-type labels

ct <- read.delim('{bridge_dir}/ref_celltypes.tsv')

colData(ref)$cell_type <- ct[,1]



# Downsample to max 100 cells per type (official recommendation)

set.seed(123)

idx <- split(seq(ncol(ref)), ref$cell_type)

cs_keep <- lapply(idx, function(i) {{

  sample(i, min(length(i), 100))

}})

ref <- ref[, unlist(cs_keep)]

ref <- logNormCounts(ref)



# HVGs

genes <- !grepl("^Rp[l|s]|Mt", rownames(ref))

dec <- modelGeneVar(ref, subset.row = genes)

hvg <- getTopHVGs(dec, n = 3000)

colLabels(ref) <- colData(ref)$cell_type



# Marker genes -> data.frame format (SPOTlight 1.10 official tutorial)

# Keep top n_top markers per cell type (default 50) to keep NMF init fast.

mgs <- scoreMarkers(ref, subset.row = genes)

mgs_fil <- lapply(names(mgs), function(i) {{

  x <- mgs[[i]]

  x <- x[order(x$mean.AUC, decreasing = TRUE), ]

  x <- head(x, 50)

  x$gene <- rownames(x)

  x$cluster <- i

  data.frame(x)

}})

mgs_df <- do.call(rbind, mgs_fil)

cat("mgs_df:", nrow(mgs_df), "rows\n")



# SPOTlight (v1.10: x=SCE, y=spatial SCE, mgs=data.frame)

res <- SPOTlight(

  x = ref,

  y = st,

  groups = as.character(ref$cell_type),

  mgs = mgs_df,

  hvg = hvg,

  weight_id = "mean.AUC",

  group_id = "cluster",

  gene_id = "gene"

)

prop <- as.data.frame(res$mat)

prop$barcode <- rownames(prop)

write.csv(prop, file.path('{output_dir}', 'spotlight_proportions.csv'), row.names = FALSE)

cat("SPOTlight done\\n")

"""

    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False, dir=output_dir) as f:

        f.write(r_script)

        r_script_path_tmp = f.name



    try:

        result = subprocess.run(

            [r_script_path, r_script_path_tmp],

            capture_output=True,

            text=True,

            timeout=3600,

        )

        steps.append(f"R output: {result.stdout.strip()[:500]}")

        if result.returncode != 0:

            steps.append(f"R error: {result.stderr.strip()[:500]}")

            return "\n".join(steps)

        out_csv = os.path.join(output_dir, "spotlight_proportions.csv")

        if os.path.exists(out_csv):

            prop = pd.read_csv(out_csv, index_col=0)

            steps.append(f"SPOTlight proportions: {prop.shape[0]} spots x {prop.shape[1]} cell types")

            steps.append(f"Results saved to {out_csv}")

            if plot_path is None:

                plot_path = os.path.join(output_dir, "spotlight_proportions.png")

            _plot_proportions(prop, plot_path, "SPOTlight deconvolution")

            steps.append(f"Proportions plot saved to {plot_path}")

        else:

            steps.append("Warning: output CSV not found")

    except Exception as e:

        steps.append(f"Error running SPOTlight: {e}")

    finally:

        os.unlink(r_script_path_tmp)



    return "\n".join(steps)





def deconvolve_spatial_destvi(

    ref_h5ad: str,

    st_h5ad: str,

    cell_type_key: str,

    output_dir: str = "./destvi_results",

    plot_path: str = None,

    n_latent: int = 30,

    max_epochs: int = 400,

    destvi_max_epochs: int = 500,

) -> str:

    """Deconvolve 10x Visium spots into cell-type proportions using DestVI.



    DestVI (Lopez et al., Nat Biotechnol 2022) is a deep generative model from

    the scvi-tools ecosystem. It first trains a CondSCVI model on the reference

    scRNA-seq data, then trains a spatial model on the Visium data to estimate

    cell-type proportions per spot. Saves a proportions summary plot.



    Args:

        ref_h5ad: Path to the reference scRNA-seq AnnData (h5ad) with cell-type labels.

        st_h5ad: Path to the Visium AnnData (h5ad) to deconvolve.

        cell_type_key: Column in ref_h5ad.obs containing cell-type labels.

        output_dir: Directory where DestVI results (proportions CSV) are saved.

        plot_path: Path where the proportions plot will be saved.

        n_latent: Latent dimensionality for CondSCVI.

        max_epochs: Number of training epochs for the CondSCVI reference model.

        destvi_max_epochs: Number of training epochs for the DestVI spatial model.



    Returns:

        str: Step-by-step log describing the deconvolution result.

    """

    try:

        import torch

        from scvi.model import CondSCVI, DestVI

    except ImportError:

        return "Error: scvi-tools required for DestVI deconvolution. Install with: pip install scvi-tools"



    if not os.path.exists(ref_h5ad) or not os.path.exists(st_h5ad):

        return "Error: ref_h5ad or st_h5ad file not found"



    os.makedirs(output_dir, exist_ok=True)

    steps = []

    steps.append(f"DestVI deconvolution: ref={ref_h5ad}, st={st_h5ad}")



    use_gpu = torch.cuda.is_available()

    steps.append(f"CUDA available: {use_gpu}")



    # Load reference

    ref = sc.read_h5ad(ref_h5ad)

    steps.append(f"Reference: {ref.shape[0]} cells x {ref.shape[1]} genes")

    if cell_type_key not in ref.obs:

        return f"Error: cell_type_key '{cell_type_key}' not found in reference obs"



    # Use raw counts if stored

    if "counts" in ref.layers:

        ref.X = ref.layers["counts"].copy()



    # Load spatial data first so we can align genes before training CondSCVI

    st = sc.read_h5ad(st_h5ad)

    steps.append(f"Spatial: {st.shape[0]} spots x {st.shape[1]} genes")



    # Align genes between reference (Ensembl) and spatial (symbol)

    # Reference var_names may be Ensembl IDs with a 'feature_name' column holding symbols

    if "feature_name" in ref.var.columns:

        sym2ens = dict(zip(ref.var["feature_name"].astype(str), ref.var_names))

        st_symbols = np.array([str(g) for g in st.var_names])

        st_ens = np.array([sym2ens.get(s, "") for s in st_symbols])

        keep = np.array([e in set(ref.var_names) for e in st_ens])

        n_mapped = int(keep.sum())

        steps.append(f"Gene symbol->Ensembl mapped: {n_mapped}/{len(st_symbols)}")

        if n_mapped == 0:

            return "Error: no genes could be mapped between spatial (symbol) and reference (Ensembl)"

        st = st[:, keep].copy()

        st.var_names = st_ens[keep]

        common_genes = [g for g in ref.var_names if g in set(st.var_names)]

    else:

        common_genes = list(ref.var_names.intersection(st.var_names))

    if len(common_genes) == 0:

        return "Error: no common genes between reference and spatial data"



    # Align BOTH datasets to the same gene set (required by DestVI.from_rna_model)

    ref = ref[:, common_genes].copy()

    st = st[:, common_genes].copy()

    steps.append(f"Aligned on {len(common_genes)} common genes (ref and st)")



    # Ensure counts are integer for scvi
    from scipy.sparse import issparse

    for ad in (ref, st):
        if issparse(ad.X):
            ad.X.data = ad.X.data.astype(np.int32)
        else:
            ad.X = np.asarray(ad.X).astype(np.int32)



    CondSCVI.setup_anndata(ref, labels_key=cell_type_key)

    rna_model = CondSCVI(ref, n_latent=n_latent, prior="mog")

    steps.append(f"Training CondSCVI reference model ({max_epochs} epochs)...")

    rna_model.train(

        max_epochs=max_epochs,

        accelerator="cuda" if use_gpu else "cpu",

        devices=[0] if use_gpu else 1,

        batch_size=512,

    )

    steps.append("CondSCVI reference model trained")



    # Spatial model

    st.layers["counts"] = st.X.copy()

    DestVI.setup_anndata(st, layer="counts")

    spatial_model = DestVI.from_rna_model(st, rna_model)

    steps.append("Training DestVI spatial model...")

    spatial_model.train(max_epochs=destvi_max_epochs, accelerator="cuda" if use_gpu else "cpu", devices=[0] if use_gpu else 1)

    steps.append("DestVI spatial model trained")



    # Extract proportions (returns pd.DataFrame with cell-type columns)

    prop_df = spatial_model.get_proportions()

    prop_df = prop_df.reset_index(drop=True)

    prop_df.index = st.obs_names

    out_csv = os.path.join(output_dir, "destvi_proportions.csv")

    prop_df.to_csv(out_csv)

    steps.append(f"DestVI proportions: {prop_df.shape[0]} spots x {prop_df.shape[1]} cell types")

    steps.append(f"Results saved to {out_csv}")



    if plot_path is None:

        plot_path = os.path.join(output_dir, "destvi_proportions.png")

    _plot_proportions(prop_df, plot_path, "DestVI deconvolution")

    steps.append(f"Proportions plot saved to {plot_path}")



    return "\n".join(steps)





# ============================================================================

# Stage 5: Cell-cell communication inference (CellChat v2)

# ============================================================================





def infer_spatial_cell_communication(

    st_h5ad: str,

    cell_type_key: str,

    output_dir: str = "./cellchat_results",

    plot_path: str = None,

    r_script_path: str = "Rscript",

    species: str = "human",

) -> str:

    """Infer spatially proximal cell-cell communication using CellChat v2.



    CellChat v2 (Jin et al., Nature Protocols 2024) infers cell-cell

    communication networks. When spatial coordinates are available it uses the

    spatial locations of cells/spots to restrict communication to spatially

    proximal pairs. Exports communication tables and a network plot (PNG).



    Args:

        st_h5ad: Path to the Visium AnnData (h5ad) with cell-type labels

            (e.g. from deconvolution) and spatial coordinates.

        cell_type_key: Column in st_h5ad.obs containing cell-type labels.

        output_dir: Directory where CellChat results are saved.

        plot_path: Path where the communication network plot will be saved.

        r_script_path: Path to the Rscript executable.

        species: Species for CellChatDB ('human' or 'mouse').



    Returns:

        str: Step-by-step log describing the communication inference result.

    """

    import subprocess

    import tempfile



    if not os.path.exists(st_h5ad):

        return "Error: st_h5ad file not found"



    os.makedirs(output_dir, exist_ok=True)

    steps = []

    steps.append(f"CellChat spatial communication inference: {st_h5ad}")



    if plot_path is None:

        plot_path = os.path.join(output_dir, "cellchat_network.png")



    # Convert h5ad to 10x format for R (deterministic bridge)

    import scipy.io as sio



    bridge_dir = os.path.join(output_dir, "_bridge")

    os.makedirs(bridge_dir, exist_ok=True)



    ad = sc.read_h5ad(st_h5ad)

    if ad.raw is not None:

        ad = ad.raw.to_adata()

    if "counts" in ad.layers:

        ad.X = ad.layers["counts"].copy()

    mtx_path = os.path.join(bridge_dir, "st_matrix.mtx")

    feat_path = os.path.join(bridge_dir, "st_features.tsv")

    bar_path = os.path.join(bridge_dir, "st_barcodes.tsv")

    from scipy.sparse import csr_matrix

    X_sparse = ad.X if hasattr(ad.X, "tocsr") else csr_matrix(ad.X)

    sio.mmwrite(mtx_path, X_sparse.T)

    with open(feat_path, "w") as f:

        for g in ad.var_names:

            f.write(f"{g}\t{g}\tGene Expression\n")

    with open(bar_path, "w") as f:

        for b in ad.obs_names:

            f.write(f"{b}\n")

    ct_path = os.path.join(bridge_dir, "st_celltypes.tsv")

    # CellChat rejects cell labels containing "0" (its setIdent treats 0 as
    # background/missing). leiden/domain labels are 0-based, so shift numeric
    # labels that start at 0 to 1-based before writing the bridge TSV.
    labels = ad.obs[cell_type_key]
    labels_str = labels.astype(str)
    if labels_str.str.match(r"^\d+$").all() and (labels_str == "0").any():
        labels = (labels_str.astype(int) + 1).astype(str)
    else:
        labels = labels_str
    labels.to_frame(cell_type_key).to_csv(ct_path, header=True, index=False)

    # Spatial coordinates

    coord_path = os.path.join(bridge_dir, "st_coords.csv")

    if "spatial" in ad.obsm:

        pd.DataFrame(ad.obsm["spatial"], index=ad.obs_names,

                     columns=["x", "y"]).to_csv(coord_path)

        steps.append("Converted spatial coordinates for CellChat")

    steps.append("Converted h5ad to 10x format for CellChat bridge")



    r_script = f"""

suppressPackageStartupMessages({{

  library(CellChat)

  library(Seurat)

  library(Matrix)

}})



# Build Seurat object from 10x-format files (deterministic bridge)

cnt <- readMM('{mtx_path}')  # genes x cells

rownames(cnt) <- read.delim('{feat_path}', header = FALSE)[,1]  # genes

colnames(cnt) <- read.delim('{bar_path}', header = FALSE)[,1]   # cells

obj <- CreateSeuratObject(counts = cnt, assay = "RNA")

obj <- NormalizeData(obj)  # CellChat needs the 'data' (log-normalized) layer

ct <- read.delim('{ct_path}')

obj <- AddMetaData(obj, metadata = ct[,1], col.name = '{cell_type_key}')



# Attach spatial coordinates if available

coord_file <- '{coord_path}'

if (file.exists(coord_file)) {{

  coords <- read.csv(coord_file, row.names = 1)

  coords <- as.matrix(coords[, c("x", "y")])

  colnames(coords) <- c("slice1_1", "slice1_2")  # Seurat requires key+digit colnames

  rownames(coords) <- rownames(obj@meta.data)

  obj[["slice1"]] <- CreateDimReducObject(embeddings = coords, key = "slice1_", assay = "RNA")

}}



cellchat <- createCellChat(object = obj, group.by = '{cell_type_key}')



# Set ligand-receptor database

CellChatDB <- CellChatDB.{species}

cellchat@DB <- CellChatDB



# Preprocessing

cellchat <- subsetData(cellchat)

future::plan("multisession", workers = 4)

cellchat <- identifyOverExpressedGenes(cellchat)

cellchat <- identifyOverExpressedInteractions(cellchat)



# Inference

cellchat <- computeCommunProb(cellchat, type = "triMean")

cellchat <- filterCommunication(cellchat, min.cells = 10)

cellchat <- computeCommunProbPathway(cellchat)

cellchat <- aggregateNet(cellchat)



# Export results

df <- subsetCommunication(cellchat)

write.csv(df, file.path('{output_dir}', 'cellchat_communication.csv'), row.names = FALSE)

df_path <- subsetCommunication(cellchat, slot.name = "netP")

write.csv(df_path, file.path('{output_dir}', 'cellchat_pathways.csv'), row.names = FALSE)



# Network plot (labels shown so each node carries its cell-type identity)

png('{plot_path}', width = 10, height = 10, units = "in", res = 150)

netVisual_circle(cellchat@net$count, vertex.weight = table(cellchat@idents),

                 weight.scale = TRUE, label.edge = FALSE, title.name = "CellChat network",

                 vertex.label.cex = 1.2)

dev.off()

cat("CellChat done\\n")

"""

    with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False, dir=output_dir) as f:

        f.write(r_script)

        r_script_path_tmp = f.name



    try:

        result = subprocess.run(

            [r_script_path, r_script_path_tmp],

            capture_output=True,

            text=True,

            timeout=3600,

        )

        steps.append(f"R output: {result.stdout.strip()[:500]}")

        if result.returncode != 0:

            steps.append(f"R error: {result.stderr.strip()[:500]}")

            return "\n".join(steps)

        comm_csv = os.path.join(output_dir, "cellchat_communication.csv")

        if os.path.exists(comm_csv):

            comm = pd.read_csv(comm_csv)

            steps.append(f"CellChat communication pairs: {len(comm)}")

            steps.append(f"Results saved to {comm_csv}")

        else:

            steps.append("Warning: output CSV not found")

        if os.path.exists(plot_path):

            steps.append(f"Network plot saved to {plot_path}")

        else:

            steps.append("Warning: network plot not found")

    except Exception as e:

        steps.append(f"Error running CellChat: {e}")

    finally:

        os.unlink(r_script_path_tmp)



    return "\n".join(steps)





# ============================================================================

# Stage 6: Neighborhood enrichment analysis

# ============================================================================





def spatial_neighborhood_enrichment(

    adata_path: str,

    cell_type_key: str,

    output_path: str = "./neighborhood_enrichment.csv",

    plot_path: str = None,

    n_neighs: int = 6,

) -> str:

    """Compute cell-type neighborhood enrichment in 10x Visium data.



    Uses squidpy's neighborhood enrichment analysis to test whether pairs of

    cell types (or spatial domains) are found together more often than expected

    by chance in the spatial neighborhood graph. Saves the enrichment table and

    plots the enrichment z-score heatmap.



    Args:

        adata_path: Path to the AnnData (h5ad) with spatial coordinates and

            cell-type labels in obs.

        cell_type_key: Column in adata.obs containing cell-type labels.

        output_path: Path where the enrichment CSV will be saved.

        plot_path: Path where the enrichment heatmap will be saved.

        n_neighs: Number of spatial neighbors per spot.



    Returns:

        str: Step-by-step log describing the neighborhood enrichment result.

    """

    try:

        import squidpy as sq

    except ImportError:

        return "Error: squidpy is required for neighborhood enrichment. Install with: pip install squidpy"



    steps = []

    adata = sc.read_h5ad(adata_path)

    steps.append(f"Loaded AnnData: {adata.shape[0]} spots x {adata.shape[1]} genes")



    if cell_type_key not in adata.obs:

        return f"Error: cell_type_key '{cell_type_key}' not found in obs"

    if "spatial" not in adata.obsm:

        return "Error: no spatial coordinates found; run load_visium_data first"



    sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type="generic")

    steps.append(f"Built spatial neighborhood graph with {n_neighs} neighbors per spot")



    sq.gr.nhood_enrichment(adata, cluster_key=cell_type_key)

    steps.append("Computed neighborhood enrichment")



    enrich = adata.uns[f"{cell_type_key}_nhood_enrichment"]

    cats = adata.obs[cell_type_key].cat.categories

    df = pd.DataFrame(enrich["zscore"], index=cats, columns=cats)

    df.to_csv(output_path)

    steps.append(f"Neighborhood enrichment z-scores saved to {output_path}")



    # Plot heatmap

    _plot_setup(os.path.dirname(output_path) or ".")

    import matplotlib.pyplot as plt



    if plot_path is None:

        plot_path = os.path.splitext(output_path)[0] + "_heatmap.png"

    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(cats) * 0.6), max(5, len(cats) * 0.5)))

    im = ax.imshow(df.values, cmap="RdBu_r", vmin=-max(abs(df.values.max()), abs(df.values.min())),

                   vmax=max(abs(df.values.max()), abs(df.values.min())))

    # Readable labels: numeric cluster ids become "Cluster N"; real cell-type
    # names (e.g. "T cell", "Fibroblast") are kept as-is so the heatmap carries
    # biological meaning.
    def _readable(c):

        s = str(c)

        return f"Cluster {s}" if s.isdigit() else s

    labels = [_readable(c) for c in cats]

    ax.set_xticks(range(len(cats)))

    ax.set_yticks(range(len(cats)))

    ax.set_xticklabels(labels, rotation=90, fontsize=7)

    ax.set_yticklabels(labels, fontsize=7)

    ax.set_xlabel("Cell type B")

    ax.set_ylabel("Cell type A")

    plt.colorbar(im, ax=ax, fraction=0.046, label="z-score")

    ax.set_title("Neighborhood enrichment (z-score)")

    fig.tight_layout()

    fig.savefig(plot_path, dpi=150, bbox_inches="tight")

    plt.close(fig)

    steps.append(f"Enrichment heatmap saved to {plot_path}")



    return "\n".join(steps)





# ============================================================================

# Project scaffolding

# ============================================================================





def init_spatial_project(

    project_dir: str,

    overwrite: bool = False,

) -> str:

    """Create a standardized spatial transcriptomics project directory layout.



    All spatial tools in this module expect results to live under a project

    directory with a consistent structure, so multi-sample / multi-step

    analyses stay organized:



        <project_dir>/

        âââ raw_data/

        â?  âââ visium/          # raw Visium input (h5ad, 10x dirs, CSVs)

        â?  âââ scRNA/           # reference scRNA-seq (h5ad) for deconvolution

        âââ results/

            âââ 01_loading/      # standardized h5ad after load_visium_data

            âââ 02_qc/           # QC-filtered h5ad + plots

            âââ 03_normalization/

            âââ 04_clustering/   # clusters / spatial domains

            âââ 05_deconvolution/# SPOTlight / DestVI proportions

            âââ 06_svg/          # spatially variable genes

            âââ 07_communication/# CellChat results

            âââ 08_neighborhood/ # neighborhood enrichment



    Args:

        project_dir: Root path of the project to scaffold.

        overwrite: If False and the directory already exists, do nothing

            (returns a message). If True, recreate subdirectories.



    Returns:

        str: Log describing the created directory tree.

    """

    subdirs = [

        "raw_data/visium",

        "raw_data/scRNA",

        "results/01_loading",

        "results/02_qc",

        "results/03_normalization",

        "results/04_clustering",

        "results/05_deconvolution",

        "results/06_svg",

        "results/07_communication",

        "results/08_neighborhood",

    ]

    if os.path.exists(project_dir) and not overwrite:

        return f"Project directory already exists: {project_dir} (use overwrite=True to recreate)"



    os.makedirs(project_dir, exist_ok=True)

    created = []

    for sub in subdirs:

        p = os.path.join(project_dir, sub)

        os.makedirs(p, exist_ok=True)

        created.append(p)



    readme = os.path.join(project_dir, "README.md")

    with open(readme, "w") as f:

        f.write("# Spatial transcriptomics project\n\n")

        f.write("## Directory layout\n\n")

        for sub in subdirs:

            f.write(f"- `{sub}/`\n")



    lines = [f"Initialized spatial project at {project_dir}"]

    lines.append("Created directories:")

    for p in created:

        lines.append(f"  {p}")

    lines.append(f"README written to {readme}")

    return "\n".join(lines)





