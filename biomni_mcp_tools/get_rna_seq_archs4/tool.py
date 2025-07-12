import gget

def get_rna_seq_archs4(gene_name: str, K: int = 10) -> dict:
    """Fetch RNA-seq expression for a gene from ARCHS4, returning top K tissues by median TPM."""
    result = {"gene_name": gene_name, "K": K, "tissues": [], "log": ""}
    try:
        result["log"] += f"Fetching RNA-seq data using gget.archs4 for gene: {gene_name}...\n"
        data = gget.archs4(gene_name, which="tissue")
        if data.empty:
            result["log"] += f"No RNA-seq data found for the gene {gene_name}.\n"
            return result
        result["log"] += f"RNA-seq expression data for {gene_name} fetched successfully. Formatting the top {K} tissues:\n"
        for index, row in data.iterrows():
            if index < K:
                result["tissues"].append({"tissue": row["id"], "median_tpm": row["median"]})
            else:
                break
    except Exception as e:
        result["log"] += f"An error occurred: {e}"
    return result
