
suppressPackageStartupMessages({
  library(SingleCellExperiment)
  library(flexmix)
  library(scater)
  library(splines)
  library(miQC)
  library(zellkonverter)
  library(Seurat)
  library(SeuratDisk)
  library(scRNAseq)
})

apply_miQC <- function(input_path, output_path) {

  # Load the data
  sce <- readH5AD(input_path)

  # Assuming necessary preprocessing has been done in Python
  # Apply miQC
  # Setup mitochondrial genes
  mt_genes <- grepl("^mt-", rownames(sce), ignore.case = TRUE)
  if (sum(mt_genes) == 0) {
    # Try with "MT-" if no "mt-" genes are found
    mt_genes <- grepl("^MT-", rownames(sce))
  }
  if (sum(mt_genes) == 0) {
    # Try with "MT-" if no "mt-" genes are found
    mt_genes <- grepl("^Mt-", rownames(sce))
  }
  feature_ctrls <- list(mito = rownames(sce)[mt_genes])
  sce <- addPerCellQC(sce, subsets = feature_ctrls)

  model <- mixtureModel(sce)
  sce_filtered <- filterCells(sce, model)

  # Save the processed data
  writeH5AD(sce_filtered, output_path)
}
