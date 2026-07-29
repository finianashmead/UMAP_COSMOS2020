# UMAP_COSMOS2020
Code and data pertaining to my work using UMAP dimensionality reduction to develop improved photo-z training sets for cosmology with the Roman Space Telescope, prototyping on the COSMOS2020 catalog.

The Python file and notebooks corresponding to the newest version (26/07/28) are: umapz2.py, embedding_optimization_SOM.ipynb, embedding_optimization_UMAP.ipynb, run_all_embeddings.ipynb, colors_kNN_z_unc_budget.ipynb, somz_unc_budget.ipynb, nearestz_unc_budget.ipynb, UMAP_unc_budget.ipynb, global_w_unc_budget.ipynb, highz_global_w_unc_budget.ipynb, SOMtco_global.ipynb, and SOMtco_highz.ipynb. The final two of these were used to calculate the parenthetical and footnote values that appear in Table 1 of the paper.

process_COSMOS.py was used in the initial data processing (dust correction, etc.), utils.py contains some functions used in SOM calculations.

The files COSMOS2020_processed_260724.parquet and speczCL95_processed_260724.parquet contain fully processed data, encompassing the processing performed in process_COSMOS.py in addition to the cuts on the data and crossmatching described in Section 2 of our paper. The original versions of the COSMOS2020 catalog and COSMOS Spectroscopic Redshift Compilation catalog were downloaded from https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/ and https://github.com/cosmosastro/speczcompilation/tree/main/specz_compilation, respectively.

umapz.py and paper_example_notebook.ipynb correspond to the original version of this paper submitted to ApJ and available on arxiv at https://arxiv.org/abs/2512.09032
