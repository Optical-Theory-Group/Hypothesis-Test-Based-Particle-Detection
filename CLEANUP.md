## Superfluous Files - Candidates for Removal

The following files/folders may be outdated, redundant, or no longer actively used:

### Potentially Obsolete Files

1. **`glrt.py`** (root directory)
   - Not called anywhere and has missing dependencies so is non-functional


### Scripts Requiring Review

1. **`scripts/unclassified_scripts.py`**
   - Name suggests unsorted/temporary utilities
   - Review contents and either:
     - Integrate useful functions into main modules
     - Document and rename appropriately
     - Remove if obsolete

2. **`scripts/json_scripts.py`**
   - May duplicate functionality in main.py config handling
   - Review for redundancy

3. **`scripts/imgproc_scripts.py`**
   - Check if functions are used elsewhere or redundant with process_algorithms.py

### Notebooks in Scripts Folder

1. **`scripts/figures_notebook.ipynb`**
   - Review if functionality is duplicated by `exp_data_analysis.py`
   - Consider consolidating plotting code

2. **`scripts/xivalues_pixelvalhistogram_psfwidth_autocorrelation_figures.ipynb`**
   - Long diagnostic name, likely for development/paper figures
   - Archive if analysis is complete



**Note:** Before removing any files, ensure they're not referenced in active analysis workflows or required for reproducing published results.
