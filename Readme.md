# Constraining Core Overshooting with LMC Clusters

This repository holds the codes and data products to reproduce the figures and the open-source aspects of the analysis of the paper Rosenfield et al 2017 (MATCH and PARSEC are closed-source, only their data products are available here).

This repo is designed to reproduce the figures from the paper. The full grid of core overshooting tracks is available elsewhere.

## Code:
* codebase: graphics and analysis used in the paper. For an intermediate python user, these codes can be generalized to other tracks in the repo, or forked/used for other projects.

## Data products:
* Photometry related (for each cluster)
   * Reduced HST observations: `.gst` catalogs
   * Artificial star tests files
   * `ASteCA` membership probabilities calculated from the `.gst` catalogs


* Stellar model related
  * Part of the `PARSEC V1.2S` core overshooting grid
  * Synthetic stellar populations created from `MATCH.fake` for use in the paper


* Files used to create posterior distribution functions
  * likelihood files from synthetic stellar populations
  * likelihood files from 70% membership probability catalogs (for each cluster)

## Making the Figures from the Paper
Figure 1 and 2 are not included. To make the rest of the figures run `bash all_plots.sh`

### Figure-by-Figure
* Figure 3: CMDs of the clusters.
  * data: `phot/obs/*gst` and `phot/asteca/*dat`
  * main code: `code/cmd_plots.cmd_plots`
  * make the figure: `python -m clusters_data.code.cmds`
* Figure 4: The effect of core overshooting on H and He burning lifetimes as a function stellar mass.
  * data: `stev/track_summary.dat`
  * main code: `cov_plots.cov_complifetimes`
  * make the figure: `python -m clusters_data.code.covs -l`
* Figure 5: HRD and CMD of tracks as a function of core overshooting
  * data: `stev/...`
  * main code: `cov_plots.plot_compare_tracks`
  * make the figure: `python -m clusters_data.code.covs -t`
* Figure 6: CMDs of synthetic stellar populations (mock data)
  * data: `mock/*full`
  * main code: `cov_plots.fake_cmds`
  * make the figure: `python -m clusters_data.code.covs -m`
* Figure 7+: PDF plots (marginalized or joint-marginalized)
  * data:
    * Fig 7, 8: `pdfs/mock/`
    * Fig 9+: `pdfs/clusters/`
  * main code: `pdf_plots.cluster_result_plots`
  * make the figures:
    * Fig 7, 8: `python -m clusters_data.code.pdfs -t`
    * Fig 9, 10: `python -m clusters_data.code.pdfs -m`
    * Fig 11, 12: `python -m clusters_data.code.pdfs`
