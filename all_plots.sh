# CMDs of gst and asteca membership catalogs
python -m core_overshoot_clusters.code.cmds
# Core overshooting tracks HRD, CMD (-t),
# Lifetime comparisons (-l),
# Synthetic cmds (-m),
# Core overshooting tracks HRD, CMD with different modelling groups (-s)
python -m core_overshoot_clusters.code.covs -mlts
# PDFs of clusters, marginalized over 0.5 Hp overshooting (--ov5), and mock data (-t)
python -m core_overshoot_clusters.code.pdfs -t
python -m core_overshoot_clusters.code.pdfs --ov5
python -m core_overshoot_clusters.code.pdfs
