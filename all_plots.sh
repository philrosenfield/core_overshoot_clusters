# CMDs of gst and asteca membership catalogs
pythonw -m core_overshoot_clusters.cmds
# Core overshooting tracks HRD, CMD (-t),
# Lifetime comparisons (-l),
# Synthetic cmds (-m),
# Core overshooting tracks HRD, CMD with different modelling groups (-s)
pythonw -m core_overshoot_clusters.covs -mlts
# PDFs of clusters, marginalized over 0.5 Hp overshooting (--ov5), and mock data (-t)
pythonw -m core_overshoot_clusters.pdfs -t
pythonw -m core_overshoot_clusters.pdfs --ov5
pythonw -m core_overshoot_clusters.pdfs
