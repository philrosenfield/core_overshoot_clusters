# CMDs of gst and asteca membership catalogs
python -m clusters_data.code.cmds
# Core overshooting tracks HRD, CMD (-t),
# lifetime comparisons (-l),
# and synthetic cmds (-m)
python -m clusters_data.code.covs -mlt
# PDFs of clusters, marginalized over 0.5 Hp overshooting (--ov5), and mock data (-t)
python -m clusters_data.code.pdfs -t
python -m clusters_data.code.pdfs --ov5
python -m clusters_data.code.pdfs
