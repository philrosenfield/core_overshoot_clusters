# Photometry
1. asteca : Membership probability catalogs (from .gst) derived using [ASteCA](http://asteca.github.io) (ascii)
1. asts : Culled columns from artificial star test files from [DOLPHOT](http://americano.dolphinsim.com/dolphot/) (ascii)
1. obs : Re-reduced HST observations via the UW Photometry Pipeline. These are the `.gst` catalogs level. (Binary fits tables)

Filenames are `_` separated values of HST proposal id, cluster name, and filterset.

Columns:
1. ASteCA:
`RA`, `DEC`, `mag`, `e_mag`, `col1`, and `e_col1` are taken from the binary fits tables. These files have been culled by ASteCA to within the cluster radius.

       ID : index
       x : RA
       y : DEC
       mag : magnitude1
       e_mag : mag1 err
       col1 : color
       e_col1 : color error (quadrature mag errs)
       memb_prob : membership probability
       sel : selection

1. ASTs:

       mag1in : input magnitude1
       mag2in : input magnitude2
       diff1 : input magnitude1 - output magnitude1
       diff2 : input magnitude2 - output magnitude2

1. obs :
For information on how to read binary fits tables see [astropy](http://docs.astropy.org/en/stable/io/fits/index.html).

  Columns:
       X                                     
       Y                                     
       RA                                    
       DEC                                   
  For each filter listed in the filename, the following columns exist:
       [filter]_RATE                            
       [filter]_RATERR                          
       [filter]_VEGA : vegamag
       [filter]_STD
       [filter]_ERR : vegamag err
  The following refer to DOLPHOT flags (see DOLPHOT readme file)
       [filter]_CHI
       [filter]_SNR
       [filter]_SHARP
       [filter]_ROUND
       [filter]_CROWD
       [filter]_FLAG
## File listing
### ./asteca
       12257_HODGE2_F475W-F814W_gst_memb.dat
       12257_NGC2213_F475W-F814W_gst_memb.dat
       12257_NGC1718_F475W-F814W_gst_memb.dat
       9891_NGC1644_F555W-F814W_gst_memb.dat
       12257_NGC2203_F475W-F814W_gst_memb.dat
       9891_NGC1795_F555W-F814W_gst_memb.dat
### ./asts
       12257_HODGE2_F475W-F814W.gst.matchfake
       12257_NGC2213_F475W-F814W.gst.matchfake
       12257_NGC1718_F475W-F814W.gst.matchfake
       9891_NGC1644_F555W-F814W.gst.matchfake
       12257_NGC2203_F475W-F814W.gst.matchfake
       9891_NGC1795_F555W-F814W.gst.matchfake
### ./obs
       12257_HODGE2_F475W-F814W_gst.fits
       12257_NGC1718_F475W-F814W_gst.fits
       12257_NGC2203_F336W-F475W-F814W_gst.fits
       12257_NGC2213_F475W-F814W_gst.fits
       9891_NGC1644_F555W-F814W_gst.fits
       9891_NGC1795_F555W-F814W_gst.fits
