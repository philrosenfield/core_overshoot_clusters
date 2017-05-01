# PARSEC V1.2S Core overshooting grid and other stellar modeling groups tracks.

These files are used to create Figures 4, 5, and 13.

    track_summary.dat: a summary of the PARSEC core overshooting grid
  Columns:

    HB : is a horizontal branch track (1 yes, 0 no; age starts at 0 on the HB track)
    Z : metallicity
    M : initial mass
    OV : core overshooting value
    QHEL : mass fraction of helium
    tau_He : He-burning lifetime
    tau_H : H-burning lifetime
    fname : filename

## Track listing

The columns are different for different modeling groups, please see the file headers. All files are ascii.

Note: Victoria Regina and YaPSI models were transformed to the acs-wfc filterset using the same bolometric corrections as the PARSEC tracks.

### ./OV0.3_Z0.006_Y0.259
    match_Z0.006Y0.259OUTA1.74_F7_M1.50.HB.dat.acs_wfc
    match_Z0.006Y0.259OUTA1.74_F7_M1.50.dat.acs_wfc
    match_Z0.006Y0.259OUTA1.74_F7_M2.00.dat.acs_wfc
### ./OV0.4_Z0.006_Y0.259
    match_Z0.006Y0.259OUTA1.74_F7_M1.50.HB.dat.acs_wfc
    match_Z0.006Y0.259OUTA1.74_F7_M1.50.dat.acs_wfc
    match_Z0.006Y0.259OUTA1.74_F7_M2.00.dat.acs_wfc
### ./OV0.5_Z0.006_Y0.259
    match_Z0.006Y0.259OUTA1.74_F7_M1.50.HB.dat.acs_wfc
    match_Z0.006Y0.259OUTA1.74_F7_M1.50.dat.acs_wfc
    match_Z0.006Y0.259OUTA1.74_F7_M2.00.dat.acs_wfc
### ./OV0.6_Z0.006_Y0.259
    match_Z0.006Y0.259OUTA1.74_F7_M1.50.HB.dat.acs_wfc
    match_Z0.006Y0.259OUTA1.74_F7_M1.50.dat.acs_wfc
    match_Z0.006Y0.259OUTA1.74_F7_M2.00.dat.acs_wfc
### ./dartmouth ([link](http://stellar.dartmouth.edu/%7Emodels/fehp00afep0.html))
    HB_Z0.006_Y0.254_M1.50.dat
    Z0.006_Y0.254_M2.0.dat
    Z0.006_Y0.254_M1.50.dat
### ./mist ([link](http://waps.cfa.harvard.edu/MIST/interp_tracks.html))
    Z0.00581867_Y0.2577_M1.5.dat
    Z0.00581867_Y0.2577_M2.0.dat
### ./vr ([link: we downloaded `vr0a.tar.Z`](http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/community/VictoriaReginaModels//VBD2005/))
    Z0.006_Y0.247_M1.5.dat
    Z0.006_Y0.247_M2.0.dat
### ./yy ([link](http://vo.aip.de/yapsi/download_grids.html#2016_release))
    Z0.005416_Y0.25_M1.5.dat
    Z0.005416_Y0.25_M2.0.dat
