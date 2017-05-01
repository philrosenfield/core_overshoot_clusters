# Posterior distribution files

These `csv` files were created from post processing output from MATCH  (closed source software).

These data tables are used to create Figures 7-12 (see `..code/pdfs.py`) and Tables 6 and 7.

Columns (Sorry, they are not always in the same order, submit a PR if this bugs you):

    Av : extinction (mag)
    IMF : initial mass function slope (Gamma)
    dmod : true distance modulus
    lage : log Age (yr)
    logZ : log metallicity
    fit : raw value of - 2 ln Probability
    sfr : star formation rate (Msun/yr)
    bf : binary fraction
    ssp : index (used to track calculations on super computer)
    ov : core overshooting value
    vstep : Hess diagram y-axis bin size (magnitude)
    dav : Av bin size
    vistep : Hess diagram x-axis bin size (color)
    tbin : age bin
    trueov : for mock_full.csv core overshooting value of mock data
## Directory listing
    HODGE2_full.csv
    NGC1644_full.csv
    NGC1718_full.csv
    NGC1795_full.csv
    NGC2203_full.csv
    NGC2213_full.csv
    mock_full.csv
