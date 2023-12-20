#!/bin/bash

# Download and unzip
curl https://zenodo.org/records/10384998/files/stochastic-birefringence-data.zip --output "stochastic-birefringence-data.zip"
unzip stochastic-birefringence-data.zip

# Move input data to ../input/
mv matlab_orfs.dat ../input/
mv o1o2o3_mass_c_iid_mag_iid_tilt_powerlaw_redshift_result.json ../input/

# Remove original zip files and annoying MAC OS directory
rm stochastic-birefringence-data.zip
rmdir __MACOSX/

# Move to input/ directory and execute two other data download scripts
cd ../input/
. download_cross_correlation_data.sh 
. download_sensitivity_curves.sh
cd ../data/
