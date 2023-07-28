#!/bin/bash

# A-sharp strain
curl -O https://dcc.ligo.org/public/0186/T2300041/001/Asharp_strain.txt

# Folder with CE strain
curl -O https://dcc.cosmicexplorer.org/public/0163/T2000017/006/ce_strain.zip
unzip ce_strain.zip
rm ce_strain.zip
mv ce_strain/cosmic_explorer_strain.txt ./ 
rm ce_strain/*.txt
rmdir ce_strain/
