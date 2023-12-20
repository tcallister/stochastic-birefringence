#!/bin/bash

# Cross-correlation data for each baseline and observing run
curl -o H1L1_O1.dat https://dcc.ligo.org/public/0169/G2001287/005/C_O1.dat
curl -o H1L1_O2.dat https://dcc.ligo.org/public/0169/G2001287/005/C_O2.dat
curl -o H1L1_O3.dat https://dcc.ligo.org/public/0169/G2001287/005/C_O3_HL.dat
curl -o H1V1_O3.dat https://dcc.ligo.org/public/0169/G2001287/005/C_O3_HV.dat
curl -o L1V1_O3.dat https://dcc.ligo.org/public/0169/G2001287/005/C_O3_LV.dat

# Anticipated design-sensitivity PI curve computed by the LVK
curl -o Design_HLV_flow_10.txt https://dcc.ligo.org/public/0169/G2001287/005/PICurve_HLV_Design.dat
