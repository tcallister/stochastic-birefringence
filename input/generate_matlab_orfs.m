addpath("/home/thomas.callister/matapps/stochastic/trunk/Utilities/ligotools/matlab/");

L1 = getdetector('LLO');
H1 = getdetector('LHO');
V1 = getdetector('VIRGO');

freqs = 20:0.03125:1800;
orf_H1_L1 = overlapreductionfunction(freqs,H1,L1);
orf_H1_V1 = overlapreductionfunction(freqs,H1,V1);
orf_L1_V1 = overlapreductionfunction(freqs,L1,V1);

out=[transpose(freqs) transpose(orf_H1_L1) transpose(orf_H1_V1) transpose(orf_L1_V1)];     % combined matrix of above vectors
writematrix(out,'output.dat','Delimiter','tab')
