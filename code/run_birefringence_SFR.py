from makeDelayedRateData import *
from gridded_likelihood import compute_likelihood_grids
import h5py

# Compute merger rate vs. redshift, following Madau+Dickinson
# Note that this is normalized to unity at z=0, as what is important
# here is just the *shape*; the local merger rate is applied later in `compute_likelihood_grids`
rateData = generateTimeDelayData()
zs_merger = rateData['zs']
alpha = 2.6 #2.7
beta = 6.2 #5.6
zpeak = 2.2 #1.9
dRdV_SFR = np.power(1.+zs_merger,alpha)/(1.+np.power((1.+zs_merger)/(1.+zpeak),beta))
dRdV_SFR /= dRdV_SFR[0]

# Define clipping function to skip likelihood calculations over gridpoints which we know
# in advance will return loglikelihood = -inf.
def toClip(kd,kz):
    if kz<-0.4-2.*kd or kz>0.4-2.*kd or kz<-0.15+3.*(kd-0.5) or kz>0.15+3.*(kd+0.5):
        return True
    else:
        return False

# Compute likelihood over grid of kappa_dc and kappa_z values
results = compute_likelihood_grids(zs_merger,dRdV_SFR,clippingFunction=toClip,kappaGridSize=400)

# Create hdf5 file and write posterior samples
hfile = h5py.File('./../data/fixed_rate_SFR.hdf','w')
posterior = hfile.create_group('result')
for key,val in results.items():
    posterior.create_dataset(key,data=val)

# Add some metadata
hfile.attrs['Created_by'] = "run_birefringence_SFR.py"
hfile.attrs['Downloadable_from'] = ""
hfile.attrs['Source_code'] = "https://github.com/tcallister/stochastic-birefringence"
hfile.close()

