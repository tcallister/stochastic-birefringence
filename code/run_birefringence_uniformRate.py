from makeDelayedRateData import *
from gridded_likelihood import compute_likelihood_grids
import h5py

# Compute (trivial) merger rate vs. redshift
# Note that this is normalized to unity at z=0, as what is important
# here is just the *shape*; the local merger rate is applied later in `compute_likelihood_grids`
rateData = generateTimeDelayData()
zs_merger = rateData['zs']
dRdV_uniform = np.ones(zs_merger.size)
dRdV_uniform[zs_merger>6] = 0

# Define clipping function to skip likelihood calculations over gridpoints which we know
# in advance will return loglikelihood = -inf.
def toClip(kd,kz):
    if kz<-0.4-2.*kd or kz>0.4-2.*kd or kz<-0.6+3.*(kd-0.5) or kz>0.6+3.*(kd+0.5):
        return True
    else:
        return False

# Compute likelihood over grid of kappa_dc and kappa_z values
results = compute_likelihood_grids(zs_merger,dRdV_uniform,clippingFunction=toClip,kappaGridSize=400)

# Create hdf5 file and write posterior samples
hfile = h5py.File('./../data/fixed_rate_uniform.hdf','w')
posterior = hfile.create_group('result')
for key,val in results.items():
    posterior.create_dataset(key,data=val)

# Add some metadata
hfile.attrs['Created_by'] = "run_birefringence_uniformRate.py"
hfile.attrs['Downloadable_from'] = "https://zenodo.org/doi/10.5281/zenodo.10384998"
hfile.attrs['Source_code'] = "https://github.com/tcallister/stochastic-birefringence"
hfile.close()

