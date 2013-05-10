from __future__ import division


from __future__ import division
import numpy as np
np.seterr(invalid='raise')
from matplotlib import pyplot as plt
import copy

from pybasicbayes import models, distributions
from pybasicbayes.util.text import progprint_xrange

#####################
#  data generation  #
#####################

N = 400
alpha_0=5.0

obs_hypparams=dict(mu_0=np.zeros(2),sigma_0=np.eye(2),kappa_0=0.05,nu_0=5)

component_library = [distributions.Gaussian(**obs_hypparams) for itr in range(30)]
priormodel = models.Mixture(alpha_0=alpha_0,components=component_library)

data = priormodel.rvs(N)

plt.figure()
plt.plot(data[:,0],data[:,1],'kx')
plt.title('data')

###############
#  inference  #
###############

all_likelihoods = models.FrozenMixtureDistribution.get_all_likelihoods(component_library,data)
posteriormodel = models.FrozenMixtureDistribution(
        likelihoods=all_likelihoods,
        alpha_0=alpha_0,
        components=component_library)

for itr in progprint_xrange(50):
    posteriormodel.resample(np.arange(N),niter=5)

plt.show()

