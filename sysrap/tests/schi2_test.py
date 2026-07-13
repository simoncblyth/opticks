#!/usr/bin/env python


from scipy.stats import chi2 as _chi2


c2obs = 5.41
ndf = 2.0
p_value = 1 - _chi2.cdf(c2obs, ndf)

print(p_value)




