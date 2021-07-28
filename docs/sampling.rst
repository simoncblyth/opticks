sampling
==========


* https://en.wikipedia.org/wiki/Inverse_transform_sampling
* 


CK rejection sampling::



    580     double nMax = 1.793 ;
    581     double maxCos = BetaInverse / nMax;
    582     double maxSin2 = ( 1. - maxCos )*( 1. + maxCos );
    ...
    594     do {
    596         u0 = curand_uniform_double(&rng) ;
    598         energy = Pmin + u0*(Pmax - Pmin) ;
    600         sampledRI = prop->interpolate( 0u, energy );
    602         cosTheta = BetaInverse / sampledRI ;
    604         sin2Theta = (1. - cosTheta)*(1. + cosTheta);
    606         u1 = curand_uniform_double(&rng) ;
    608         u_mxs2_s2 = u1*maxSin2 - sin2Theta ;
    610         loop += 1 ;
    612     } while ( u_mxs2_s2 > 0. );

The condition for the sample to be rejected and to keep looping::

   u1*maxSin2 > sin2Theta      

Is that equivalent to rejecting based on cosTheta ?:: 

   u1*maxCos < cosTheta 

Or maybe need to 1-cosTheta (to do this corresponding to zero) ?


* BetaInverse is an input (from the velocity of the particle) 
  making it difficult to imagine how to convert the rejection sampling 
  into anything else

* But if can avoid the non-linearity of the sin2Theta its just a linear factor and cut  

* There can be holes in the allowable energies, is that a problem ? It will mean the 
  cumulative probability will stay constant across the disallowed energy region then
  in inverse the probability of hitting the delta function is zero(?) 


How to map from rejection sampling that can take hundreds of randoms and double precision to a simple single random ICDF lookups using floats ?
--------------------------------------------------------------------------------------------------------------------------------------------------

* https://en.wikipedia.org/wiki/Rejection_sampling

::

    Rejection sampling is based on the observation that to sample a random variable
    in one dimension, one can perform a uniformly random sampling of the
    two-dimensional Cartesian graph, and keep the samples in the region under the
    graph of its density function.[1][2][3] Note that this property can be extended
    to N-dimension functions.


Hmm : this makes me think, can I create an inverse CDF from the RINDEX ?
RINDEX is a piecewise linear function, so an analytic integral can be calculated.
The BetaInverse is a constant factor so that could be kept factored.
Want to have one CDF integral that can be used for all BetaInverse.

The non-linear transform from cosTheta to sin2Theta is problematic ? 
But is it necassary ? Can I not just do the sampling cut on the cosTheta, 
or equivalently do the the integal on the cosTheta ?







