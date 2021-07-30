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


* http://patricklam.org/teaching/sampling_print.pdf


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


* kinda did this (numerically and for constant BetaInverse) in ana/rindex.py 
  but the results are very poor... pilot error : must do it with the s2 
  and it gives a good match : chi2/ndf ~ 1.1 



ana/rindex.py cumulative integral for s2 
-------------------------------------------------

::

    BetaInverse = 1.5 
    ri_ = lambda e:np.interp(e, ri[:,0], ri[:,1] )
    rif_ = lambda e:np.interp(e, ri[:,0], ri[:,1] ) - BetaInverse
    xri = find_cross(ri, BetaInverse=BetaInverse)
    nMax = ri[:,1].max() 
    print(xri)

    ct_ = lambda e:BetaInverse/np.interp(e, ri[:,0], ri[:,1] )
    s2_ = lambda e:(1-ct_(e))*(1+ct_(e))

    ed = np.linspace(xri[0],xri[1],40960)   # assuming simple 2 crossings "single peak" RINDEX 
    s2e = s2_(ed)

    cs2e = np.cumsum(s2e)
    cs2e /= cs2e[-1]           # last bin will be max

    look_ = lambda u:np.interp(u, cs2e, ed )     # numerical inversion of CDF 
    u = np.random.rand(1000000)
    l = look_(u)     ## ck energy distrib obtained with single random throw 


But to use this would need to use a 2d texture with one dimension 
being BetaInverse ( or BetaInverse/nMax ? ).
Could repeat the about for 1000 steps across the BetaInverse domain ... 
but theres a variable energy range problem too

::

    cosTheta = BetaInverse/sampleRI 

At the point of maximum RI::

    cosTheta_max = BetaInverse/sampleRI_max


In energy regions where sampleRI < BetaInverse, cosTheta > 1 => no Cerenkov
As lookups always return a value have to avoid that happening : so must 
constrain the domain ?

No, I dont think this is needed the energy is a looked up value from 
a (BetaInverse, u:0->1 ) texture so the permissable energy ranges for each BetaInverse 
are baked into the ICDF

Hmm I can see that interpolation between u points is appropriate but 
what about between BetaInverse points in the other dimension ?


What about disallowed energy ranges from multi-peak rindex ? 

That will lead to flat regions of the CDF that become vertical in the ICDF 
hence the probability of landing there becomes zero.

What about when there is only a tiny energy region left, just a tiny peak of RINDEX 
poking above the BetaInverse sea ?  The find_crossing algorithm defines the energy domain 
of the CDF which becomes the energy value range of the ICDF

find_crossing will need to find all crossings and pick the extreme ones to define 
the ranges to bake into the texture.  

Hmm what about an "inverted" two lobe RINDEX situation ?



Why use s2: sin^2(th) rather than ct or 1-ct ? Perhaps because of sin^2(th) range 0->1 ?
---------------------------------------------------------------------------------------------------------------------------

::

    sin^2(th) == 1 - cos^2(th) = (1 - cos(th)) (1 + cos(th))


    Hmm can trig identities be used ?

    sin^2(th) =  1 - cos(2x)
                --------------
                     2

::

    double reject ;
    double u0 ;
    double u1 ; 
    double energy ; 

    double maxCos = BetaInverse / nMax;
    double maxSin2 = ( 1. - maxCos )*( 1. + maxCos );
    double cosTheta ;
    double sin2Theta ;

    //double oneMinusCosTheta ;

    unsigned loop = 0u ; 

    do {
        u0 = curand_uniform_double(&rng) ;
        u1 = curand_uniform_double(&rng) ;
        energy = Pmin + u0*(Pmax - Pmin) ; 
        sampledRI = prop->interpolate( 0u, energy );  
        //oneMinusCosTheta = (sampledRI - BetaInverse) / sampledRI ; 
        //reject = u1*maxOneMinusCosTheta - oneMinusCosTheta ;
        loop += 1 ; 

        cosTheta = BetaInverse / sampledRI ;
        sin2Theta = (1. - cosTheta)*(1. + cosTheta);  
        reject = u1*maxSin2 - sin2Theta ;

    } while ( reject > 0. );





