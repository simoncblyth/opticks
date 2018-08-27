Reconstruction
==================


Quotes from : Chroma: Reconstruct like it’s 2099
-----------------------------------------------------

* ~/opticks_refs/Chroma-ProjectX-2099.pdf

Stan Seibert, Anthony LaTorre 
University of Pennsylvania
PXPS, June 18, 2012


* A sufficiently fast Monte Carlo is indistinguishable from a maximum
  likelihood reconstruction algorithm.  (Tongue in cheek)

* Minor Challenge: Minimization in a Stochastic Space


* Chroma right now is a great photon Monte Carlo and a really fiddly reconstruction tool.

* Stochastic minimizer is incredibly simple and fragile. 

  * Need more function evaluations to do smarter things, but likelihood function takes too long.

* Need 10-100x speed improvement before Chroma can be easily used as an event fitter. 

* Have ideas for where to get some of this performance.

* Should explore other PDF estimation techniques beyond the N’th nearest neighbor method used now.


My Thoughts On Chroma
~~~~~~~~~~~~~~~~~~~~~~~~~

* suspect the title may be realistic regards when such recon becomes possible
* but I have no experience of stochastic minimization 

* :google:`PDF estimation k-nn` 

* https://www.ssc.wisc.edu/~bhansen/718/NonParametrics10.pdf




Solving the Rendering Equation
---------------------------------

* :google:`global illumination solving rendering equation`

Huge amounts of effort has gone into optimizing path tracing

* photon maps using kd trees 
* metropolis sampling 
* MCMC


Robust Monte Carlo Methods for Light Transport Simulation, Veach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* ~/opticks_refs/veach_thesis.pdf 
* http://graphics.stanford.edu/papers/veach_thesis/

* Metropolis


Can more continuous approaches help
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Monte Carlo Ray Tracing V Metropolis Algorithm and Photon Mapping
April 8, 2005

* http://web.cse.ohio-state.edu/~parent.1/classes/782/Papers/MCalgorithms.pdf



Monte Carlo : take a step back 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://en.wikipedia.org/wiki/Monte_Carlo_method





