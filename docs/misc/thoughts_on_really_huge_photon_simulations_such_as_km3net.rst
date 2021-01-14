thoughts_on_really_huge_photon_simulations_such_as_km3net
===========================================================

Hi Tao, 

> Fan Hu, a PhD student from PKU, is working with Donglian Xu from SJTU on the
> simulation of neutrino telescope in the deep sea. He is interested in your work
> and hope to invite you to give an online seminar.  Do you have time? If you
> have time, I will tell them.  BTW: the LHAASO experiment is also interested in
> your Opticks. Maybe we could arrange another seminar at IHEP.  Tao

I think a small “workshop” type meeting bringing together the people who are actually 
working on simulation from KM3Net, LHAASO etc.. would be more productive than just 
giving my seminar again to a general audience.
For better communication everyone attending should give a short presentation 
on how they currently simulate and how they would like to use Opticks 
or similar to accelerate it.

Optical simulation for deep underwater neutrino telescopes 
like KM3Net or Baikal GVD inevitably needs to take an indirect approach
due to the extreme numbers of photons being impossible to store.
Instead of storing photons I guess it will be necessary 
to develop a way to progressively accumulate into data structures 
such as progressive photon maps (eg kd-tree based)
or light fields at chosen positions relative to the cosmics 
and the strings of PMTs.

The Opticks approach could be adapted to accumulating into such 
data structures. Basically instead of collecting 
photon parameter "samples" binned probability distributions 
are collected.

Designing the light field/photon map data structure and 
a way to accumulate into it and demonstrating that it can answer the 
physics questions that need to be answered are the major requirements.

I would start by searching for recent developments from
the graphics community in such data structures. 

For an introduction to global illumination and photon mapping 
For background I recommend a classic book :

   "Realistic Image Synthesis Using Photon Mapping"
   Henrik Wann Jensen
   http://graphics.stanford.edu/~henrik/papers/book/

However the static photon map (using a kd-tree) described 
is probably not the thing to do. 
Instead investigate "progressive photon mapping"
techniques from the graphics community.  
Computer vision research has developed light field structures 
that might also be worth investigating.

One paper that describes progressive photon mapping:
   https://www.sciencedirect.com/science/article/pii/S0038092X15000559

The thesis of Eric Veach 
   Robust Monte Carlo Methods for Light Transport Simulation 
   http://graphics.stanford.edu/papers/veach_thesis/

is a good starting point for getting familiar with 
graphics community developments in light transport and getting
used to their terminology, eg "bi-directional path tracing"
and "global illumination".

Simon
