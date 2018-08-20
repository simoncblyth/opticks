rng_aligned_cerenkov_generation
=================================

* Recall gensteps can be regarded as a copy of the "stack" 
  from just before the photon generation loop, with values that 
  include the number of photons to generate, positions etc..

* By input gensteps : I mean Opticks runs with input gensteps, Geant4 
  runs as normal from primaries : so it is consuming RNGs before getting 
  to the photon generation, to for example decide how many photons.  

* contrast with input photons where all RNG consumption on CPU/GPU sides 
  is matched as the bifurcation between the simulations is immediate.

* With input gensteps the bifurcation is in the middle of G4Cerenkov
  at the generation loop : so need to switch the RNG stream from 
  a normal one to an aligned one at each spin of that loop, 
  with the photon index to pick the sub-stream 

* Can use photon index -1 to mean ordinary non-aligned RNG stream 


Shakedown
----------

* :doc:`OKG4Test_direct_two_executable_shakedown`





