Meetings
=========


HOW2019 (March) : Joint HSF/OSG/WLCG Workshop 
-----------------------------------------------

* https://indico.cern.ch/event/759388/


Neutrino experiments simulation overview : Michael Kirby
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://indico.cern.ch/event/759388/contributions/3331550/

GPUs for IceCube (reconstruction/simulation/deployment) : Benedikt Riedel 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://indico.cern.ch/event/759388/contributions/3303407/


Modernisation of simulation code : Witold Pokorski 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://indico.cern.ch/event/759388/contributions/3331551/


Geant4 and effective use of Accelerators : Philippe Canal
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Geant Exascale Pilot Project

* https://indico.cern.ch/event/759388/contributions/3303061/


ALICE GPU Algorithms : David Rohr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://indico.cern.ch/event/759388/contributions/3303054/
* 90% source sharing between GPU and CPU 

To facilitate software distribution, we have one binary package that contains all versions.

* The GPU versions of the code are contained in special GPU-tracking libraries.
* These GPU-tracking libraries are accessed via dlopen.
* Only the GPU-tracking libraries link to the GPU driver / runtime.
* The tracking software (without GPU acceleration) runs on all compute nodes, irrespective of the presence of a driver.




