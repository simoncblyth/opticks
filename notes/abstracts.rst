Opticks Abstracts
==================

CHEP 2018
----------


Short Version
~~~~~~~~~~~~~~~~

**Opticks : GPU Optical Photon Simulation for Particle Physics with NVIDIA OptiX**

Opticks is an open source project that integrates the NVIDIA OptiX 
GPU ray tracing engine with Geant4 toolkit based simulations.
Massive parallelism brings drastic performance improvements with  
optical photon simulation speedup expected to exceed 1000 times Geant4 
with workstation GPUs. 

Optical physics processes of scattering, absorption, reemission and 
boundary processes are implemented as CUDA OptiX programs based on the Geant4
implementations. Wavelength dependent material and surface properties as well as  
inverse cumulative distribution functions for reemission are interleaved into 
GPU textures providing fast interpolated property lookup or wavelength generation.
OptiX handles the creation and application of a choice of acceleration structures
such as boundary volume hierarchies and the transparent use of multiple GPUs. 

A major recent advance is the implementation of GPU ray tracing of  
complex constructive solid geometry shapes,  enabling
automated translation of Geant4 geometries to the GPU without approximation.
Using common initial photons and random number sequences allows 
the Opticks and Geant4 simulations to be run point-by-point aligned.
Aligned running has reached near perfect equivalence with test geometries.


Long Version
~~~~~~~~~~~~~~

**Opticks : GPU Optical Photon Simulation for Particle Physics with NVIDIA OptiX**

Opticks is an open source project that integrates the NVIDIA OptiX 
GPU ray tracing engine with Geant4 toolkit based simulations.
Massive parallelism brings drastic performance improvements with  
optical photon simulation speedup expected to exceed 1000 times Geant4 
when using workstation GPUs. Optical photon simulation time becomes 
effectively zero compared to the rest of the simulation.

Optical photons from scintillation and Cherenkov processes
are allocated, generated and propagated entirely on the GPU, minimizing 
transfer overheads and allowing CPU memory usage to be restricted to
optical photons that hit photomultiplier tubes or other photon detectors.
Collecting hits into standard Geant4 hit collections then allows the 
rest of the simulation chain to proceed unmodified.

Optical physics processes of scattering, absorption, reemission and 
boundary processes are implemented as CUDA OptiX programs based on the Geant4
implementations. Wavelength dependent material and surface properties as well as  
inverse cumulative distribution functions for reemission are interleaved into 
GPU textures providing fast interpolated property lookup or wavelength generation.
Geometry is provided to OptiX in the form of CUDA programs that return bounding boxes 
for each primitive and single ray geometry intersection results. 
OptiX handles the creation and application of a choice of acceleration structures
such as boundary volume hierarchies and the transparent use of multiple GPUs. 

A major recent advance is the implementation of GPU ray tracing of 
complex constructive solid geometry (CSG) shapes. This enables
fully automated translation of Geant4 geometries into GPU 
appropriate forms without approximation, greatly simplifying 
the adoption of Opticks. 

Opticks is validated by comparison with Geant4 simulations. Use of 
common initial photons and random number sequences allows 
the two simulations to be run point-by-point aligned.
Aligned running has reached near perfect equivalence with 
test geometries, for larger geometries it provides a powerful 
debugging tool.

OptiX supports interoperation with OpenGL and CUDA Thrust that has enabled 
unprecedented visualisations of photon propagations to be developed 
using OpenGL geometry shaders to provide interactive time scrubbing and    
CUDA Thrust photon indexing to provide interactive history selection. 



CHEP 2016
-----------

**Opticks : GPU Optical Photon Simulation for Particle Physics with NVIDIA OptiX**

Opticks is an open source project that integrates the NVIDIA OptiX 
GPU ray tracing engine with Geant4 toolkit based simulations.
Massive parallelism brings drastic performance improvements with  
optical photon simulation speedup expected to exceed 1000 times Geant4 
when using workstation GPUs. Optical photon simulation time becomes 
effectively zero compared to the rest of the simulation.

Optical photons from scintillation and Cherenkov processes
are allocated, generated and propagated entirely on the GPU, minimizing 
transfer overheads and allowing CPU memory usage to be restricted to
optical photons that hit photomultiplier tubes or other photon detectors.
Collecting hits into standard Geant4 hit collections then allows the 
rest of the simulation chain to proceed unmodified.

Optical physics processes of scattering, absorption, reemission and 
boundary processes are implemented as CUDA OptiX programs based on the Geant4
implementations. Wavelength dependent material and surface properties as well as  
inverse cumulative distribution functions for reemission are interleaved into 
GPU textures providing fast interpolated property lookup or wavelength generation.

Geometry is provided to OptiX in the form of CUDA programs that return bounding boxes 
for each primitive and single ray geometry intersection results. Some critical parts 
of the geometry such as photomultiplier tubes have been implemented analytically 
with the remainder being tesselated. 
OptiX handles the creation and application of a choice of acceleration structures
such as boundary volume heirarchies and the transparent use of multiple GPUs. 

OptiX supports interoperation with OpenGL and CUDA Thrust that has enabled 
unprecedented visualisations of photon propagations to be developed 
using OpenGL geometry shaders to provide interactive time scrubbing and    
CUDA Thrust photon indexing to provide interactive history selection. 


GTC 2016
-----------

Session Description
~~~~~~~~~~~~~~~~~~~~~

**Opticks : Optical Photon Simulation for High Energy Physics with OptiX**

Opticks is an open source project that brings NVIDIA OptiX ray tracing 
to existing Geant4 toolkit based simulations.
Advantages of separate optical photon simulation and    
the approaches developed to integrate it with the general Geant4
particle simulation are presented. Approaches to minimize overheads
arising from split are shown.
Challenges included bringing complex CSG geometries with wavelength
dependent material and surface properties to the GPU.
Techniques for visualisation of photon propagations with
interactive time scrubbing and history selection using OpenGL/OptiX/Thrust
interop and geometry shaders are described.
Results and demonstrations are shown for the photomultiplier based 
Daya Bay and JUNO Neutrino detectors. 


Extended Abstract and Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

http://simoncblyth.bitbucket.org/env/presentation/optical_photon_simulation_with_nvidia_optix.html

As a member of the Daya Bay and JUNO collaborations I encountered the 
bottleneck of Geant4 optical photon processing, taking ~95% of CPU time.
Geant4 is a 20 year old project that simulates all particles as they travel through matter, 
it is used by almost all High Energy Physics detectors.

https://bitbucket.org/simoncblyth/g4dae

To enable the use of GPUs to avoid this bottleneck I developed G4DAE, 
which exports triangulated detector geometries into COLLADA DAE files, 
together with  material/surface properties as a function of wavelength. 
I presented G4DAE at the 19th Geant4 Collaboration Meeting in September 2014. 
The Geant4 Collaboration accepted my proposal to contribute the G4DAE exporter 
to Geant4 and we plan to include it with the 2015 Geant4 release.

Some aspects of my work to adopt OptiX:

* transporting generation steps of the primary particles from Geant4
  allows scintillation and Cerenkov photons to be generated
  within OptiX, avoiding the need to transport photons

* optical physics of reflection, refraction, scattering, absorption
  and reemission from scintillators have been ported to OptiX programs
  based on Geant4 and Chroma implementations

* material/surface properties as a function of wavelength and reemission CDFs
  are encoded into textures, providing fast interpolated access

* compression is used to record photon steps, which are visualized
  using OpenGL geometry shaders which use an input time
  to find the corresponding steps and interpolate the photon positions

* up to 16 steps of photon material/flag histories are encoded into big ints
  which are indexed using Thrust sparse histogram techniques
  and the indices are used from geometry shaders to select photons based on
  flag/material histories

* photons that reach photomultiplier tubes are termed hits, just these
  need to be returned to the host and back to Geant4 : using Thrust
  stream compaction

* JUNO detector design has 30k photomultiplier tubes of two
  types, instancing has been used in OpenGL and OptiX to enable this
  geometry to be loaded onto the GPU.

* work on moving to analytic instead of triangulated geometry definition
  of the photomultiplier tubes is ongoing, for improved performance and realism.


https://bitbucket.org/simoncblyth/env

My developments are currently housed in multiple packages
within my bitbucket "env" repository.
I plan to create a separate repository named Opticks to house these,
to act as a focal point and ease understanding.

I expect my work on Opticks to transition from mainly development
to mainly validation against Geant4 over the next months.
Fortunately achieving significant performance improvements seems inevitable.
The main remaining work is to achieve a match with the Geant4 simulations.





