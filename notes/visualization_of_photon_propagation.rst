Notes on the Opticks Visualization of Photon Propagations
==========================================================

Overview of Opticks 
--------------------

Understanding Opticks visualization requires some familiarity with 
how the Opticks optical photon simulation works.
Opticks glues together Geant4 and the NVIDIA OptiX ray tracer, doing
the optical photon generation and propagation entirely on the GPU.
The below physics processes were rather straightforwardly 
ported to CUDA/OptiX programs:          

* G4Scintillation (just photon generation loop + scintillator reemission)
* G4Cerenkov (just photon generation loop)
* G4Absorption
* G4OpBoundary (reflection, refraction for a subset of surface types)


The difficult part was the geometry, this had to be implemented 
from scratch : solving polynomials to get intersects and 
developing a CSG algorithm without the benefit of recursion, 
as recursion is not supported within NVIDIA OptiX intersect programs.
The great benefit of the CSG algorithm is that it allows Opticks to
reproduce the Geant4 intersects without approximation, making it
possible to precisely match Geant4 results. 

Controlling the G4 random number sequence allows a direct match 
to be achieved between the G4+Opticks GPU simulation and G4 simulations.  
This works near perfectly for simple geometries, for more complicated ones 
it is taking quite an effort to get everything to match.


Overview of Opticks Visualization
------------------------------------

OpenGL/CUDA/Thrust/OptiX interop GPU buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In compute mode without visualization OptiX buffers are used, which are faster than OpenGL buffers.
In interop mode for visualization the buffers used by the simulation are actually
OpenGL buffers. OpenGL/CUDA/OptiX interop enables an OpenGL GPU buffer to be written from CUDA/Thrust or OptiX 
and of course directly rendered by OpenGL without bringing the propagation data to the CPU.

Propagation record buffer
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Opticks optical photon simulation records up to 16 step points of the propagation.
The data for each point (position, time, polarization, wavelength, history+material flags)
are domain compressed into 128 bits (2*4 shorts) and slotted into a fixed size GPU record buffer
of dimension for example (3M, 16, 2, 4). The fixed layout is needed for parallel writing.

The 3M photon limitation was imposed while working with a mobile GPU with only 2GB VRAM.  
When working with more VRAM and more performant GPUs increasing the photon limit is 
expected to be straightforward. 
Increasing beyond 16 recorded step points while possible would entail considerable 
code disruption.

Rendering propagations of millions of photons with one OpenGL draw call
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The render uses the entire record buffer for all the millions of photons
in a single OpenGL glDrawArrays call with GL_LINE_STRIP input primitive and the
capability of geometry shaders to change the type of primitive and filter the primitives
passing down the pipe is used to control the render in response to the input event time.
This allowing the event time to be interactively scrubbed backwards and forwards.

The geometry shader receives two of the recorded step points across the
entire fixed size buffer of eg 3M*16 step points. Whether to emit a primitive
for this pair of step points is decided based on the times of the points
and of the input event time. For example the invalid point pairs
between one photons points and the nexts or between the last point of a propagation and the
unwritten next point are skipped just based on the times.

This approach enables a single draw call to render the entire propagation of millions
of photons.


Geometry Shaders and Renderers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For details of the geometry shaders and renderers see *oglrap/gl/index.rst*





