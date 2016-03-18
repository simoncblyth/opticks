Production Requirements
========================


My experience is mainly with development, where the requirements are clear: 

* direct access to Linux machines with a recent NVIDIA GPU (Maxwell architecture) 
* ability to update to latest drivers for NVIDIA OptiX and to run OpenGL 
  for visualisation/debugging.

Direct access (having the machine physically beside the desk) is really needed 
for development to allow visual debugging. 
There may be ways to allow remote visualisation but getting side tracked on such complications, 
which probably inevitably yield poor visualisation performance seems better left to others.


My validations are progressing well (single PMT is agreeing between Opticks/Geant4)
so I am hopeful of getting full geometries to agree before the summer.  
As such we need to consider/plan for how Opticks can be validated/used in production.
However, I have little experience of production running and none with Phi co-processors.

Opticks is intended to be used in hybrid G4+Opticks fashion but 
we will initially need to run standard G4 as well over the same events, on the same machine,
which will of course be a lot more demanding than the hybrid running (especially CPU memory).
So people with JUNO CPU simulation/production experience can be a better guide 
as to the memory/CPU requirements of large scale G4 running.  

On the GPU side, at least 2 NVIDIA GPUs per machine 
(with same requirements as for development) are required with memory 
of at minimum 4GB per GPU.  Using more than one GPU allows scaling to 
be evaluated : prior measurements suggest near linear OptiX performance 
scaling with CUDA cores, but this will always need to be checked again
on every system used.

My measurements/extrapolations suggest that the time for GPU 
optical photon processing is going to be effectively zero compared to CPU processing,
with any modern workstation NVIDIA GPU being capable of reaching this level.

This would suggest that maximal throughput can be obtained by focussing resources 
more on the CPU side than the GPU side, as that is where I predict the bottleneck will be.

Given this, the NVIDIA VCA machines aimed at rendering companies (advertising, design etc..) 
seem inappropriate for JUNO simulation as they are GPU resource heavy.  

     http://www.nvidia.com/object/visual-computing-appliance.html
     http://www.nvidia.com/object/vca-for-optix.html

Future developments such as development of GPU reconstruction techniques may shift 
the balance however. 

