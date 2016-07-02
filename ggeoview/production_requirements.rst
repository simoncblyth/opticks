Production Requirements
========================

Choosing a NVIDIA Product Line
---------------------------------

Regards choosing NVIDIA GPUs to purchase. The first 
thing is to pick an appropriate product line,  

GeForce 
    consumer/gaming 
Quadro   
    workstation CAD/3D modelling 
Tesla      
    compute, FP64   

As OptiX/Opticks doesn’t benefit from double precision performance 
the much cheaper consumer Geforce is appropriate.


Choosing GeForce GPU
-----------------------

My suggestions for Geforce GPUs in descending order:


Pascal   
~~~~~~~

https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#GeForce_10_Series

GeForce GTX 1080  (8 GB)    599 USD    available May 27 
GeForce GTX 1070  (8 GB)    379 USD    available June 10 

http://www.anandtech.com/show/10304/nvidia-announces-the-geforce-gtx-1080-1070

These were announced May 6th, 2016.
Pascal uses a 16nm process, prior architectures (Kepler, Maxwell) have been at 28 nm 
since 2012, so this is a major development and will probably mean a 
big jump in performance and power efficiency.

Maxwell  
~~~~~~~
     
https://en.wikipedia.org/wiki/List_of_Nvidia_graphics_processing_units#GeForce_900_Series

GeForce GTX 980 Ti  (6 GB)   640 USD   (price will soon drop due to Pascal release)



Opticks/OptiX requirements
-----------------------------

Opticks/OptiX doesn’t need Pascal performance (Maxwell would do fine) 
but looking ahead to “OpticksVR” : with JUNO geometry extreme 
performance and lots of fast memory becomes necessary to achieve the 90fps needed for VR.

My measurements/extrapolations suggest that the time for GPU 
optical photon processing is going to be effectively zero compared to CPU processing,
with any Maxwell or Pascal GeForce GPU being capable of reaching this level.
So multiple GPU machines are not necessary.  

Although I would like to test with a multi-GPU machine I think IHEP will likely have such systems 
that can be borrowed for that.

Opticks is intended to be used in hybrid G4+Opticks fashion but 
we will initially need to run standard G4 as well over the same events, on the same machine,
which will of course be a lot more demanding than the hybrid running (especially CPU memory).
So Tao who has JUNO CPU simulation/production experience can be a better guide 
as to the memory/CPU requirements of G4 running.  

This would suggest that maximal throughput can be obtained by focussing resources 
more on the CPU side than the GPU side, as that is where I predict the bottleneck will be. 

Opticks currently requires Linux/Mac, but VR will probably only be supported on Windows 
for several years.  So I would suggest a dual boot Windows/Linux machine, 
although I do not have experience with dual boot systems.



Overview
---------


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

