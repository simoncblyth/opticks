deciding_to_develop_a_gpu_accelerated_simulation
=================================================

::

    > I'm ... from .... I'm intrested in the GPU accelerate for
    > the tranditional programing.I have a requirements to accelerate the avalanche
    > for gaseous detector taken from Garfield++. I have attached part of the code
    > for the avalanche. The source code is here. Do you think it's possible and
    > worthy to implement an GPU version?
    >
    > Looking forward for your reply. Thank you very much!

To determine if something is worthwhile to attempt GPU parallelization
requires prototype development and performance measurements with
realistically sized problem sets. 

Also the bigger picture of how important something is to accelerate and 
how that fits in with other processing are inputs to how much effort it is 
worthwhile to expend on acceleration. 

Moving things to the GPU requires you to understand and port 
everything that is happening down to first principals you do not have 
the luxury of bringing in libraries and hiding details.

There are however some shortcuts available by using GPU textures 
to capture CPU side processing into 1/2/3d data structures with hardware 
accelerated interpolated look ups. This for example can be used to 
lookup field values/vectors or material properties as a function of energy.

The code that you pointed at:

https://gitlab.cern.ch/garfield/garfieldpp/-/blob/master/Source/AvalancheMicroscopic.cc#L485

uses many helper objects and uses shuffling between stacks.  This kind of 
thing does not translate easily to the GPU. Thus as is typically the case 
it will be necessary to totally reimplement both data structures and algorithm 
to benefit from GPU parallelism.

It is necessary to think in terms of fixed size arrays that GPU kernels 
operate upon. Changing float values for positions, times etc.. and bitfield flags 
to indicate the state of the particles represented. 
As there is no control over the order of GPU thread processing it is necessary 
for each GPU thread to only write to its own "slots" in the arrays. 

For example modelling an avalanche or shower with secondaries requires the fixed array 
size to start with many empty slots. 
For example if each particle could only ever yield a maximum of two additional 
particles at each iteration then the fixed array size for a "stack" that starts with 
N particles would need 3*N entries with threads i=0,1,2,..,N-1 being for example 
owner of slots i,i+N,i+2*N in the fixed array. Where the last 2/3 of the 
array starts empty.

At each iteration the technique of stream compaction can be used
to reposition the non-dead particles together down into the first third 
by copying back and forth between two GPU buffers (A and B) that keep getting 
resized. 

The skeleton of the steps of the approach could be:

0. load/generate initial particles
1. perform transport iteration, some slots will die and change flags 
   to reflect that, some slots will yield secondaries writing into i+N, i+N*2  
2. scan GPU buffer A counting particles that are still alive (CUDA Thrust can conveniently do this)  
3. resize GPU buffer B with the appropriate multiplier of spare slots to hold future secondaries
4. use CUDA thrust copy_if (or something similar) to copy from A->B filling the first third of B 
5. check for some termination condition 
6. repeat 1-6 with buffers A and B interchanged  

This approach seems like it can be a workable way to simulate particle 
showers however to prevent thread divergence killing performance it 
will probably be necessary to use binning approaches in particle type, 
energy, position, or combinations of those to try to make 
the processing of each thread as similar as possible.

Simon
