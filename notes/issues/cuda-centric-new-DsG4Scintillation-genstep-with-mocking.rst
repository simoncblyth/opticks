cuda-centric-new-DsG4Scintillation-genstep-with-mocking
=========================================================

Objective
-----------

Updates to DsG4Scintillation require a new implementation of 
the scintillation genstep GPU generation. Aim to develop that in 
a CUDA-centric fashion, without any OptiX dependency, in preparation 
for the move to OptiX 7 and also as wish to benefit from cuda mocked
CPU development of GPU code.  

What needs to be different
-----------------------------

* remove OptiX dependency as much as possible, pure-CUDA is better
* develop in "many simple header style" that allows "mock CUDA" unit testing 
* only resort to using Thrust when that adds something very significant 


Ingedients : curand + texture + photon gen 
------------------------------------------------

Old Machinery
~~~~~~~~~~~~~~~~

Traditionally used some ancient ugly code developed as a CUDA+OptiX novice

cudarap/cuRANDWrapper 
    managing the curandState files in overly complicated manner

optixrap/ORng 
    populate optix context with curandState loaded from file

optixrap/tests/cu/rngTest.cu 
    optix-entangled kernel calling curand_uniform using ORng.hh

optixrap/tests/rngTest.cc 
    optix-entangled curand generation using OLaunchTest 

optixrap/tests/rngTest.py 
    plotting the uniform randoms 

optixrap/tests/cu/reemissionTest.cu 
    using optix-entagled reemission_lookup.h 

optixrap/tests/reemissionTest.cc 
    using OScintillatorLib.cc 

optixrap/tests/reemissionTest.py 
    plotting GPU texture generated wavelength distributions and comparing to expectation

optixrap/OScintillatorLib 
    brings ggeo/GScintillatorLib into optix context


WIP replacements
~~~~~~~~~~~~~~~~~~~~~~

qudarap/QRng 
    load and uploads curandState and generates test randoms

qudarap/tests/QRngTest.cc
qudarap/tests/QRngTest.sh 
    mimimal dependency curand generation

qudarap/QScintillatorLib ? 
    create and populate pure CUDA reemission texture with icdf from GScintillatorLib 



Pure CUDA reemission texture access ? Investigate how to implement CScintillatorLib
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maybe following the approach of: /Users/blyth/intro_to_cuda/textures/rotate_image/


* http://cuda-programming.blogspot.com/2013/02/texture-memory-in-cuda-what-is-texture.html




