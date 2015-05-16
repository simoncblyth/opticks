# === func-gen- : numerics/npy/npy fgp numerics/npy/npy.bash fgn npy fgh numerics/npy
npy-src(){      echo numerics/npy/npy.bash ; }
npy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(npy-src)} ; }
npy-vi(){       vi $(npy-source) ; }
npy-env(){      elocal- ; }
npy-usage(){ cat << EOU

npy : C++ array manipulation machinery inspired by NumPy
==========================================================

Overview
---------

Maybe this package should be split into two portions: 

* application specifics 
* generic functionality 


Dependencies
------------

* Boost 
* GLM headers, matrix/math  


NB no OpenGL dependency, instead classes designed to be 
easily usable from oglrap- when doing things like 
uploading OpenGL buffers etc..


Classes
-------

NumpyEvt
    Holder of several NPY* instance constituents:

    * GenstepData
    * PhotonData

    High level layout specific actions like:

    * determining the number of photons the Genstep data 
      corresponds to and allocating space for them

    * composing the MultiVecNPY addressing into the genstep 
      and photon data with names like "vpos" and "vdir" 
      which correpond to OpenGL shader attribute names

    * provides NPY arrays to optixrap-/OptiXEngine::initGenerate 
      which uploads the data into OptiX GPU buffers


NPY
   Holder of array shape, data and metadata.
   Currently float specific.
   Provides persistency using numpy.hpp, allowing interop
   with real NumPy from python/ipython.
   
   TODO: turn into templated class handling: float, double, int, unsigned int,...   

G4StepNPY
    Weak holder of a single NPY* instance constituent.
    Provides G4Step layout specializations: 

    * dumping 
    * lookups for material code mapping 

VecNPY
    Weak holder of a single NPY* instance consituent, 
    together with offset, strides and size to identify 
    a subset of the data. Also provides bounds finding
    typically useful with geometrical data. 

    Used by oglrap-/Rdr to: 
  
    * turn NPY into OpenGL buffer objects
    * turn VecNPY into glVertexAttribPointer, allowing OpenGL
      drawing of the data

    TODO: rename to ViewNPY

MultiVecNPY
    A list of VecNPY with name and index access.
    All the VecNPY are constrained to 
    refer to the same NPY array, 

    TODO: rename to MultiViewNPY

Lookup
    Creates material code translation lookup tables from 
    material name to code mappings loaded from json files.

numpy
    somewhat modified Open Source numpy.hpp that 
    provides persistency of NPY instances in "NPY" serialization format, 
    the standard NumPy array serialization allowing loading from python/ipython with::

         import numpy as np
         a = np.load("/path/to/name.npy")

stringutil
    string and MD5 digest utils

GLMPrint
    collection of print functions for various GLM vector and matrix types


EOU
}

npy-sdir(){ echo $(env-home)/numerics/npy ; }
npy-idir(){ echo $(local-base)/env/numerics/npy ; }
npy-bdir(){ echo $(local-base)/env/numerics/npy.build ; }

npy-cd(){   cd $(npy-sdir); }
npy-scd(){  cd $(npy-sdir); }
npy-icd(){  cd $(npy-idir); }
npy-bcd(){  cd $(npy-bdir); }

npy-bindir(){ echo $(npy-idir)/bin ; } 

npy-wipe(){
   local bdir=$(npy-bdir)
   rm -rf $bdir
}

npy-cmake(){
   local iwd=$PWD

   local bdir=$(npy-bdir)
   mkdir -p $bdir

   npy-bcd
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(npy-idir) \
       $(npy-sdir)

   cd $iwd
}

npy-make(){
   local iwd=$PWD

   npy-bcd
   make $*

   cd $iwd
}

npy-install(){
   npy-make install
}

npy--()
{
    npy-wipe
    npy-cmake
    npy-make
    npy-install
}

npy-lookup-test()
{
    ggeo-
    $(npy-bindir)/LookupTest $(ggeo-meta-dir)
}

npy-g4stepnpy-test()
{
    ggeo-
    $(npy-bindir)/G4StepNPYTest $(ggeo-meta-dir)
}





