npy-src(){      echo opticksnpy/npy.bash ; }
npy-rel(){      echo opticksnpy ; }
npy-src(){      echo opticksnpy ; }
npy-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(npy-src)} ; }
npy-vi(){       vi $(npy-source) ; }
npy-env(){      olocal- ; opticks- ; }
npy-usage(){ cat << EOU

NPY : C++ array manipulation machinery inspired by NumPy
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

ViewNPY
    Weak holder of a single NPY* instance constituent, 
    together with offset, strides and size to identify 
    a subset of the data. Also provides bounds finding
    typically useful with geometrical data. 

    Used by oglrap-/Rdr to: 
  
    * turn NPY into OpenGL buffer objects
    * turn VecNPY into glVertexAttribPointer, allowing OpenGL
      drawing of the data

MultiViewNPY
    A list of ViewNPY with name and index access.
    All the ViewNPY are constrained to 
    refer to the same NPY array, 

Lookup
    Creates material code translation lookup tables from 
    material name to code mappings loaded from json files.

numpy
    somewhat modified Open Source numpy.hpp that 
    provides persistency of NPY instances in "NPY" serialization format, 
    the standard NumPy array serialization allowing loading from python/ipython with::

         import numpy as np
         a = np.load("/path/to/name.npy")

GLMPrint
    collection of print functions for various GLM vector and matrix types



EOU
}

npy-notes(){ cat << \EON


Peeking at files
-----------------

::

    simon:rxtorch blyth$ xxd -l 96 -- "-5.npy"
    0000000: 934e 554d 5059 0100 5600 7b27 6465 7363  .NUMPY..V.{'desc
    0000010: 7227 3a20 273c 6932 272c 2027 666f 7274  r': '<i2', 'fort
    0000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    0000030: 652c 2027 7368 6170 6527 3a20 2831 3030  e, 'shape': (100
    0000040: 3030 3030 2c20 3130 2c20 322c 2034 292c  0000, 10, 2, 4),
    0000050: 207d 2020 2020 2020 2020 2020 2020 200a   }             .
    simon:rxtorch blyth$ 

    simon:rxtorch blyth$ xxd -l 96 -- "5.npy"
    0000000: 934e 554d 5059 0100 5600 7b27 6465 7363  .NUMPY..V.{'desc
    0000010: 7227 3a20 273c 6932 272c 2027 666f 7274  r': '<i2', 'fort
    0000020: 7261 6e5f 6f72 6465 7227 3a20 4661 6c73  ran_order': Fals
    0000030: 652c 2027 7368 6170 6527 3a20 2831 3030  e, 'shape': (100
    0000040: 3030 3030 302c 2032 2c20 3429 2c20 7d20  00000, 2, 4), } 
    0000050: 2020 2020 2020 2020 2020 2020 2020 200a   





EON
}


npy-sdir(){ echo $(opticks-home)/opticksnpy ; }
npy-tdir(){ echo $(opticks-home)/opticksnpy/tests ; }
npy-idir(){ echo $(opticks-idir) ; }
npy-bdir(){ echo $(opticks-bdir)/$(npy-rel) ; }

npy-cd(){   cd $(npy-sdir)/$1 ; }
npy-scd(){  cd $(npy-sdir); }
npy-tcd(){  cd $(npy-tdir); }
npy-icd(){  cd $(npy-idir); }
npy-bcd(){  cd $(npy-bdir); }


npy-name(){ echo NPY ; }
npy-tag(){  echo NPY ; }

npy-apihh(){  echo $(npy-sdir)/$(npy-tag)_API_EXPORT.hh ; }
npy---(){     touch $(npy-apihh) ; npy--  ; }


npy--(){      opticks--     $(npy-bdir) $* ; }
npy-t() {     opticks-t $(npy-bdir) $* ; }
npy-genproj(){ npy-scd ; opticks-genproj $(npy-name) $(npy-tag) ; }
npy-gentest(){ npy-tcd ; opticks-gentest ${1:-NExample} $(npy-tag) ; }
npy-txt(){     vi $(npy-sdir)/CMakeLists.txt $(npy-tdir)/CMakeLists.txt ; }



npy-i(){ npy-scd ; i ; }

npy-bindir(){ echo $(npy-idir)/bin ; } 
npy-bin(){    echo $(npy-bindir)/$1 ; } 

npy-wipe(){
   local bdir=$(npy-bdir)
   rm -rf $bdir
}
