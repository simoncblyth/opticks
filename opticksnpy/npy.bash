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


Disentangling DualContouringSample (DCS)
-------------------------------------------

ImplicitMesher (IM) avoids the user having to 
be concerned with grid coordinates... can the 
same thing be done with DCS

ImplicitMesher
~~~~~~~~~~~~~~~~~~~

Grid cubesize p->size obtained from bounds and resolution::

     492 polygonizer * MakePolygonizer( ImplicitPolygonizer * wrapper,
     493                aabb * boundingbox,
     494                int grid_resolution,
     495                unsigned int convergence,
     496                int verbosity)
     497 {
     498     polygonizer * p;
     ...
     508     float center[3] = {0,0,0};
     509     aabb_centroid(boundingbox, center);
     510     float cubesize = aabb_avgcubesize(boundingbox, grid_resolution);
     511 
     512     iaabb bounds;
     513     aabb_to_iaabb(boundingbox, cubesize, &bounds);
     514 
     515     p->center.x = center[0]; p->center.y = center[1], p->center.z = center[2];
     ...
     518     iaabb_copy(&bounds, &(p->bounds));
     519 
     520     p->size = cubesize;
     521     p->convergence = convergence;
     522     p->verbosity = verbosity ;


World positions obtained from ijk with the p->size used for the Value calls::

     774     c = ALLOC_CORNER(p);
     ...
     776     c->i = i;
     777     c->x = p->center.x+((float)i-.5f)*p->size;
     778     c->j = j;
     779     c->y = p->center.y+((float)j-.5f)*p->size;
     780     c->k = k;
     781     c->z = p->center.z+((float)k-.5f)*p->size;
     ...
     786     l->value = c->value = p->wrapper->Function()->ValueT(c->x, c->y, c->z);
     787     l->corner = c;

     ###   i = -1 : p->center.x + (-1.-0.5)*p.size  = cen + -1.5*sz  
     ###   i = 0 :  p->center.x +  (0.-0.5)*p.size  = cen + -0.5*sz  
     ###   i = 1 :  p->center.x +  (1.-0.5)*p.size  = cen +  0.5*sz 
     ###   i = 2 :  p->center.x +  (2.-0.5)*p.size  = cen +  1.5*sz 

     ### -.5 offset discretization:
     ###
     ###         -2.0     -1.0       0.0       1.0       2.0
     ###              -1.5  |   -.5   |    .5   |   1.5   |
     ###       ----|----|---|----|----|----|----|----|--------
     ###                ^        ^         ^         ^
     ###               -1        0         1         2

ijk from position::

    1471 void find_cube(polygonizer * p, MC_POINT * point,
    1472            int * ijk)
    1473 {
    1474   float d, n;
    1475 
    1476   d = point->x - p->center.x;
    1477   n = fabs(d) / (0.5f*p->size);
    1478   n = ceil( floor(n) / 2.0f );
    1479   ijk[0] = (d > 0.0) ? (int)n : -(int)n;

    # ceil(x) : smallest integral value not less than x
    #    




NFieldGrid3 -> std::function ? Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OctreeNode::GenerateVertexIndices from floated ijk grid coordinates to world coordinates::

    301         OctreeDrawInfo* d = node->drawInfo;
    308         d->index = vertices.size();
    310         vec3 pos = d->position ;
    323         vec3 world = fg->position_f(pos);
    326         vertices.push_back(world);
       
Function evaluation from floated ijk::

    604 float Density_Func(FG3* fg, const vec3& offset_ijk)
    605 {
    606      float fp = fg->value_f(offset_ijk );
    607      return fp ;
    608 }




Alternative serializations to NPY that support compression ?
---------------------------------------------------------------

Bloscpack: a compressed lightweight serialization format for numerical data
Valentin Haenel

* https://arxiv.org/pdf/1404.6383.pdf


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
npy-prim(){     npy-cd ; vi `grep pdump -l *.*` ; }
npy-prim-hpp(){ npy-cd ; vi `grep pdump -l *.hpp` ; }
npy-prim-cpp(){ npy-cd ; vi `grep pdump -l *.cpp` ; }



npy-i(){ npy-scd ; i ; }

npy-bindir(){ echo $(npy-idir)/bin ; } 
npy-bin(){    echo $(npy-bindir)/$1 ; } 

npy-wipe(){
   local bdir=$(npy-bdir)
   rm -rf $bdir
}


npy-node-(){
   local iwd=$PWD
   npy-cd
   local hpp

   grep -l :\ nnode *.hpp | while read hpp ; do
      echo $hpp
   done
   grep -l :\ nnode *.hpp | while read hpp ; do
      echo ${hpp/.hpp}.cpp
   done


   cd $iwd
}

npy-node(){
   npy-cd
   vi $(npy-node-)
}


