# === func-gen- : optix/ggeo/ggeo fgp optix/ggeo/ggeo.bash fgn ggeo fgh optix/ggeo
ggeo-src(){      echo optix/ggeo/ggeo.bash ; }
ggeo-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ggeo-src)} ; }
ggeo-vi(){       vi $(ggeo-source) ; }
ggeo-env(){      elocal- ; }
ggeo-usage(){ cat << EOU

GGEO : Intermediary Geometry Model
=====================================

Investigating how to represent Geometry with OptiX

GGeo Geometry Model Objective
------------------------------

* dumb holder of geometry information including the 
  extra material and surface properties, build on top of 
  mostly low level primitives with some use of map, string, vector,...  
  being admissable

  * NO imports from Assimp or OptiX
  * depend on user for most of the construction 

* intended to be a lightweight/slim intermediary format, eg  
  between raw Assimp geometry and properties
  to be consumed by the OptiX geometry creator/parameter setter.

* NB not trying to jump directly to an optix on GPU struct
  as the way to represent info efficiently within optix 
  needs experimentation : eg perhaps using texturemap lookups.
  Nevertheless, keep fairly low level to ease transition to
  on GPU structs

* intended to be a rather constant model, from which 
  a variety of OptiX representations can be developed 

* possible side-feature : geometry caching for fast starts
  without having to parse COLLADA.


Relationship to AssimpGeometry ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

AssimpGeometry/AssimpTree orchestrates creation of GGeo model
from the imported Assimp model. The OptiX model should then 
be created entirely from the GGeo model with no use of 
the Assimp model.

Analogs 
--------

* GMaterial : chroma.geometry.Material
* GSurface : chroma.geometry.Surface
* GMesh : chroma.geometry.Mesh
* GSolid : chroma.geometry.Solid
* GGeo : chroma.geometry.Geometry 

* AssimpWrap/AssimpGeometry/AssimpTree 
  are analogs of g4daenode.py and collada_to_chroma.py 


Basis Classes
---------------
::

    GProperty.hh:template <class T>
    GProperty.hh:class GProperty {

       * domain and value arrays + length

    GPropertyMap.hh:class GPropertyMap {

       * string keyed map of GProperty<float>

    GMaterial.hh:class GMaterial : public GPropertyMap {
    GBorderSurface.hh:class GBorderSurface : public GPropertyMap {
    GSkinSurface.hh:class GSkinSurface : public GPropertyMap {


    GMesh.hh:class GMesh {

        * vertices and faces


Client Classes
---------------

::

    GSolid.hh:class GSolid {

        * mesh + inside/outside materials and surfaces
        * nexus of structure

    GGeo.hh:class GGeo {

        * vectors of pointers to solids, materials, skin surfaces, border surfaces 




OptiX Geometry Model
---------------------

* many little programs and their parameters in flexible context 

* Material (closest hit program, anyhit program ) and params the programs use 
* Geometry (bbox program, intersection program) and params the programs use
* GeometryInstance associate Geometry with usually one Material (can be more than one) 

* try: representing material/surface props into 1D(wavelength) textures 



Chroma Geometry Model
----------------------

Single all encompassing Geometry instance containing:

* arrays of materials and surfaces
* material codes identifying material and surface indices for every triangle

chroma/chroma/cuda/geometry_types.h::

    struct Material
    {
        float *refractive_index;
        float *absorption_length;
        float *scattering_length;
        float *reemission_prob;
        float *reemission_cdf;   // SCB ? misleading as not only applicable to reemission ?  maybe intensity_cdf better
        unsigned int n;
        float step;
        float wavelength0;
    };

    struct Surface
    {
        float *detect;
        float *absorb;
        float *reemit;
        float *reflect_diffuse;
        float *reflect_specular;
        float *eta;
        float *k; 
        float *reemission_cdf;

        unsigned int model;
        unsigned int n;
        unsigned int transmissive;
        float step;
        float wavelength0;
        float thickness;
    };



    struct Geometry
    {
        float3 *vertices;
        uint3 *triangles;
        unsigned int *material_codes;
        unsigned int *colors;
        uint4 *primary_nodes;
        uint4 *extra_nodes;
        Material **materials;
        Surface **surfaces;
        float3 world_origin;
        float world_scale;
        int nprimary_nodes;
    };



 




EOU
}

ggeo-idir(){ echo $(local-base)/env/optix/ggeo; }  # prefix
ggeo-bdir(){ echo $(local-base)/env/optix/ggeo.build ; }
ggeo-sdir(){ echo $(env-home)/optix/ggeo ; }

ggeo-icd(){  cd $(ggeo-idir); }
ggeo-bcd(){  cd $(ggeo-bdir); }
ggeo-scd(){  cd $(ggeo-sdir); }

ggeo-cd(){  cd $(ggeo-sdir); }

ggeo-mate(){ mate $(ggeo-dir) ; }

ggeo-wipe(){
    local bdir=$(ggeo-bdir)
    rm -rf $bdir
}


ggeo-cmake(){
   local bdir=$(ggeo-bdir)
   mkdir -p $bdir
   ggeo-bcd
   cmake $(ggeo-sdir) -DCMAKE_INSTALL_PREFIX=$(ggeo-idir) -DCMAKE_BUILD_TYPE=Debug 
}

ggeo-make(){
    local iwd=$PWD
    ggeo-bcd
    make $*
    cd $iwd
}

ggeo-install(){
   ggeo-make install
}


ggeo-bbin(){ echo $(ggeo-bdir)/GGeoTest ; }
ggeo-bin(){ echo $(ggeo-idir)/bin/GGeoTest ; }


ggeo-export(){
    env | grep GGEO
}


ggeo-run(){
    ggeo-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

    ggeo-export 
    $DEBUG $(ggeo-bin) $*  
}

ggeo--(){
    ggeo-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 
    ggeo-install $*
}

ggeo-lldb(){
    DEBUG=lldb ggeo-run
}

ggeo-brun(){
   echo running from bdir not idir : no install needed, but much set library path
   local bdir=$(ggeo-bdir)
   DYLD_LIBRARY_PATH=$bdir $DEBUG $bdir/GGeoTest 
}


ggeo-test(){
    local arg=$1
    ggeo-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

    ggeo-export $arg
    DEBUG=lldb ggeo-brun
}


ggeo-otool(){
   otool -L $(ggeo-bin)
}


