# === func-gen- : optix/ggeo/ggeo fgp optix/ggeo/ggeo.bash fgn ggeo fgh optix/ggeo
ggeo-src(){      echo optix/ggeo/ggeo.bash ; }
ggeo-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ggeo-src)} ; }
ggeo-vi(){       vi $(ggeo-source) ; }
ggeo-env(){      elocal- ; }
ggeo-usage(){ cat << EOU

GGEO : Intermediary Geometry Model
=====================================

Unencumbered geometry and material/surface property
model, intended for:

* investigation of how to represent Geometry within OptiX


Classes
--------

GGeo
    top level control and holder of other instances:
    GMesh GSolid GMaterial GSkinSurface GBorderSurface GSubstanceLib

GMesh
    holder of vertices, indices

GSolid
    GNode specialized with associated GSubstance

GNode
    identity index, GMesh and GMatrixF transform

GSubstance
    holder of inner/outer material and inner/outer surface GPropertMaps

GSubstanceLib
    manager of substances, ensures duplicates are not created via digests


GBorderSurface
    PropertyMap specialization, specialization only used for creation
GSkinSurface
    PropertyMap specialization, specialization only used for creation
GMaterial
    PropertyMap specialization, specialization only used for creation
GPropertyMap
    ordered holder of GProperty<double> and GDomain<double>
GProperty<T>
    pair of GAry<T> for values and domain
GAry<T>
    array of values with linear interpolation functionality
GDomain
    standard range for properties, eg wavelength range and step


GVector
    gfloat3 guint3 structs
GMatrix
    4x4 matrix
GEnums
    material/surface property enums 

md5digest
    hashing



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

Analogs to Chroma
-------------------

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

    template <class T>
    class GProperty {

       * domain and value arrays + length

    class GPropertyMap {

       * string keyed map of GProperty<float>

    class GMaterial      : public GPropertyMap {
    class GBorderSurface : public GPropertyMap {
    class GSkinSurface   : public GPropertyMap {

    class GMesh {

        * vertices and faces


Client Classes
---------------

::

    class GSolid {

        * mesh + inside/outside materials and surfaces
        * nexus of structure

    class GGeo {

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


