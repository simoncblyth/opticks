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
    holder of vertices, indices which fulfils GDrawable
    The GBuffer are created when setters like setVertices etc.. 
    are called 

    NB a relatively small number ~250 of GMesh instances are referenced
    from a much larger number ~12k of GNode arranged in the geometry tree 

    MUST THINK OF GMESH AS ABSTRACT SHAPES **NOT PLACED INSTANCES OF GEOMETRY**
    IT IS INCORRECT TO ASCRIBE SUBSTANCE OR NODE INDICES FOR EXAMPLE  
    SUCH THINGS BELONG ON THE GNODE


GMergedMesh
    specialization of GMesh that combines a tree of GNode 
    and referenced GNode shapes into a flattened single instance
    with transforms applied


GSolid
    GNode specialized with associated GSubstance and selection bit constituents

GNode
    identity index, GMesh and GMatrixF transform 
    also constituent unsigned int* arrays of length matching the face count

    m_substance_indices

    m_node_indices


GDrawable
    abstract interface definition returning GBuffer for vertices, normals, colors, texcoordinates, ...
    Only GMesh and GMergedMesh (by inheritance) fulfil the GDrawable interface

GBuffer
    holds pointer to some bytes together with integers describing the bytes : 
    
    nbytes 
           total number of bytes
    itemsize 
           size of each item in bytes
    nelem
           number of elements within each item 
           (eg item could be gfloat3 of itemsize 12, with nelem 3 
           corresponding to 3 floats) 

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


Where are substance indices formed and associated to every triangle ?
-----------------------------------------------------------------------

* indices are assigned by GSubstanceLib::get based on distinct property values
  this somewhat complicated approach is necessary as GSubstance incorporates info 
  from inner/outer material/surface so GSubstance 
  does not map to simple notions of identity it being a boundary between 
  materials with specific surfaces(or maybe no associated surface) 

* substance indices are affixed to the triangles of the geometry 
  by GSolid::setSubstance GNode::setSubstanceIndices
  which repeats the indice for every triangle of the solid. 
 
  This gets done within the AssimpGGeo::convertStructureVisit,
  the visitor method of the recursive AssimpGGeo::convertStructure 
  in assimpwrap-::

    506     GSubstanceLib* lib = gg->getSubstanceLib();
    507     GSubstance* substance = lib->get(mt, mt_p, isurf, osurf );
    508     //substance->Summary("subst");
    509 
    510     solid->setSubstance(substance);
    511 
    512     char* desc = node->getDescription("\n\noriginal node description");
    513     solid->setDescription(desc);
    514     free(desc);

* substances indices are collected/flattened into 
  the unsigned int* substanceBuffer by GMergedMesh
       

How to map from Geant4 material indices into substance indices ?
--------------------------------------------------------------------

* chroma used a handshake to do this mapping using G4DAEChroma/G4DAEOpticks 
  communication : is this needed here ?

* ggeo is by design a dumb subtrate with which the geometry is represented, 
  the brains of ggeo creation are in assimpwrap-/AssimpGGeo especially:

  * AssimpGGeo::convertMaterials 
  * AssimpGGeo::addPropertyVector (note untested m_domain_reciprocal)



material code handshake between geant4<-->g4daechroma<-->chroma
------------------------------------------------------------------

Geant4/g4daechroma/chroma used metadata handshake resulting in 
a lookup table used by G4DAEChroma to convert geant4 material
codes into chroma ones, where is this implemented ?
 
* gdc-
* env/chroma/G4DAEChroma/G4DAEChroma/G4DAEMaterialMap.hh
* env/chroma/G4DAEChroma/src/G4DAEMaterialMap.cc 
* G4DAEChroma::SetMaterialLookup

* dsc- huh cant find this one locally 
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSimChroma/src 
* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSimChroma/src/DsChromaRunAction_BeginOfRunAction.icc


G4/C++ side
~~~~~~~~~~~~~

DsChromaRunAction_BeginOfRunAction.icc::

    68      G4DAETransport* transport = new G4DAETransport(_transport.c_str());
    69      chroma->SetTransport( transport );
    70      chroma->Handshake();
    71  
    72      G4DAEMetadata* handshake = chroma->GetHandshake();
    73      //handshake->Print("DsChromaRunAction_BeginOfRunAction handshake");
    74  
    75      G4DAEMaterialMap* cmm = new G4DAEMaterialMap(handshake, "/chroma_material_map");
    76      chroma->SetMaterialMap(cmm);
    ..
    79  
    80  #ifndef NOT_NUWA
    81      // full nuwa environment : allows to obtain g4 material map from materials table
    82      G4DAEMaterialMap* gmm = new G4DAEMaterialMap();
    83  #else
    84      // non-nuwa : need to rely on handshake metadata for g4 material map
    85      G4DAEMaterialMap* gmm = new G4DAEMaterialMap(handshake, "/geant4_material_map");
    86  #endif
    87      //gmm->Print("#geant4_material_map");
    88  
    89      int* g2c = G4DAEMaterialMap::MakeLookupArray( gmm, cmm );
    90      chroma->SetMaterialLookup(g2c);
    91  

* http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Simulation/DetSimChroma/src/DsChromaG4Cerenkov.cc

Lookup conversion applied as steps are collected (the most efficient place to do it)::

    308 #ifdef G4DAECHROMA_GPU_OPTICAL
    309     {
    310         // serialize DsG4Cerenkov::PostStepDoIt stack, just before the photon loop
    311         G4DAEChroma* chroma = G4DAEChroma::GetG4DAEChroma();
    312         G4DAECerenkovStepList* csl = chroma->GetCerenkovStepList();
    313         int* g2c = chroma->GetMaterialLookup();
    314 
    315         const G4ParticleDefinition* definition = aParticle->GetDefinition(); 
    316         G4ThreeVector deltaPosition = aStep.GetDeltaPosition();
    317         G4double weight = fPhotonWeight*aTrack.GetWeight();
    318         G4int materialIndex = aMaterial->GetIndex();
    319 
    320         // this relates Geant4 materialIndex to the chroma equivalent
    321         G4int chromaMaterialIndex = g2c[materialIndex] ;

 
::

    104 void G4DAEChroma::Handshake(G4DAEMetadata* request)
    105 {
    106     if(!m_transport) return;
    107     m_transport->Handshake(request);
    108 }

    066 void G4DAETransport::Handshake(G4DAEMetadata* request)
     67 {
     68     if(!request) request = new G4DAEMetadata("{}");
     ..
     76     m_handshake = reinterpret_cast<G4DAEMetadata*>(m_socket->SendReceiveObject(request));
     ..
     85 }



python side
~~~~~~~~~~~~~~

Other end of that handshake:

* env/geant4/geometry/collada/g4daeview/daedirectpropagator.py 

::

     38     def incoming(self, request):
     39         """
     40         Branch handling based on itemshape (excluding first dimension) 
     41         of the request array 
     42         """
     43         self.chroma.incoming(request)  # do any config contained in request
     44         itemshape = request.shape[1:]
     ..
     54         elif itemshape == ():
     55 
     56             log.warn("empty itemshape received %s " % str(itemshape))
     57             extra = True
     ..
     76         return self.chroma.outgoing(response, results, extra=extra)

* env/geant4/geometry/collada/g4daeview/daechromacontext.py 

::

    224     def outgoing(self, response, results, extra=False):
    225         """
    226         :param response: NPL propagated photons
    227         :param results: dict of results from the propagation, eg times 
    228         """
    ...
    230         metadata = {}
    ...
    235         if extra:
    236             metadata['geometry'] = self.gpu_detector.metadata
    237             metadata['cpumem'] = self.mem.metadata()
    238             metadata['chroma_material_map'] = self.chroma_material_map
    239             metadata['geant4_material_map'] = self.geant4_material_map
    240         pass
    241         response.meta = [metadata]
    242         return response




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

AssimpGGeo together with AssimpGeometry and AssimpTree 
(all from AssimpWrap) orchestrate creation of GGeo model
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


