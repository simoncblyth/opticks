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


TODO
-----

* rejig relationship between GSubstanceLib and GSubstance
  
  * lib doing too much, substance doing too little

  * move standardization from the lib to the substance
    so that lib keys are standard digests, this will 
    allow the standard lib to be reconstructed from the 
    wavelengthBuffer and will offer simple digest matching
    to the metadata 

  * can have a separate non-standardized lib intstance 
    as a container for all properties 


Material index mapping 
------------------------

Cerenkov steps contain indices which have been chroma mapped already::

    In [4]: cs = CerenkovStep.get(1)

    In [8]: cs.materialIndices
    Out[8]: CerenkovStep([ 1,  8, 10, 12, 13, 14, 19], dtype=int32)

    In [9]: cs.materialIndex
    Out[9]: CerenkovStep([12, 12, 12, ...,  8,  8,  8], dtype=int32)

    In [11]: np.unique(cs.materialIndex)
    Out[11]: CerenkovStep([ 1,  8, 10, 12, 13, 14, 19], dtype=int32)


Modified daechromacontext.py to dump the chroma mapping with 
names translated into geant4 style::

    In [13]: import json

    In [15]: cmm = json.load(file("/tmp/ChromaMaterialMap.json"))

    In [16]: cmm
    Out[16]: 
    {u'/dd/Materials/ADTableStainlessSteel': 0,
     u'/dd/Materials/Acrylic': 1,
     u'/dd/Materials/Air': 2,
     u'/dd/Materials/Aluminium': 3,
     u'/dd/Materials/BPE': 4,
     u'/dd/Materials/Bialkali': 5,
     u'/dd/Materials/C_13': 6,
     u'/dd/Materials/Co_60': 7,
     u'/dd/Materials/DeadWater': 8,
     u'/dd/Materials/ESR': 9,
     u'/dd/Materials/GdDopedLS': 10,
     u'/dd/Materials/Ge_68': 11,
     u'/dd/Materials/IwsWater': 12,
     u'/dd/Materials/LiquidScintillator': 13,
     u'/dd/Materials/MineralOil': 14,
     u'/dd/Materials/Nitrogen': 15,
     u'/dd/Materials/NitrogenGas': 16,
     u'/dd/Materials/Nylon': 17,
     u'/dd/Materials/OpaqueVacuum': 18,
     u'/dd/Materials/OwsWater': 19,
     u'/dd/Materials/PVC': 20,
     u'/dd/Materials/Pyrex': 21,
     u'/dd/Materials/Silver': 22,
     u'/dd/Materials/StainlessSteel': 23,
     u'/dd/Materials/Teflon': 24,
     u'/dd/Materials/Tyvek': 25,
     u'/dd/Materials/UnstStainlessSteel': 26,
     u'/dd/Materials/Vacuum': 27,
     u'/dd/Materials/Water': 28}



::

    delta:npy blyth$ npy-g4stepnpy-test 
    Lookup::dump LookupTest 
    A   29 entries from /tmp/ChromaMaterialMap.json
    B   24 entries from /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GSubstanceLibMetadataMaterialMap.json
    A2B 21 entries in lookup  
      A    0 :     ADTableStainlessSteel  B  212 :     ADTableStainlessSteel 
      A    1 :                   Acrylic  B   56 :                   Acrylic 
      A    2 :                       Air  B    8 :                       Air 
      A    3 :                 Aluminium  B   16 :                 Aluminium 
      A    5 :                  Bialkali  B   84 :                  Bialkali 
      A    8 :                 DeadWater  B   28 :                 DeadWater 
      A    9 :                       ESR  B  104 :                       ESR 
      A   10 :                 GdDopedLS  B   68 :                 GdDopedLS 
      A   12 :                  IwsWater  B   44 :                  IwsWater 
      A   13 :        LiquidScintillator  B   60 :        LiquidScintillator 
      A   14 :                MineralOil  B   52 :                MineralOil 
      A   15 :                  Nitrogen  B  128 :                  Nitrogen 
      A   16 :               NitrogenGas  B  172 :               NitrogenGas 
      A   17 :                     Nylon  B  140 :                     Nylon 
      A   19 :                  OwsWater  B   36 :                  OwsWater 
      A   21 :                     Pyrex  B   76 :                     Pyrex 
      A   23 :            StainlessSteel  B   48 :            StainlessSteel 
      A   25 :                     Tyvek  B   32 :                     Tyvek 
      A   26 :        UnstStainlessSteel  B   88 :        UnstStainlessSteel 
      A   27 :                    Vacuum  B    0 :                    Vacuum 
      A   28 :                     Water  B  125 :                     Water 
    cs.dump
     ni 7836 nj 6 nk 4 nj*nk 24 
     (    0,    0)               -1                1               44               80  sid/parentId/materialIndex/numPhotons 
     (    0,    1)       -16536.295      -802084.812        -7066.000            0.844  position/time 
     (    0,    2)           -2.057            3.180            0.000            3.788  deltaPosition/stepLength 
     (    0,    3)               13           -1.000            1.000          299.791  code 
     (    0,    4)            1.000            0.000            0.000            0.719 
     (    0,    5)            0.482           79.201           79.201            0.000 
     ( 7835,    0)            -7836                1               28               48  sid/parentId/materialIndex/numPhotons 
     ( 7835,    1)       -20842.291      -795380.438        -7048.775           27.423  position/time 
     ( 7835,    2)           -1.068            1.669            0.004            1.981  deltaPosition/stepLength 
     ( 7835,    3)               13           -1.000            1.000          299.790  code 
     ( 7835,    4)            1.000            0.000            0.000            0.719 
     ( 7835,    5)            0.482           79.201           79.201            0.000 

    ... 28 [DeadWater] 
    ... 36 [OwsWater] 
    ... 44 [IwsWater] 
    ... 52 [MineralOil] 
    ... 56 [Acrylic] 
    ... 60 [LiquidScintillator] 
    ... 68 [GdDopedLS] 
    delta:npy blyth$ 

    (chroma_env)delta:env blyth$ ggeo-meta 28 36 44 52 56 60 68
    /usr/local/env/optix/ggeo/bin/GSubstanceLibTest /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae 28 36 44 52 56 60 68
    wavelength buffer NumBytes 134784 Ptr 0x10acde000 ItemSize 4 NumElements_PerItem 1 NumItems(NumBytes/ItemSize) 33696 NumElementsTotal (NumItems*NumElements) 33696 

    GSubstanceLib::dumpWavelengthBuffer wline 28 wsub 7 wprop 0 numSub 54 domainLength 39 numProp 16 

      28 |   7/  0 __dd__Materials__DeadWater0xbf8a548 
               1.390           1.390           1.372           1.357           1.352           1.346           1.341           1.335
             273.208         273.208        3164.640       12811.072       28732.207       13644.791        2404.398         371.974
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    GSubstanceLib::dumpWavelengthBuffer wline 36 wsub 9 wprop 0 numSub 54 domainLength 39 numProp 16 

      36 |   9/  0 __dd__Materials__OwsWater0xbf90c10 
               1.390           1.390           1.372           1.357           1.352           1.346           1.341           1.335
             273.208         273.208        3164.640       12811.072       28732.207       13644.791        2404.398         371.974
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    GSubstanceLib::dumpWavelengthBuffer wline 44 wsub 11 wprop 0 numSub 54 domainLength 39 numProp 16 

      44 |  11/  0 __dd__Materials__IwsWater0xc288f98 
               1.390           1.390           1.372           1.357           1.352           1.346           1.341           1.335
             273.208         273.208        3164.640       12811.072       28732.207       13644.791        2404.398         371.974
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    GSubstanceLib::dumpWavelengthBuffer wline 52 wsub 13 wprop 0 numSub 54 domainLength 39 numProp 16 

      52 |  13/  0 __dd__Materials__MineralOil0xbf5c830 
               1.434           1.758           1.540           1.488           1.471           1.464           1.459           1.457
              11.100          11.100          11.394        1078.898       24925.316       21277.369        5311.868         837.710
             850.000         850.000        4901.251       19819.381       52038.961      117807.406      252854.656      420184.219
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    GSubstanceLib::dumpWavelengthBuffer wline 56 wsub 14 wprop 0 numSub 54 domainLength 39 numProp 16 

      56 |  14/  0 __dd__Materials__Acrylic0xc02ab98 
               1.462           1.793           1.573           1.519           1.500           1.494           1.490           1.488
               0.008           0.008        4791.046        8000.000        8000.000        8000.000        8000.000        8000.000
             850.000         850.000        4901.251       19819.381       52038.961      117807.406      252854.656      420184.219
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    GSubstanceLib::dumpWavelengthBuffer wline 60 wsub 15 wprop 0 numSub 54 domainLength 39 numProp 16 

      60 |  15/  0 __dd__Materials__LiquidScintillator0xc2308d0 
               1.454           1.793           1.563           1.511           1.494           1.485           1.481           1.479
               0.001           0.001           0.198           1.913       26433.846       31710.930        6875.426         978.836
             850.000         850.000        4901.251       19819.381       52038.961      117807.406      252854.656      420184.219
               0.400           0.400           0.599           0.800           0.169           0.072           0.023           0.000
    GSubstanceLib::dumpWavelengthBuffer wline 68 wsub 17 wprop 0 numSub 54 domainLength 39 numProp 16 

      68 |  17/  0 __dd__Materials__GdDopedLS0xc2a8ed0 
               1.454           1.793           1.563           1.511           1.494           1.485           1.481           1.479
               0.001           0.001           0.198           1.913       26623.084       27079.125        7315.331         989.154
             850.000         850.000        4901.251       19819.381       52038.961      117807.406      252854.656      420184.219
               0.400           0.400           0.599           0.800           0.169           0.072           0.023           0.000


reemission_cdf reciprocation
-----------------------------

* for Geant4 match of photons generated in Chroma context needed to sample on 
  an energywise domain 1/wavelength[::-1]  

  * env/geant4/geometry/collada/collada_to_chroma.py::construct_cdf_energywise  

  * sampling wavelength-wise gives poor match at the extremes of the distribution 

  * i dont this there is anything fundamental here, its just matching precisely
    what is done by Geant4/NuWa generation

* this was implemented by special casing chroma material reemission_cdf property

env/geant4/geometry/collada/collada_to_chroma.py::

    515     def setup_cdf(self, material, props ):
    516         """
    517         Chroma uses "reemission_cdf" cumulative distribution function 
    518         to generate the wavelength of reemission photons. 
    519 
    520         Currently think that the name "reemission_cdf" is misleading, 
    521         as it is the RHS normalized CDF obtained from an intensity distribution
    522         (photon intensity as function of wavelength) 
    523 
    524         NB REEMISSIONPROB->reemission_prob is handled as a 
    525         normal keymapped property, no need to integrate to construct 
    526         the cdf for that.
    527     
    528         Compare this with the C++
    529 
    530            DsChromaG4Scintillation::BuildThePhysicsTable()  
    531 
    532         """ 

how to do this within ggeo/OptiX
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#. check what properties beyond the gang-of-four for materials and surfaces 
   are used by Chroma propagation

   * looks like makes sense to use a separate ggeo texture holding just reemission_cdf

/usr/local/env/chroma_env/src/chroma/chroma/cuda/geometry_types.h::

     04 struct Material
      5 {
      6     float *refractive_index;
      7     float *absorption_length;
      8     float *scattering_length;
      9     float *reemission_prob;
     10     float *reemission_cdf;   // SCB ? misleading as not only applicable to reemission ?  maybe intensity_cdf better
     11     unsigned int n;          // domain spec
     12     float step;              // domain spec
     13     float wavelength0;       // domain spec
     14 };
     ..
     18 struct Surface
     19 {
     20     float *detect;                 
     21     float *absorb;
     22     float *reemit;              // only used by propagate_at_wls 
     23     float *reflect_diffuse;
     24     float *reflect_specular;
     25     float *eta;                 // only used by propagate_complex
     26     float *k;                   // only used by propagate_complex
     27     float *reemission_cdf;      // only used by propagate_at_wls 
     28 
     29     unsigned int model;         // selects between SURFACE_DEFAULT, SURFACE_COMPLEX, SURFACE_WLS 
     30     unsigned int n;             // domain spec
     31     unsigned int transmissive;  // only used by propagate_complex
     32     float step;                 // domain spec
     33     float wavelength0;          // domain spec
     34     float thickness;A           // only used by propagate_complex
     35 };
        




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


Where does chroma_material_map get written ?
----------------------------------------------

* env/geant4/geometry/collada/g4daeview/daegeometry.py 

::

     796         cc = ColladaToChroma(DAENode, bvh=bvh )
     797         cc.convert_geometry(nodes=self.nodes())
     798 
     799         self.cc = cc
     800         self.chroma_material_map = DAEChromaMaterialMap( self.config, cc.cmm )
     801         self.chroma_material_map.write()
     802         log.debug("completed DAEChromaMaterialMap.write")


* env/geant4/geometry/collada/collada_to_chroma.py creates contiguous 0-based indices 
  for each unique material, the chroma array of materials then gets copied to GPU 
  hence the indices are as needed for GPU side material lookups

::

    634     def convert_make_maps(self):
    635         self.cmm = self.make_chroma_material_map( self.chroma_geometry )
    636         self.csm = self.make_chroma_surface_map( self.chroma_geometry )

::

    663     def make_chroma_material_map(self, chroma_geometry):
    664         """
    665         Curiously the order of chroma_geometry.unique_materials on different invokations is 
    666         "fairly constant" but not precisely so. 
    667         How is that possible ? Perfect or random would seem more likely outcomes. 
    668         """
    669         unique_materials = chroma_geometry.unique_materials
    670         material_lookup = dict(zip(unique_materials, range(len(unique_materials))))
    671         cmm = dict([(material_lookup[m],m.name) for m in filter(None,unique_materials)])
    672         cmm[-1] = "ANY"
    673         cmm[999] = "UNKNOWN"
    674         return cmm




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


Boundary Check
----------------

::

    In [1]: a = oxc_(1)

    In [6]: count_unique(a[:,3,0].view(np.int32))
    Out[6]: 
    array([[    -1,  53472],
           [    11,  10062],   IwsWater/IwsWater
           [    12,   6612],   StainlessSteel/IwsWater
           [    13,   9021],   MineralOil/StainlessSteel
           [    14,  28583],   Acrylic/MineralOil
           [    15,  45059],   LiquidScintillator/Acrylic
           [    16,  95582],   Acrylic/LiquidScintillator
           [    17, 311100],   GdDopedLS/Acrylic
           [    19,    576],   Pyrex/MineralOil
           [    20,    282],   Vacuum/Pyrex
           [    22,   2776],   UnstStainlessSteel/MineralOil
           [    24,  36214],   Acrylic/MineralOil
           [    31,    710],   StainlessSteel/Water
           [    32,    194],   Nitrogen/StainlessSteel
           [    49,  11831],   UnstStainlessSteel/IwsWater
           [    50,     23],   Nitrogen/Water
           [    52,    744]])  Pyrex/IwsWater

 




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
ggeo-bin(){ echo $(ggeo-idir)/bin/${1:-GGeoTest} ; }


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

ggeo-meta-dir(){
   echo /usr/local/env/geant4/geometry/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
}

ggeo-libtest(){
   local bin=$(ggeo-bin GSubstanceLibTest)
   local cmd="$bin $(ggeo-meta-dir) $*"
   echo $cmd
   eval $cmd
}


ggeo-metatest(){
   local bin=$(ggeo-bin GSubstanceLibMetadataTest)
   local cmd="$bin $(ggeo-meta-dir) $*"
   echo $cmd
   eval $cmd
}





ggeo-gpropertytest(){
   local bin=$(ggeo-bin GPropertyTest)
   echo $bin
   eval $bin
}

