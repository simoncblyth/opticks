Direct to GLTF feasibility
===========================

Basic Tech
------------

GLTF Learning
~~~~~~~~~~~~~~

* https://github.com/KhronosGroup/glTF

* DONE : review gltf standard regarding binary data description, check how
  oyoctogl handles this, see if can integrate NPY buffers in a compliant way 

  * created YoctoGLRap to learn how to use ygltf with NPY buffers, the small 
    demo gltf files created can be loaded by GLTF viewers  

GLTF Questions 
~~~~~~~~~~~~~~~~

* binary data in extras/extensions, how to do that with ygltf ?
* extras vs extensions whats the difference ?

* what package (ie which dependencies to base on) further investigations

X4 : ExtG4 package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

ExtG4 
   new package bringing together G4 and lower level Opticks 
   (SysRap, NPY, GGeo?) : using lots of static enhancers to 
   single Geant4 classes  

extg4/X4VSolid

   1. DONE: polygonize G4VSolid instances into vtx and tri NPY buffers 
   2. TODO: get YOG to work with X4VSolid instances, so can save them as GLTF
   3. TODO: add analytic type info regarding the solid into ... hmm actually the
      test solid is a sphere, the analytic GDML style info needs to be collected 
      prior to descending to generic solid G4VSolid and somehow communicated thru to 
      the X4VSolid 
      ... perhaps static Create methods accepting the various types like G4Sphere etc.., 
      which collect some ygltf json describing the type and its parameters 
      look at (NCSG, nnode, GParts, sc.py, GDML) as thats where this data is headed
      or is replacing.   
      This json can be planted inside a mesh.extra property. 
        
   For example to convert a G4VSolid into X4VSolid comprising 
   GMesh and GParts constituents


Approaches 
------------

Following current AssimpImprter ? NO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* perhaps replace the AssimpImporter with a G4LiveImporter ? 
  ie that populates GGeo from a live G4 tree rather than 
  the Assimp tree loaded from G4DAE file 

  * this is too tied to current overly organic structure, 
    the aim is to simplify : ending up with significantly less code, 
    not to add more code

Unified geometry handling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~


* ORIGINALLY THOUGHT : aim to replace GGeo/GScene .. ie to unify the analytic 
  and triangulated into a new GLTF based approach 

* but actually GGeo is OK, what I dont like is the historical split into 
  two lines of geometry that come from G4DAE (GGeo) and GDML (GScene)
  and the resulting duplication in the handling of two geometry trees :
  the analytic information that GDML adds (GParts) should live together with 
  the triangulated information (GMesh) in the unified GLTF based geometry 
  using NPY buffers

  * dont like split of analytic args to GGeo constituents : analytic/GDML 
    info lives beside the rest on equal footing

* aim to replace G4DAE+GDML writing, GDML python parsing with sc.py  

* aim to keep NPY and the GPropLibs : so geometry consumers 
  (OptiXRap, OGLRap) can be mostly unchanged  

* BUT what about GScene, analytic geometry hailing 
  from the python GDML parse

* structure of the consumers expects both triangulated and
  analytic in GGeo and GScene (dispensed by OpticksHub)


analytic/gdml.py 
~~~~~~~~~~~~~~~~~~

* converts some parsed raw GDML solid primitives (depending on their parameters, eg rmin) 
  into CSG boolean composities

  * line between solid and composite is not fixed  

  * treating such shapes as composite CSG avoids code duplication (so reduces bug potential)
    as would otherwise require reimplementing the same logic for multiple shapes

  * where is appropriate to do this kind of specialization ? how general to make the GLTF ?
    whichever the choice need to record all the parameters of the solids 


analytic/sc.py 
~~~~~~~~~~~~~~~~~~

* /Volumes/Delta/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf

Observations on the GLTF:

* not in geocache, its regarded are source : living in opticksdata


gltf for materials ?
~~~~~~~~~~~~~~~~~~~~~~

* currently no materials in the gltf : that comes the trianulated route 
* need to come up with a structure to live in json extras.

Whats needed::

* material shortname
* list of uri of the properties (directory structure?) 
 
Can refer to the properties NPY using a list of GLTF buffers (
just needs list of uri with bytelengths, offsets).  

Is this needed, as are in extras. Depends on how can get ygltf to 
handle extras and writing binaries ? 

* "save_ygltf" expects memory data buffers std::vector<unsigned char> 
   which it can save, can do that, but maybe no point as will need to 
   implement separate saving and loading of extras 
 

Hmm can use standard buffers for the properties::

    2652 YGLTF_API void save_buffers(const glTF_t* gltf, const std::string& dirname) {
    2653     for (auto& buffer_ : gltf->buffers) {
    2654         auto buffer = &buffer_;
    2655         if (_startsiwith(buffer->uri, "data:"))
    2656             throw gltf_exception("saving of embedded data not supported");
    2657         _save_binfile(dirname + buffer->uri, buffer->data);
    2658     }
    2659 }

Will need to add handling of non existing and intermediate directories. OR just 
using existing persisting capabilities of GMaterial/GPropertyMap. 

Also no need for accessor descriptor machinery : as this data is 
intended for Opticks code (not OpenGL renderers).


/usr/local/opticks-cmake-overhaul/externals/g4dae/g4dae-opticks/src/G4DAEWriteMaterials.cc::

    088 void G4DAEWriteMaterials::MaterialWrite(const G4Material* const materialPtr)
     89 {
     90    const G4String matname = GenerateName(materialPtr->GetName(), materialPtr);
     91    const G4String fxname = GenerateName(materialPtr->GetName() + "_fx_", materialPtr);
     92 
     93    xercesc::DOMElement* materialElement = NewElementOneNCNameAtt("material","id",matname);
     94    xercesc::DOMElement* instanceEffectElement = NewElementOneNCNameAtt("instance_effect","url",fxname, true);
     95    materialElement->appendChild(instanceEffectElement);
     96 
     97    G4MaterialPropertiesTable* ptable = materialPtr->GetMaterialPropertiesTable();
     98    if(ptable)
     99    {
    100        xercesc::DOMElement* extraElement = NewElement("extra");
    101        PropertyWrite(extraElement, ptable);
    102        materialElement->appendChild(extraElement);
    103    }
    104 
    105    materialsElement->appendChild(materialElement);
    106 
    107      // Append the material AFTER all the possible components are appended!
    108 }

/usr/local/opticks-cmake-overhaul/externals/g4dae/g4dae-opticks/src/G4DAEWrite.cc


question : how much processing prior to forming the YGLTF structure ?
------------------------------------------------------------------------

* should GGeo constituent instances eg GMaterial be formed at that juncture or later ? 

GMaterialLib
~~~~~~~~~~~~~~~

* GMaterialLib focusses on the optical properties, should unigeo "G4GLTF" be more general ? 
* eg domain regularization of material/surface properties 


how to do direct shortcutting of material props ?
---------------------------------------------------------

1. devise gltf approach and file layout to hold the props that 
   is close to the geocache layout of GMaterialLib 
   with NPY buffers for binary data 

   * granularity decisions : per-material, per-property ? start with the existing G*Lib decisions

2. translate the COLLADA export in G4DAE to populate in memory gltf tree, from live G4 
   hmm how is binary handled in gltf world ?



reminders GMesh, GMergedMesh when is merge done ?
---------------------------------------------------




geocache description of materials
-------------------------------------

::

    epsilon:1 blyth$ l GMaterialLib/
    total 96
    -rw-r--r--  1 blyth  staff  - 47504 Apr  4 21:59 GMaterialLib.npy

    epsilon:1 blyth$ head -10 GItemList/GMaterialLib.txt 
    GdDopedLS
    LiquidScintillator
    Acrylic
    MineralOil
    Bialkali
    IwsWater
    Water
    DeadWater
    OwsWater
    ESR

    epsilon:1 blyth$ wc -l GItemList/GMaterialLib.txt 
          38 GItemList/GMaterialLib.txt


::

    In [1]: a = np.load("GMaterialLib.npy")

    In [2]: a.shape
    Out[2]: (38, 2, 39, 4)

    In [3]: pwd
    Out[3]: u'/usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GMaterialLib'



GLTF materials : not relevant : will need to use extras
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

     644 struct material_t : glTFChildOfRootProperty_t {
     645     /// The emissive color of the material.
     646     std::array<float, 3> emissiveFactor = {{0, 0, 0}};
     647     /// The emissive map texture.
     648     textureInfo_t emissiveTexture = {};
     649     /// The normal map texture.
     650     material_normalTextureInfo_t normalTexture = {};
     651     /// The occlusion map texture.
     652     material_occlusionTextureInfo_t occlusionTexture = {};
     653     /// A set of parameter values that are used to define the metallic-roughness
     654     /// material model from Physically-Based Rendering (PBR) methodology.
     655     material_pbrMetallicRoughness_t pbrMetallicRoughness = {};
     656 };
     657 



G4DAEWrite::PropertyWrite 
~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    441 void G4DAEWrite::PropertyWrite(xercesc::DOMElement* extraElement,  const G4MaterialPropertiesTable* const ptable)
    442 {
    443    xercesc::DOMElement* propElement;
    444    const std::map< G4String, G4MaterialPropertyVector*,
    445                  std::less<G4String> >* pmap = ptable->GetPropertiesMap();
    446    const std::map< G4String, G4double,
    447                  std::less<G4String> >* cmap = ptable->GetPropertiesCMap();
    448    std::map< G4String, G4MaterialPropertyVector*,
    449                  std::less<G4String> >::const_iterator mpos;
    450    std::map< G4String, G4double,
    451                  std::less<G4String> >::const_iterator cpos;
    452    for (mpos=pmap->begin(); mpos!=pmap->end(); mpos++)
    453    {
    454       propElement = NewElement("property");
    455       propElement->setAttributeNode(NewAttribute("name", mpos->first));
    456       propElement->setAttributeNode(NewAttribute("ref",
    457                                     GenerateName(mpos->first, mpos->second)));
    458       if (mpos->second)
    459       {
    460          PropertyVectorWrite(mpos->first, mpos->second, extraElement);
    461          extraElement->appendChild(propElement);
    462       }
    463       else
    464       {
    465          G4String warn_message = "Null pointer for material property -" + mpos->first ;
    466          G4Exception("G4DAEWrite::PropertyWrite()", "NullPointer",
    467                      JustWarning, warn_message);
    468          continue;
    469       }
    470    }
    471    for (cpos=cmap->begin(); cpos!=cmap->end(); cpos++)
    472    {
    473       propElement = NewElement("property");
    474       propElement->setAttributeNode(NewAttribute("name", cpos->first));
    475       propElement->setAttributeNode(NewAttribute("ref", cpos->first));
    476       xercesc::DOMElement* constElement = NewElement("constant");
    477       constElement->setAttributeNode(NewAttribute("name", cpos->first));
    478       constElement->setAttributeNode(NewAttribute("value", cpos->second));
    479       // tacking onto a separate top level define element for GDML
    480       // but that would need separate access on reading 
    481 
    482       //defineElement->appendChild(constElement);
    483       extraElement->appendChild(constElement);
    484       extraElement->appendChild(propElement);
    485    }
    486 }





Start with something manageable : translating G4 materials to a gltf representation (oyoctogl- structs)
----------------------------------------------------------------------------------------------------------




