Direct to GLTF feasibility
===========================


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


GLTF Rep 
~~~~~~~~

* learn some more GLTF (especially binary handling) and then 
  decide how best to represent the geometry in that language :
  dont be to too concerned with fitting with classes like GScene/GGeo 

  * aim to replace GGeo/GScene .. ie to unify the analytic 
    and triangulated into a new GLTF based approach 

  * aim to replace G4DAE+GDML writing, GDML python parsing with sc.py  
 
  * aim to keep NPY and the GPropLibs : so geometry consumers 
    (OptiXRap, OGLRap) can be mostly unchanged  

  * BUT what about GScene, analytic geometry hailing 
    from the python GDML parse

  * structure of the consumers expects both triangulated and
    analytic in GGeo and GScene (dispensed by OpticksHub)



GLTF refs
~~~~~~~~~~~

* https://github.com/KhronosGroup/glTF



how to do direct shortcutting of material props ?
---------------------------------------------------------

0. review gltf standard regarding binary data description, check how
   oyoctogl handles this, see if can integrate NPY buffers in a compliant way 

1. devise gltf approach and file layout to hold the props that 
   is close to the geocache layout of GMaterialLib 
   with NPY buffers for binary data 

   * granularity decisions : per-material, per-property ? start with the existing G*Lib decisions

2. translate the COLLADA export in G4DAE to populate in memory gltf tree, from live G4 
   hmm how is binary handled in gltf world ?



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




