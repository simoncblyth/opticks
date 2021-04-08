gdml-export-not-writing-all-materials-causing-mismatch
=========================================================

Geant4 gdml export is only writing out used materials whereas the 
Opticks conversion does all materials of the geometry.

* YES : so what : where does the error happen ?


::

    epsilon:src blyth$ g4-cls G4GDMLWriteMaterials
    /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02
    vi -R source/persistency/gdml/include/G4GDMLWriteMaterials.hh source/persistency/gdml/src/G4GDMLWriteMaterials.cc
    2 files to edit


    318 void G4GDMLWriteMaterials::AddMaterial(const G4Material* const materialPtr)
    319 {
    320    for (size_t i=0;i<materialList.size();i++)    // Check if material is
    321    {                                             // already in the list!
    322       if (materialList[i] == materialPtr)  { return; }
    323    }
    324    materialList.push_back(materialPtr);
    325    MaterialWrite(materialPtr);
    326 }


    epsilon:src blyth$ grep AddMaterial *.*
    G4GDMLReadMaterials.cc:         if (materialPtr != 0) { material->AddMaterial(materialPtr,n); }
    G4GDMLWriteMaterials.cc:void G4GDMLWriteMaterials::AddMaterial(const G4Material* const materialPtr)
    G4GDMLWriteStructure.cc:   AddMaterial(volumePtr->GetMaterial());
    epsilon:src blyth$ 


AddMaterial invoked in the recursive tail of TraverseVolumeTree::

    388 G4Transform3D G4GDMLWriteStructure::
    389 TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth)
    390 {
    391    if (VolumeMap().find(volumePtr) != VolumeMap().end())
    392    { 
    393      return VolumeMap()[volumePtr]; // Volume is already processed
    394    }


    395   
    396    G4VSolid* solidPtr = volumePtr->GetSolid();
    397    G4Transform3D R,invR;
    398    G4int trans=0;
    399 
    400    std::map<const G4LogicalVolume*, G4GDMLAuxListType>::iterator auxiter;
    401 
    402    levelNo++;
    481    for (G4int i=0;i<daughterCount;i++)   // Traverse all the children!
    482      {
    483        const G4VPhysicalVolume* const physvol = volumePtr->GetDaughter(i);
    484        const G4String ModuleName = Modularize(physvol,depth);
    485 
    486        G4Transform3D daughterR;
    487 
    488        if (ModuleName.empty())   // Check if subtree requested to be 
    489      {                         // a separate module!
    490        daughterR = TraverseVolumeTree(physvol->GetLogicalVolume(),depth+1);
    491      }
    492        else
    493      {
    494        G4GDMLWriteStructure writer;
    495        daughterR = writer.Write(ModuleName,physvol->GetLogicalVolume(),
    496                     SchemaLocation,depth+1);
    497      }

    ...
    567    AddExtension(volumeElement, volumePtr);
    568    // Add any possible user defined extension attached to a volume
    569      
    570    AddMaterial(volumePtr->GetMaterial());
    571    // Add the involved materials and solids!
    572      
    573    AddSolid(solidPtr);
    574      
    575    SkinSurfaceCache(GetSkinSurface(volumePtr));
    576      
    577    return R;
    578 }




