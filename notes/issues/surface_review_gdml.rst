Surface Review GDML
=======================

* review of GDML write/read/reconstruction of border/skin surfaces


Observations:

* ptr in names not optional
* use of ptr in names essential to flatten graph into tree ?
* pv names with the old pointers must get revived



::

    854388       <skinsurface name="Tube_surf" surfaceproperty="TubeSurface">
    854389         <volumeref ref="lSurftube0x254b8d0"/>
    854390       </skinsurface>
    854391       <bordersurface name="UpperChimneyTyvekSurface" surfaceproperty="UpperChimneyTyvekOpticalSurface">
    854392         <physvolref ref="pUpperChimneyLS0x2547680"/>
    854393         <physvolref ref="pUpperChimneyTyvek0x2547de0"/>
    854394       </bordersurface>


::

    g4-cls G4GDMLParser
    g4-cls G4GDMLReadStructure
    g4-cls G4GDMLWriteStructure

rg::

   277146     <skinsurface name="Tube_surf" surfaceproperty="TubeSurface">
   277147       <volumeref ref="lSurftube0x254b8d0"/>
   277148     </skinsurface>
   277149     <bordersurface name="UpperChimneyTyvekSurface" surfaceproperty="UpperChimneyTyvekOpticalSurface">
   277150       <physvolref ref="pUpperChimneyLS0x2547680"/>
   277151       <physvolref ref="pUpperChimneyTyvek0x2547de0"/>
   277152     </bordersurface>

   553   <structure>
   554     <volume name="lUpperChimneyLS0x2547ae0">
   555       <materialref ref="LS0x1481580"/>
   556       <solidref ref="Upper_LS_tube0x2547790"/>
   557     </volume>
   558     <volume name="lUpperChimneySteel0x2547bb0">
   559       <materialref ref="Steel0x14aa2a0"/>
   560       <solidref ref="Upper_Steel_tube0x2547890"/>
   561     </volume>
   562     <volume name="lUpperChimneyTyvek0x2547c80">
   563       <materialref ref="Tyvek0x14a7890"/>
   564       <solidref ref="Upper_Tyvek_tube0x2547990"/>
   565     </volume>
   566     <volume name="lUpperChimney0x2547a50">
   567       <materialref ref="Air0x14bb680"/>
   568       <solidref ref="Upper_Chimney0x25476d0"/>
   569       <physvol name="pUpperChimneyLS0x2547680">
   570         <volumeref ref="lUpperChimneyLS0x2547ae0"/>
   571       </physvol>
   572       <physvol name="pUpperChimneySteel0x2547d50">
   573         <volumeref ref="lUpperChimneySteel0x2547bb0"/>
   574       </physvol>
   575       <physvol name="pUpperChimneyTyvek0x2547de0">
   576         <volumeref ref="lUpperChimneyTyvek0x2547c80"/>
   577       </physvol>
   578     </volume>



::


    g4-cls G4GDMLReadStructure

    111 
    112       if (tag != "physvolref")  { continue; }
    113 
    114       if (index==0)
    115         { pv1 = GetPhysvol(GenerateName(RefRead(child))); index++; } else
    116       if (index==1)
    117         { pv2 = GetPhysvol(GenerateName(RefRead(child))); index++; } else
    118       break;
    119    }
    120 
    121    new G4LogicalBorderSurface(Strip(name),pv1,pv2,prop);
    122 }

    816 G4VPhysicalVolume* G4GDMLReadStructure::
    817 GetPhysvol(const G4String& ref) const
    818 {
    819    G4VPhysicalVolume* physvolPtr =
    820      G4PhysicalVolumeStore::GetInstance()->GetVolume(ref,false);
    821 
    822    if (!physvolPtr)
    823    {
    824      G4String error_msg = "Referenced physvol '" + ref + "' was not found!";
    825      G4Exception("G4GDMLReadStructure::GetPhysvol()", "ReadError",
    826                  FatalException, error_msg);
    827    }
    828 
    829    return physvolPtr;
    830 }


::

    g4-cls G4PhysicalVolumeStore

    078     G4VPhysicalVolume* GetVolume(const G4String& name,
     79                                  G4bool verbose=true) const;
     80       // Return the pointer of the first volume in the collection having
     81       // that name.



    160 // ***************************************************************************
    161 // Retrieve the first volume pointer in the container having that name
    162 // ***************************************************************************
    163 //
    164 G4VPhysicalVolume*
    165 G4PhysicalVolumeStore::GetVolume(const G4String& name, G4bool verbose) const
    166 {
    167   for (iterator i=GetInstance()->begin(); i!=GetInstance()->end(); i++)
    168   {
    169     if ((*i)->GetName() == name) { return *i; }
    170   }
    171   if (verbose)
    172   {
    173      std::ostringstream message;
    174      message << "Volume NOT found in store !" << G4endl
    175             << "        Volume " << name << " NOT found in store !" << G4endl
    176             << "        Returning NULL pointer.";
    177      G4Exception("G4PhysicalVolumeStore::GetVolume()",
    178                  "GeomMgt1001", JustWarning, message);
    179   }
    180   return 0;
    181 }



    g4-cls G4GDMLRead

    145    G4String GenerateName(const G4String& name, G4bool strip=false);



::

    382 G4Transform3D G4GDMLWriteStructure::
    383 TraverseVolumeTree(const G4LogicalVolume* const volumePtr, const G4int depth)
    384 {


    443    const G4String name
    444      = GenerateName(tmplv->GetName(), tmplv);
    445    const G4String materialref
    446          = GenerateName(volumePtr->GetMaterial()->GetName(),
    447                         volumePtr->GetMaterial());
    448    const G4String solidref
    449          = GenerateName(solidPtr->GetName(),solidPtr);
    450 
    451    xercesc::DOMElement* volumeElement = NewElement("volume");
    452    volumeElement->setAttributeNode(NewAttribute("name",name));
    453    xercesc::DOMElement* materialrefElement = NewElement("materialref");
    454    materialrefElement->setAttributeNode(NewAttribute("ref",materialref));
    455    volumeElement->appendChild(materialrefElement);
    456    xercesc::DOMElement* solidrefElement = NewElement("solidref");
    457    solidrefElement->setAttributeNode(NewAttribute("ref",solidref));
    458    volumeElement->appendChild(solidrefElement);
    459 
    460    const G4int daughterCount = volumePtr->GetNoDaughters();
    461       
    462    for (G4int i=0;i<daughterCount;i++)   // Traverse all the children!
    463      {
    464        const G4VPhysicalVolume* const physvol = volumePtr->GetDaughter(i);
    465        const G4String ModuleName = Modularize(physvol,depth);
    466       
    467        G4Transform3D daughterR;
    468       
    469        if (ModuleName.empty())   // Check if subtree requested to be 
    470      {                         // a separate module!
    471        daughterR = TraverseVolumeTree(physvol->GetLogicalVolume(),depth+1);
    472      }

    ...

    514        else   // Is it a physvol?
    515          {
    516            G4RotationMatrix rot;
    517            if (physvol->GetFrameRotation() != 0)
    518          {
    519            rot = *(physvol->GetFrameRotation());
    520          }
    521            G4Transform3D P(rot,physvol->GetObjectTranslation());
    522 
    523            PhysvolWrite(volumeElement,physvol,invR*P*daughterR,ModuleName);
    524          }
    525        BorderSurfaceCache(GetBorderSurface(physvol));
    526      }




    338 const G4LogicalBorderSurface*
    339 G4GDMLWriteStructure::GetBorderSurface(const G4VPhysicalVolume* const pvol)
    340 {
    341   G4LogicalBorderSurface* surf = 0;
    342   G4int nsurf = G4LogicalBorderSurface::GetNumberOfBorderSurfaces();
    343   if (nsurf)
    344   {
    345     const G4LogicalBorderSurfaceTable* btable =
    346           G4LogicalBorderSurface::GetSurfaceTable();
    347     std::vector<G4LogicalBorderSurface*>::const_iterator pos;
    348     for (pos = btable->begin(); pos != btable->end(); pos++)
    349     {
    350       if (pvol == (*pos)->GetVolume1())  // just the first in the couple 
    351       {                                  // is enough
    352         surf = *pos; break;
    353       }
    354     }
    355   }
    356   return surf;
    357 }




::

    107 void G4GDMLWriteStructure::PhysvolWrite(xercesc::DOMElement* volumeElement,
    108                                         const G4VPhysicalVolume* const physvol,
    109                                         const G4Transform3D& T,
    110                                         const G4String& ModuleName)
    111 {
    112    HepGeom::Scale3D scale;
    113    HepGeom::Rotate3D rotate;
    114    HepGeom::Translate3D translate;
    115 
    116    T.getDecomposition(scale,rotate,translate);
    117 
    118    const G4ThreeVector scl(scale(0,0),scale(1,1),scale(2,2));
    119    const G4ThreeVector rot = GetAngles(rotate.getRotation());
    120    const G4ThreeVector pos = T.getTranslation();
    121 
    122    const G4String name = GenerateName(physvol->GetName(),physvol);
    123    const G4int copynumber = physvol->GetCopyNo();
    124 
    125    xercesc::DOMElement* physvolElement = NewElement("physvol");
    126    physvolElement->setAttributeNode(NewAttribute("name",name));
    127    if (copynumber) physvolElement->setAttributeNode(NewAttribute("copynumber", copynumber));
    128 
    129    volumeElement->appendChild(physvolElement);
    130 
    131    G4LogicalVolume* lv;
    132    // Is it reflected?
    133    if (reflFactory->IsReflected(physvol->GetLogicalVolume()))
    134      {
    135        lv = reflFactory->GetConstituentLV(physvol->GetLogicalVolume());
    136      }
    137    else
    138      {
    139        lv = physvol->GetLogicalVolume();
    140      }
    141 
    142    const G4String volumeref = GenerateName(lv->GetName(), lv);
    143 
    144    if (ModuleName.empty())
    145    {
    146       xercesc::DOMElement* volumerefElement = NewElement("volumeref");
    147       volumerefElement->setAttributeNode(NewAttribute("ref",volumeref));
    148       physvolElement->appendChild(volumerefElement);
    149    }
    150    else

