g4-1062-geocache-create-reflectivity-assert
=============================================

Issue
------

When loading DYB GDML with Geant4 1062. 

* for all border and skin surfaces the number of values 39 is as expected, BUT all the values are zero
* material properties are not zeroed 

Investigation techniques
--------------------------

::

    geocache-create -D
        asserts with 1062

    CGDMLPropertyTest /tmp/v1.gdml
        load gdml and dump surface and material property values

    X4GDMLReadStructureTest 
        test_readString : parse GDML string literal, attempting to make the problem 
        manifest with a small geometry : so far the issue does not manifest as shown below.

        If a path argument such as /tmp/outpath.gdml is provided the GDML string 
        is written to it, allowing use of CGDMLPropertyTest for the small geometry.

    X4GDMLReadStructure2Test  /tmp/v1.gdml
        X4GDMLReadStructure inherits from the G4GDMLReadStructure giving 
        access to protected intermediates from the GDML parsing such as the matrixMap
        Doing this suggests the values are still instact within the matrixMap

    extg4/tests/G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc
        capture the issue 



Geant4 GDML source
--------------------

Read and Write inheritance chains::

    g4-;g4n-cd source/persistency/gdml/src

    epsilon:include blyth$ grep :\ public *.hh
    G4GDMLMessenger.hh:class G4GDMLMessenger : public G4UImessenger
    G4GDMLParameterisation.hh:class G4GDMLParameterisation : public G4VPVParameterisation
    G4GDMLRead.hh:class G4GDMLErrorHandler : public xercesc::ErrorHandler

    G4GDMLReadDefine.hh:class G4GDMLReadDefine : public G4GDMLRead
    G4GDMLReadMaterials.hh:class G4GDMLReadMaterials : public G4GDMLReadDefine 
    G4GDMLReadSolids.hh:class G4GDMLReadSolids : public G4GDMLReadMaterials
    G4GDMLReadSetup.hh:class G4GDMLReadSetup : public G4GDMLReadSolids
    G4GDMLReadParamvol.hh:class G4GDMLReadParamvol : public G4GDMLReadSetup
    G4GDMLReadStructure.hh:class G4GDMLReadStructure : public G4GDMLReadParamvol

    G4GDMLWriteDefine.hh:class G4GDMLWriteDefine : public G4GDMLWrite
    G4GDMLWriteMaterials.hh:class G4GDMLWriteMaterials : public G4GDMLWriteDefine
    G4GDMLWriteSolids.hh:class G4GDMLWriteSolids : public G4GDMLWriteMaterials
    G4GDMLWriteSetup.hh:class G4GDMLWriteSetup : public G4GDMLWriteSolids
    G4GDMLWriteParamvol.hh:class G4GDMLWriteParamvol : public G4GDMLWriteSetup
    G4GDMLWriteStructure.hh:class G4GDMLWriteStructure : public G4GDMLWriteParamvol

    epsilon:include blyth$ 


G4GDMLParser has reader and writer constituents which are the tops of the inheritance chains::

    G4GDMLReadStructure

    G4GDMLWriteStructure


::

    362 void G4GDMLRead::Read(const G4String& fileName,
    363                             G4bool validation,
    364                             G4bool isModule,
    365                             G4bool strip)
    366 {
    367    dostrip = strip;

    ... loop over top level gdml child elements ...

    441       const G4String tag = Transcode(child->getTagName());
    442 
    443       if (tag=="define")    { DefineRead(child);    } else
    444       if (tag=="materials") { MaterialsRead(child); } else
    445       if (tag=="solids")    { SolidsRead(child);    } else
    446       if (tag=="setup")     { SetupRead(child);     } else
    447       if (tag=="structure") { StructureRead(child); } else
    448       if (tag=="userinfo")  { UserinfoRead(child);  } else
    449       if (tag=="extension") { ExtensionRead(child); }
    450       else
    451       {
    452         G4String error_msg = "Unknown tag in gdml: " + tag;
    453         G4Exception("G4GDMLRead::Read()", "InvalidRead",
    454                     FatalException, error_msg);
    455       }
    456    }
    ...     strip never done for modules 
    468    {
    469       G4cout << "G4GDML: Reading '" << fileName << "' done!" << G4endl;
    470       if (strip)  { StripNames(); }
    471    }
    472 }


    465 void
    466 G4GDMLReadDefine::DefineRead(const xercesc::DOMElement* const defineElement)
    467 {
    ...
    484       const G4String tag = Transcode(child->getTagName());
    485 
    486       if (tag=="constant") { ConstantRead(child); } else
    487       if (tag=="matrix")   { MatrixRead(child); }   else
    488       if (tag=="position") { PositionRead(child); } else
    489       if (tag=="rotation") { RotationRead(child); } else
    490       if (tag=="scale")    { ScaleRead(child); }    else
    491       if (tag=="variable") { VariableRead(child); } else
    492       if (tag=="quantity") { QuantityRead(child); } else
    493       if (tag=="expression") { ExpressionRead(child); }
    494       else
    495       {


::

    079 void G4GDMLEvaluator::DefineMatrix(const G4String& name,
     80                                          G4int coldim,
     81                                          std::vector<G4double> valueList)
     82 {
    ...
    118    else   // Normal matrix
    119    {
    120       const G4int rowdim = size/coldim;
    121 
    122       for (G4int i=0;i<rowdim;i++)
    123       {
    124         for (G4int j=0;j<coldim;j++)
    125         {
    126           std::stringstream MatrixElementNameStream;
    127           MatrixElementNameStream << name << "_" << i << "_" << j;
    128           DefineConstant(MatrixElementNameStream.str(),valueList[coldim*i+j]);
    129         }
    130       }
    131    }
    132 }


::

    632 G4GDMLMatrix G4GDMLReadDefine::GetMatrix(const G4String& ref)
    633 {
    634    if (matrixMap.find(ref) == matrixMap.end())
    635    {
    636      G4String error_msg = "Matrix '"+ref+"' was not found!";
    637      G4Exception("G4GDMLReadDefine::getMatrix()", "ReadError",
    638                  FatalException, error_msg);
    639    }
    640    return matrixMap[ref];
    641 }


    epsilon:src blyth$ grep GetMatrix *.cc
    G4GDMLReadDefine.cc:G4GDMLMatrix G4GDMLReadDefine::GetMatrix(const G4String& ref)
    G4GDMLReadMaterials.cc:      if (attName=="ref")  { matrix = GetMatrix(ref=attValue); }
    G4GDMLReadSolids.cc:      if (attName=="ref")  { matrix = GetMatrix(ref=attValue); }
    epsilon:src blyth$ 
    epsilon:src blyth$ 
    epsilon:src blyth$ grep GetMatrix ../include/*.icc
    G4GDMLMatrix G4GDMLParser::GetMatrix(const G4String& name) const
      return reader->GetMatrix(name);

    epsilon:src blyth$ grep GetMatrix ../include/*.hh
    ../include/G4GDMLParser.hh:   inline G4GDMLMatrix GetMatrix(const G4String& name) const;
    ../include/G4GDMLReadDefine.hh:   G4GDMLMatrix GetMatrix(const G4String&);
    epsilon:src blyth$ 


1062::

    627 void G4GDMLReadMaterials::
    628 PropertyRead(const xercesc::DOMElement* const propertyElement,
    629              G4Material* material)
    630 {
    631    G4String name;
    632    G4String ref;
    633    G4GDMLMatrix matrix;
    ...

    675    G4MaterialPropertiesTable* matprop=material->GetMaterialPropertiesTable();
    676    if (!matprop)
    677    {
    678      matprop = new G4MaterialPropertiesTable();
    679      material->SetMaterialPropertiesTable(matprop);
    680    }
    681    if (matrix.GetCols() == 1)  // constant property assumed
    682    {
    683      matprop->AddConstProperty(Strip(name), matrix.Get(0,0));
    684    }
    685    else  // build the material properties vector
    686    {
    687      G4MaterialPropertyVector* propvect = new G4MaterialPropertyVector();
    688      for (size_t i=0; i<matrix.GetRows(); i++)
    689      {
    690        propvect->InsertValues(matrix.Get(i,0),matrix.Get(i,1));
    691      }
    692      matprop->AddProperty(Strip(name),propvect);
      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  wisps of smoke here : stripping the name could cause the zeroing 
      ^^^^^^^^^^^^ actually it is not a problem it is the reference to the matrix name that could not cope with being 0x stripped 
      ^^^^^^^^^^^^ not the property name
    693    }
    694 }


::

    g4n-cls G4GDMLReadMaterials



DYB has loadsa optical surfaces with same property names like "EFFICIENCY" and "REFLECTIVITY" 
where most of the EFFICIENCY are zero.  But some are non-zero and need to stay that way::

     87     <opticalsurface finish="0" model="0" name="SCB_photocathode_opsurf" type="0" value="1">
     88          <property name="EFFICIENCY" ref="EFFICIENCY0x1d79780"/>   <!-- the non-zero efficiency-->
     89     </opticalsurface>
     ...
     2632     <opticalsurface finish="3" model="1" name="TablePanelSurface" type="0" value="1">
     2633       <property name="REFLECTIVITY" ref="REFLECTIVITY0x1e2eae0"/>
     2634       <property name="EFFICIENCY" ref="EFFICIENCY0x1e2e540"/>
     2635     </opticalsurface>
     2636     <box lunit="mm" name="support_rib1_box0xc0d3bc00x3eb1550" x="3429" y="20" z="230"/>
     2637     <opticalsurface finish="3" model="1" name="SupportRib1Surface" type="0" value="1">
     2638       <property name="REFLECTIVITY" ref="REFLECTIVITY0x1e2b9f0"/>
     2639       <property name="EFFICIENCY" ref="EFFICIENCY0x1e2b450"/>
     2640     </opticalsurface>


1062::

    2477 void G4GDMLReadSolids::
    2478 PropertyRead(const xercesc::DOMElement* const propertyElement,
    2479              G4OpticalSurface* opticalsurface)
    2480 {
    2481    G4String name;
    2482    G4String ref;
    2483    G4GDMLMatrix matrix;

    ....    attribute loop ...  

    2484 
    2505       const G4String attName = Transcode(attribute->getName());
    2506       const G4String attValue = Transcode(attribute->getValue());
    2507 
    2508       if (attName=="name") { name = GenerateName(attValue); } else
    2509       if (attName=="ref")  { matrix = GetMatrix(ref=attValue); }

    ....

    2525    G4MaterialPropertiesTable* matprop=opticalsurface->GetMaterialPropertiesTable();
    2526    if (!matprop)
    2527    {
    2528      matprop = new G4MaterialPropertiesTable();
    2529      opticalsurface->SetMaterialPropertiesTable(matprop);
    2530    }

    2531    if (matrix.GetCols() == 1)  // constant property assumed
    2532    {
    2533      matprop->AddConstProperty(Strip(name), matrix.Get(0,0));
    2534    }
    2535    else  // build the material properties vector
    2536    {
    2537      G4MaterialPropertyVector* propvect;
    2538      // first check if it was already built
    2539      if ( mapOfMatPropVects.find(Strip(name)) == mapOfMatPropVects.end())
    2540      {
    2541           // if not create a new one
    2542           propvect = new G4MaterialPropertyVector();
    2543           for (size_t i=0; i<matrix.GetRows(); i++)
    2544           {
    2545               propvect->InsertValues(matrix.Get(i,0),matrix.Get(i,1));
    2546           }
    2547           // and add it to the list for potential future reuse
    2548           mapOfMatPropVects[Strip(name)] = propvect;
    2549       }


    2550      else
    2551      {
    2552           propvect = mapOfMatPropVects[Strip(name)];
    2553      }


    //////  this assumes the names of all properties are unique across all surfaces 
    //////   THAT IS AN OBVIOUS BUG AND IT IS STILL THERE IN  1070
    //////        https://github.com/Geant4/geant4/blob/master/source/persistency/gdml/src/G4GDMLReadSolids.cc
    //////
    //////   


    
    2554    
    2555      matprop->AddProperty(Strip(name),propvect);
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^  assumption of stripped names being unique very smoky 
    2556    }
    2557 }

1062::

    113 private:
    114   std::map<G4String, G4MaterialPropertyVector*> mapOfMatPropVects;
    115 
    116 };


1042::

    2525    if (matrix.GetRows() == 0) { return; }
    2526 
    2527    G4MaterialPropertiesTable* matprop=opticalsurface->GetMaterialPropertiesTable();
    2528    if (!matprop)
    2529    {
    2530      matprop = new G4MaterialPropertiesTable();
    2531      opticalsurface->SetMaterialPropertiesTable(matprop);
    2532    }
    2533    if (matrix.GetCols() == 1)  // constant property assumed
    2534    {
    2535      matprop->AddConstProperty(Strip(name), matrix.Get(0,0));
    2536    }
    2537    else  // build the material properties vector
    2538    {
    2539      G4MaterialPropertyVector* propvect = new G4MaterialPropertyVector();
    2540      for (size_t i=0; i<matrix.GetRows(); i++)
    2541      {
    2542        propvect->InsertValues(matrix.Get(i,0),matrix.Get(i,1));
    2543      }
    2544      matprop->AddProperty(Strip(name),propvect);
    2545    }
    2546 }


Huh 1042 also stripping. Yes the problem is not the stripping its the use of mapOfMatPropVects::

    2525    if (matrix.GetRows() == 0) { return; }
    2526 
    2527    G4MaterialPropertiesTable* matprop=opticalsurface->GetMaterialPropertiesTable();
    2528    if (!matprop)
    2529    {
    2530      matprop = new G4MaterialPropertiesTable();
    2531      opticalsurface->SetMaterialPropertiesTable(matprop);
    2532    }
    2533    if (matrix.GetCols() == 1)  // constant property assumed
    2534    {
    2535      matprop->AddConstProperty(Strip(name), matrix.Get(0,0));
    2536    }
    2537    else  // build the material properties vector
    2538    {
    2539      G4MaterialPropertyVector* propvect = new G4MaterialPropertyVector();
    2540      for (size_t i=0; i<matrix.GetRows(); i++)
    2541      {
    2542        propvect->InsertValues(matrix.Get(i,0),matrix.Get(i,1));
    2543      }
    2544      matprop->AddProperty(Strip(name),propvect);
    2545    }
    2546 }





Name stripping is a really dumb thing to do::


    097 G4String G4GDMLRead::Strip(const G4String& name) const
     98 {
     99   G4String sname(name);
    100   return sname.remove(sname.find("0x"));
    101 }
    102 
    103 void G4GDMLRead::StripName(G4String& name) const
    104 {
    105   name.remove(name.find("0x"));
    106 }







Geant4 Release Notes
----------------------

* https://geant4-data.web.cern.ch/ReleaseNotes/ReleaseNotes4.10.6.html

GDML:
Added support for writing out assemblies envelopes.
*Improved reading of optical properties reader, by allowing reuse of the 
same G4MaterialPropertyVector object for identical properties.*

   * this change could be implicated : CONFIRMED : Improved -> Broken 

G4GDMLMessenger: fix to avoid UI commands from being broadcasted to worker threads.
G4GDMLRead: fix to avoid double-definition of system units.


* https://github.com/Geant4/geant4/releases


* https://github.com/Geant4/geant4/files/2855943/Patch4.10.4-3.txt

  o Persistency - gdml
    ------------------
    + Clear auxiliary map information in G4GDMLReadStructure::Clear().
      Addressing problem report #2064.
    + Added stripping of invalid characters for names generation in writer classes
      to prevent invalid NCName strings in exported GDML files. Adopt properly
      stripped generated strings for exporting names of optical surfaces.


* https://github.com/Geant4/geant4/files/3088653/Patch4.10.5-1.txt

 o Persistency
    -----------
    + ascii:
      o Fixed shadowing compilation warnings.
    + gdml:
      o Fix in G4GDMLReadStructure::PhysvolRead() to allow correct import of
        recursive assembly structures. Addressing problem report #2141.
      o Added protection to G4GDMLParser for dumping geometry only through
        the master thread. Added extra protection also in reading.
        Addressing problem report #2156.
      o Fixed export of optical surface properties.
        Addressing problem reports #2142 and 2143.


* https://bugzilla-geant4.kek.jp/show_bug.cgi?id=2142

Problem 2142 - A PhysicalVolume with more than two BorderSurface will not be writed to GDML file in correct way.

* https://bugzilla-geant4.kek.jp/show_bug.cgi?id=2143

Problem 2143 - Diffrent OpticalSurface with same MaterialPropertiesTable will not be writed to GDML file in correct way.

* https://bugzilla-geant4.kek.jp/show_bug.cgi?id=2156

Problem 2156 - GDML export shows Segmentation Fault in Multithreading, but runs fine in Sequential


* file:///Users/blyth/Downloads/ReleaseNotes4.10.7.html

Optical
Added second wavelength shifting process, G4OpWLS2, within the same material.
Use new ConstPropertyExists(int) method rather than passing strings.
G4OpRayleigh, G4OpAbsorption, G4OpMieHG, G4OpWLS, G4OpWLS2: moved to new G4OpticalParameters class to control simulation parameters.
G4OpRayleigh: avoid double deletion of property vectors.
G4OpBoundaryProcess: increase geometry tolerance to kCarTolerance.
Fixed reading of Davis LUT data out of bounds. Addressing problem report #2287.
Code cleanup/formatting and improved readability.




GDML schema validation
------------------------

Change gdml element to have the schema location::

    <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://service-spi.web.cern.ch/service-spi/app/releases/GDML/schema/gdml.xsd">

::


    epsilon:extg4 blyth$ X4DumpTest /tmp/v1.gdml > /tmp/out

    ## loads of invalid NCName

    epsilon:extg4 blyth$ grep -v NCName /tmp/out
    2020-12-22 12:20:27.386 INFO  [8494907] [main@24] OKConf::Geant4VersionInteger() : 1042
    2020-12-22 12:20:27.386 INFO  [8494907] [main@32]  parsing /tmp/v1.gdml
    G4GDML: Reading '/tmp/v1.gdml'...
    G4GDML: VALIDATION ERROR! element 'auxiliary' is not allowed for content model '(materialref,solidref,(physvol+|divisionvol|replicavol|paramvol),loop*,auxiliary*)' at line: 8810


         8754     <volume name="/dd/Geometry/AD/lvADE0xc2a78c00x3ef9140">
         8755       <auxiliary auxtype="label" auxvalue="target"/>
         8756       <materialref ref="/dd/Materials/IwsWater0x3e6bf60"/>
         8757       <solidref ref="ade0xc2a74380x3eafdb0"/>
         ...
         8805       <physvol name="/dd/Geometry/AD/lvADE#pvlvMOOverflowTankE20xbf4f7a80x3ef9be0">
         8806         <volumeref ref="/dd/Geometry/AdDetails/lvMOOverflowTankE0xbfa59f00x3f2f570"/>
         8807         <position name="/dd/Geometry/AD/lvADE#pvlvMOOverflowTankE20xbf4f7a80x3ef9be0_pos" unit="mm" x="-1637.57647137626" y="-678.306383867122" z="2153.5"/>
         8808         <rotation name="/dd/Geometry/AD/lvADE#pvlvMOOverflowTankE20xbf4f7a80x3ef9be0_rot" unit="deg" x="0" y="0" z="157.5"/>
         8809       </physvol>
         8810     </volume>

         Try repositioning auxiliary to the last child position.


    G4GDML: VALIDATION ERROR! ID value 'NearPoolCoverSurface' is not unique at line: 31833
    G4GDML: VALIDATION ERROR! ID value 'RSOilSurface' is not unique at line: 31836
    G4GDML: VALIDATION ERROR! ID value 'AdCableTraySurface' is not unique at line: 31839
    G4GDML: VALIDATION ERROR! ID value 'PmtMtTopRingSurface' is not unique at line: 31842
    G4GDML: VALIDATION ERROR! ID value 'PmtMtBaseRingSurface' is not unique at line: 31845
    G4GDML: VALIDATION ERROR! ID value 'PmtMtRib1Surface' is not unique at line: 31848
    G4GDML: VALIDATION ERROR! ID value 'PmtMtRib2Surface' is not unique at line: 31851
    G4GDML: VALIDATION ERROR! ID value 'PmtMtRib3Surface' is not unique at line: 31854
    G4GDML: VALIDATION ERROR! ID value 'LegInIWSTubSurface' is not unique at line: 31857
    G4GDML: VALIDATION ERROR! ID value 'TablePanelSurface' is not unique at line: 31860
    G4GDML: VALIDATION ERROR! ID value 'SupportRib1Surface' is not unique at line: 31863
    G4GDML: VALIDATION ERROR! ID value 'SupportRib5Surface' is not unique at line: 31866
    G4GDML: VALIDATION ERROR! ID value 'SlopeRib1Surface' is not unique at line: 31869
    G4GDML: VALIDATION ERROR! ID value 'SlopeRib5Surface' is not unique at line: 31872
    G4GDML: VALIDATION ERROR! ID value 'ADVertiCableTraySurface' is not unique at line: 31875
    G4GDML: VALIDATION ERROR! ID value 'ShortParCableTraySurface' is not unique at line: 31878
    G4GDML: VALIDATION ERROR! ID value 'NearInnInPiperSurface' is not unique at line: 31881
    G4GDML: VALIDATION ERROR! ID value 'NearInnOutPiperSurface' is not unique at line: 31884
    G4GDML: VALIDATION ERROR! ID value 'LegInOWSTubSurface' is not unique at line: 31887
    G4GDML: VALIDATION ERROR! ID value 'UnistrutRib6Surface' is not unique at line: 31890
    G4GDML: VALIDATION ERROR! ID value 'UnistrutRib7Surface' is not unique at line: 31893
    G4GDML: VALIDATION ERROR! ID value 'UnistrutRib3Surface' is not unique at line: 31896
    G4GDML: VALIDATION ERROR! ID value 'UnistrutRib5Surface' is not unique at line: 31899
    G4GDML: VALIDATION ERROR! ID value 'UnistrutRib4Surface' is not unique at line: 31902
    G4GDML: VALIDATION ERROR! ID value 'UnistrutRib1Surface' is not unique at line: 31905
    G4GDML: VALIDATION ERROR! ID value 'UnistrutRib2Surface' is not unique at line: 31908
    G4GDML: VALIDATION ERROR! ID value 'UnistrutRib8Surface' is not unique at line: 31911
    G4GDML: VALIDATION ERROR! ID value 'UnistrutRib9Surface' is not unique at line: 31914
    G4GDML: VALIDATION ERROR! ID value 'TopShortCableTraySurface' is not unique at line: 31917
    G4GDML: VALIDATION ERROR! ID value 'TopCornerCableTraySurface' is not unique at line: 31920
    G4GDML: VALIDATION ERROR! ID value 'VertiCableTraySurface' is not unique at line: 31923
    G4GDML: VALIDATION ERROR! ID value 'NearOutInPiperSurface' is not unique at line: 31926
    G4GDML: VALIDATION ERROR! ID value 'NearOutOutPiperSurface' is not unique at line: 31929
    G4GDML: VALIDATION ERROR! ID value 'LegInDeadTubSurface' is not unique at line: 31932
    G4GDML: VALIDATION ERROR! ID value 'ESRAirSurfaceTop' is not unique at line: 31935
    G4GDML: VALIDATION ERROR! ID value 'ESRAirSurfaceBot' is not unique at line: 31939
    G4GDML: VALIDATION ERROR! ID value 'SSTOilSurface' is not unique at line: 31943
    G4GDML: VALIDATION ERROR! ID value 'SSTWaterSurfaceNear1' is not unique at line: 31947
    G4GDML: VALIDATION ERROR! ID value 'SSTWaterSurfaceNear2' is not unique at line: 31951
    G4GDML: VALIDATION ERROR! ID value 'NearIWSCurtainSurface' is not unique at line: 31955
    G4GDML: VALIDATION ERROR! ID value 'NearOWSLinerSurface' is not unique at line: 31959
    G4GDML: VALIDATION ERROR! ID value 'NearDeadLinerSurface' is not unique at line: 31963
    G4GDML: VALIDATION ERROR! element 'userinfo' is not allowed for content model '(define,materials,solids,structure,userinfo?,setup+)' at line: 31990

        31989 
        31990 </gdml>

        complaint at last line  : move userinfo between structure and setup


    G4GDML: Reading userinfo...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/v1.gdml' done!
    mt,sk,bs
    G4VERSION_NUMBER 1042


After moving auxiliary and userinfo get two types of complaint.

1. invalid NCName
2. ID value is not unique

::

     grep -v NCName /tmp/out2 | grep -v not\ unique

NCName
-------

* https://stackoverflow.com/questions/1631396/what-is-an-xsncname-type-and-when-should-it-be-used

The practical restrictions of NCName are that it cannot contain several symbol
characters like :, @, $, %, &, /, +, ,, ;, whitespace characters or different
parenthesis. Furthermore an NCName cannot begin with a number, dot or minus
character although they can appear later in an NCName.




X4GDMLReadStructureTest with 1062 : zeroing not happening with simple gdml
-----------------------------------------------------------------------------

::

    epsilon:extg4 charles$ X4GDMLReadStructureTest
    G4GDML: Reading '/tmp/charles/opticks/X4GDMLReadStructure__WriteGDMLStringToTmpPath_3c55-c24c-235a-3dc3'...
    G4GDML: Reading userinfo...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/charles/opticks/X4GDMLReadStructure__WriteGDMLStringToTmpPath_3c55-c24c-235a-3dc3' done!
    mt,sk,bs
    G4VERSION_NUMBER 1062
    G4VERSION_TAG    $Name: geant4-10-06-patch-02 $
    G4Version        $Name: geant4-10-06-patch-02 $
    G4Date           (29-May-2020)
    mt
    nmat 2 tab.size 2
    /dd/Materials/fakePyrex  NULL mpt 
    /dd/Materials/fakeVacuum  NULL mpt 
    sk
    nlss 0 tab.size 0
    bs
    nlbs 2 tab.size 2
    SCB_photocathode_logsurf1 23
    i     4 pidx     4              EFFICIENCY pvec 0x7fc5b585b730 plen  39 0.0001 0.0001 0.000440306 0.000782349 0.00112439 ... 0 0 0 0  mn 0 mx 0.24
    SCB_photocathode_logsurf2 23
    i     4 pidx     4              EFFICIENCY pvec 0x7fc5b585b730 plen  39 0.0001 0.0001 0.000440306 0.000782349 0.00112439 ... 0 0 0 0  mn 0 mx 0.24
    G4VERSION_NUMBER 1062
    G4VERSION_TAG    $Name: geant4-10-06-patch-02 $
    G4Version        $Name: geant4-10-06-patch-02 $
    G4Date           (29-May-2020)
    epsilon:extg4 charles$ 



Lack of a non-zero efficiency causes a REFLECTIVITY assert for geocache-create with 1062
------------------------------------------------------------------------------------------

* assert occurs because non-sensor surfaces are required to have a REFLECTIVITY : if the 
  efficiency values were not all zero then it would be classified as a sensor surface
  and the assert would not be tripped 


::

    epsilon:opticks charles$ geocache-create -D
    === o-cmdline-parse 1 : START
    === o-cmdline-specials 1 :
    === o-cmdline-specials 1 :
    === o-cmdline-binary-match 1 : finding 1st argument with associated binary
    === o-cmdline-binary-match 1 : --okx4test
    === o-cmdline-parse 1 : DONE

    2020-12-20 19:13:31.787 INFO  [6787389] [X4PhysicalVolume::convertMaterials@255]  num_materials 36 num_material_with_efficiency 1
    2020-12-20 19:13:31.787 INFO  [6787389] [GMaterialLib::dumpSensitiveMaterials@1230] X4PhysicalVolume::convertMaterials num_sensitive_materials 1
     0 :                       Bialkali
    Assertion failed: (_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "), function createStandardSurface, file /Users/charles/opticks/ggeo/GSurfaceLib.cc, line 595.
    Process 73978 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff77d74b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff77d74b66 <+10>: jae    0x7fff77d74b70            ; <+20>
        0x7fff77d74b68 <+12>: movq   %rax, %rdi
        0x7fff77d74b6b <+15>: jmp    0x7fff77d6bae9            ; cerror_nocancel
        0x7fff77d74b70 <+20>: retq   
    Target 0: (OKX4Test) stopped.

    Process 73978 launched: '/Users/charles/local/opticks/lib/OKX4Test' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff77d74b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff77f3f080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff77cd01ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff77c981ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010cc0be84 libGGeo.dylib`GSurfaceLib::createStandardSurface(this=0x0000000111b17690, src=0x0000000111a68690) at GSurfaceLib.cc:595
        frame #5: 0x000000010cc0ae42 libGGeo.dylib`GSurfaceLib::add(this=0x0000000111b17690, surf=0x0000000111a68690) at GSurfaceLib.cc:486
        frame #6: 0x000000010cc0ad84 libGGeo.dylib`GSurfaceLib::addBorderSurface(this=0x0000000111b17690, surf=0x0000000111a68690, pv1="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0", pv2="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720", direct=false) at GSurfaceLib.cc:374
        frame #7: 0x000000010cc0aa48 libGGeo.dylib`GSurfaceLib::add(this=0x0000000111b17690, raw=0x0000000111a68690) at GSurfaceLib.cc:358
        frame #8: 0x00000001038ba51e libExtG4.dylib`X4LogicalBorderSurfaceTable::init(this=0x00007ffeefbfd478) at X4LogicalBorderSurfaceTable.cc:66
        frame #9: 0x00000001038ba1d4 libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd478, dst=0x0000000111b17690) at X4LogicalBorderSurfaceTable.cc:45
        frame #10: 0x00000001038ba18d libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd478, dst=0x0000000111b17690) at X4LogicalBorderSurfaceTable.cc:44
        frame #11: 0x00000001038ba15c libExtG4.dylib`X4LogicalBorderSurfaceTable::Convert(dst=0x0000000111b17690) at X4LogicalBorderSurfaceTable.cc:37
        frame #12: 0x00000001038c6f63 libExtG4.dylib`X4PhysicalVolume::convertSurfaces(this=0x00007ffeefbfe558) at X4PhysicalVolume.cc:282
        frame #13: 0x00000001038c670f libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfe558) at X4PhysicalVolume.cc:192
        frame #14: 0x00000001038c63f5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe558, ggeo=0x0000000111b14760, top=0x0000000118d44660) at X4PhysicalVolume.cc:177
        frame #15: 0x00000001038c56b5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe558, ggeo=0x0000000111b14760, top=0x0000000118d44660) at X4PhysicalVolume.cc:168
        frame #16: 0x0000000100015707 OKX4Test`main(argc=15, argv=0x00007ffeefbfed58) at OKX4Test.cc:108
        frame #17: 0x00007fff77c24015 libdyld.dylib`start + 1
        frame #18: 0x00007fff77c24015 libdyld.dylib`start + 1
    (lldb) 



::

     185 void X4PhysicalVolume::init()
     186 {
     187     LOG(LEVEL) << "[" ;
     188     LOG(LEVEL) << " query : " << m_query->desc() ;
     189 
     190 
     191     convertMaterials();   // populate GMaterialLib
     192     convertSurfaces();    // populate GSurfaceLib
     193     closeSurfaces();
     194     convertSolids();      // populate GMeshLib with GMesh converted from each G4VSolid (postorder traverse processing first occurrence of G4LogicalVolume)  
     195     convertStructure();   // populate GNodeLib with GVolume converted from each G4VPhysicalVolume (preorder traverse) 
     196     convertCheck();
     197 
     198     postConvert();
     199 
     200     LOG(LEVEL) << "]" ;
     201 }

     275 void X4PhysicalVolume::convertSurfaces()
     276 {
     277     LOG(LEVEL) << "[" ;
     278 
     279     size_t num_surf0 = m_slib->getNumSurfaces() ;
     280     assert( num_surf0 == 0 );
     281 
     282     X4LogicalBorderSurfaceTable::Convert(m_slib);
     283     size_t num_lbs = m_slib->getNumSurfaces() ;
     284 
     285     X4LogicalSkinSurfaceTable::Convert(m_slib);
     286     size_t num_sks = m_slib->getNumSurfaces() - num_lbs ;
     287 
     288     m_slib->addPerfectSurfaces();
     289     m_slib->dumpSurfaces("X4PhysicalVolume::convertSurfaces");
     290 
     291     m_slib->collectSensorIndices();
     292     m_slib->dumpSensorIndices("X4PhysicalVolume::convertSurfaces");
     293 
     294     LOG(LEVEL) 
     295         << "]" 
     296         << " num_lbs " << num_lbs
     297         << " num_sks " << num_sks
     298         ;  
     299 
     300 }

     40 X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(GSurfaceLib* dst )
     41     :
     42     m_src(G4LogicalBorderSurface::GetSurfaceTable()),
     43     m_dst(dst)
     44 {
     45     init();
     46 }
     47 
     48 
     49 void X4LogicalBorderSurfaceTable::init()
     50 {
     51     unsigned num_src = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ;
     52     assert( num_src == m_src->size() );
     53 
     54     LOG(LEVEL) << " NumberOfBorderSurfaces " << num_src ;
     55 
     56     for(size_t i=0 ; i < m_src->size() ; i++)
     57     {
     58         G4LogicalBorderSurface* src = (*m_src)[i] ;
     59 
     60         LOG(LEVEL) << src->GetName() ;
     61 
     62         GBorderSurface* dst = X4LogicalBorderSurface::Convert( src );
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ convert missing REFLECTIVITY with 1062 ??  
     63 
     64         assert( dst );
     65 
     66         m_dst->add(dst) ; // GSurfaceLib
     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
     67     }
     68 }


Darwin.charles.1062::

    X4PhysicalVolume=INFO X4LogicalBorderSurfaceTable=INFO geocache-create -D 
    ...
    2020-12-20 19:30:17.643 INFO  [6804160] [X4PhysicalVolume::init@187] [
    2020-12-20 19:30:17.643 INFO  [6804160] [X4PhysicalVolume::init@188]  query :  queryType undefined query_string all query_name NULL query_index 0 query_depth 0 no_selection 1
    2020-12-20 19:30:17.648 INFO  [6804160] [X4PhysicalVolume::convertMaterials@255]  num_materials 36 num_material_with_efficiency 1
    2020-12-20 19:30:17.648 INFO  [6804160] [GMaterialLib::dumpSensitiveMaterials@1230] X4PhysicalVolume::convertMaterials num_sensitive_materials 1
     0 :                       Bialkali
    2020-12-20 19:30:17.648 INFO  [6804160] [X4PhysicalVolume::convertSurfaces@277] [
    2020-12-20 19:30:17.648 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@54]  NumberOfBorderSurfaces 10
    2020-12-20 19:30:17.648 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] ESRAirSurfaceTop
    2020-12-20 19:30:17.649 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] ESRAirSurfaceBot
    2020-12-20 19:30:17.649 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] SSTOilSurface
    2020-12-20 19:30:17.649 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] SSTWaterSurfaceNear1
    2020-12-20 19:30:17.649 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] SSTWaterSurfaceNear2
    2020-12-20 19:30:17.649 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] NearIWSCurtainSurface
    2020-12-20 19:30:17.650 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] NearOWSLinerSurface
    2020-12-20 19:30:17.650 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] NearDeadLinerSurface
    2020-12-20 19:30:17.650 INFO  [6804160] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    Assertion failed: (_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "), function createStandardSurface, file /Users/charles/opticks/ggeo/GSurfaceLib.cc, line 595.
    ...
    (lldb) bt
        frame #3: 0x00007fff77c981ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010cc0be84 libGGeo.dylib`GSurfaceLib::createStandardSurface(this=0x0000000111ac5540, src=0x0000000111bb9160) at GSurfaceLib.cc:595
        frame #5: 0x000000010cc0ae42 libGGeo.dylib`GSurfaceLib::add(this=0x0000000111ac5540, surf=0x0000000111bb9160) at GSurfaceLib.cc:486
        frame #6: 0x000000010cc0ad84 libGGeo.dylib`GSurfaceLib::addBorderSurface(this=0x0000000111ac5540, surf=0x0000000111bb9160, pv1="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0", pv2="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720", direct=false) at GSurfaceLib.cc:374
        frame #7: 0x000000010cc0aa48 libGGeo.dylib`GSurfaceLib::add(this=0x0000000111ac5540, raw=0x0000000111bb9160) at GSurfaceLib.cc:358
        frame #8: 0x00000001038ba5ee libExtG4.dylib`X4LogicalBorderSurfaceTable::init(this=0x00007ffeefbfd5b8) at X4LogicalBorderSurfaceTable.cc:66
        frame #9: 0x00000001038ba2a4 libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd5b8, dst=0x0000000111ac5540) at X4LogicalBorderSurfaceTable.cc:45
        frame #10: 0x00000001038ba25d libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd5b8, dst=0x0000000111ac5540) at X4LogicalBorderSurfaceTable.cc:44
        frame #11: 0x00000001038ba22c libExtG4.dylib`X4LogicalBorderSurfaceTable::Convert(dst=0x0000000111ac5540) at X4LogicalBorderSurfaceTable.cc:37
        frame #12: 0x00000001038c7030 libExtG4.dylib`X4PhysicalVolume::convertSurfaces(this=0x00007ffeefbfe518) at X4PhysicalVolume.cc:282
        frame #13: 0x00000001038c67df libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfe518) at X4PhysicalVolume.cc:192
        frame #14: 0x00000001038c64c5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe518, ggeo=0x0000000111ac25f0, top=0x0000000111b8cee0) at X4PhysicalVolume.cc:177
        frame #15: 0x00000001038c5785 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe518, ggeo=0x0000000111ac25f0, top=0x0000000111b8cee0) at X4PhysicalVolume.cc:168
        frame #16: 0x0000000100015707 OKX4Test`main(argc=15, argv=0x00007ffeefbfed10) at OKX4Test.cc:108
        frame #17: 0x00007fff77c24015 libdyld.dylib`start + 1
    (lldb) 

::

     483 void GSurfaceLib::add(GPropertyMap<float>* surf)
     484 {
     485     assert(!isClosed());
     486     GPropertyMap<float>* ssurf = createStandardSurface(surf) ;
     487     addDirect(ssurf);
     488 }
     489 
     490 void GSurfaceLib::addDirect(GPropertyMap<float>* surf)
     491 {
     492     assert(!isClosed());
     493     m_surfaces.push_back(surf);
     494 }

::

     548 
     549 GPropertyMap<float>* GSurfaceLib::createStandardSurface(GPropertyMap<float>* src)
     550 {
     551     GProperty<float>* _detect           = NULL ;
     552     GProperty<float>* _absorb           = NULL ;
     553     GProperty<float>* _reflect_specular = NULL ;
     554     GProperty<float>* _reflect_diffuse  = NULL ;

     ...
     572         if(src->isSensor())  // this means it has non-zero EFFICIENCY or detect property
     573         {
     574             GProperty<float>* _EFFICIENCY = src->getProperty(EFFICIENCY);
     575             assert(_EFFICIENCY && os && "sensor surfaces must have an efficiency" );
     576 
     577             if(m_fake_efficiency >= 0.f && m_fake_efficiency <= 1.0f)
     578             {
     579                 _detect           = makeConstantProperty(m_fake_efficiency) ;
     580                 _absorb           = makeConstantProperty(1.0-m_fake_efficiency);
     581                 _reflect_specular = makeConstantProperty(0.0);
     582                 _reflect_diffuse  = makeConstantProperty(0.0);
     583             }
     584             else
     585             {
     586                 _detect = _EFFICIENCY ;
     587                 _absorb = GProperty<float>::make_one_minus( _detect );
     588                 _reflect_specular = makeConstantProperty(0.0);
     589                 _reflect_diffuse  = makeConstantProperty(0.0);
     590             }
     591         }
     592         else
     593         {
     594             GProperty<float>* _REFLECTIVITY = src->getProperty(REFLECTIVITY);
     595             assert(_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity " );
     596 

     276 template <class T>
     277 bool GPropertyMap<T>::isSensor()
     278 {
     279     return hasNonZeroProperty(EFFICIENCY) || hasNonZeroProperty(detect) ;
     280 }

     785 template <typename T>
     786 bool GPropertyMap<T>::hasNonZeroProperty(const char* pname)
     787 {
     788      if(!hasProperty(pname)) return false ;
     789      GProperty<T>* prop = getProperty(pname);
     790      return !prop->isZero();
     791 }


     40 GBorderSurface* X4LogicalBorderSurface::Convert(const G4LogicalBorderSurface* src)
     41 {
     42     const char* name = X4::Name( src );
     43     size_t index = X4::GetOpticksIndex( src ) ;
     44 
     45     G4OpticalSurface* os = dynamic_cast<G4OpticalSurface*>(src->GetSurfaceProperty());
     46     assert( os );
     47     GOpticalSurface* optical_surface = X4OpticalSurface::Convert(os);   ;
     48     assert( optical_surface );
     49 
     50     GBorderSurface* dst = new GBorderSurface( name, index, optical_surface) ;
     51     // standard domain is set by GBorderSurface::init
     52 
     53     X4LogicalSurface::Convert( dst, src);
     54 
     55     const G4VPhysicalVolume* pv1 = src->GetVolume1();
     56     const G4VPhysicalVolume* pv2 = src->GetVolume2();

     34 void X4LogicalSurface::Convert(GPropertyMap<float>* dst,  const G4LogicalSurface* src)
     35 {
     36     LOG(LEVEL) << "[" ; 
     37     const G4SurfaceProperty*  psurf = src->GetSurfaceProperty() ;   
     38     const G4OpticalSurface* opsurf = dynamic_cast<const G4OpticalSurface*>(psurf);
     39     assert( opsurf );   
     40     G4MaterialPropertiesTable* mpt = opsurf->GetMaterialPropertiesTable() ;
     41     X4MaterialPropertiesTable::Convert( dst, mpt );
     42     
     43     LOG(LEVEL) << "]" ;
     44 }







Darwin.blyth.1042::

    2020-12-20 19:32:44.516 INFO  [6807645] [X4PhysicalVolume::init@187] [
    2020-12-20 19:32:44.516 INFO  [6807645] [X4PhysicalVolume::init@188]  query :  queryType undefined query_string all query_name NULL query_index 0 query_depth 0 no_selection 1
    2020-12-20 19:32:44.521 INFO  [6807645] [X4PhysicalVolume::convertMaterials@255]  num_materials 36 num_material_with_efficiency 1
    2020-12-20 19:32:44.521 INFO  [6807645] [GMaterialLib::dumpSensitiveMaterials@1230] X4PhysicalVolume::convertMaterials num_sensitive_materials 1
     0 :                       Bialkali
    2020-12-20 19:32:44.522 INFO  [6807645] [X4PhysicalVolume::convertSurfaces@277] [
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@54]  NumberOfBorderSurfaces 10
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] ESRAirSurfaceTop
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] ESRAirSurfaceBot
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] SSTOilSurface
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] SSTWaterSurfaceNear1
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] SSTWaterSurfaceNear2
    2020-12-20 19:32:44.522 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] NearIWSCurtainSurface
    2020-12-20 19:32:44.523 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] NearOWSLinerSurface
    2020-12-20 19:32:44.523 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] NearDeadLinerSurface
    2020-12-20 19:32:44.523 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    2020-12-20 19:32:44.523 INFO  [6807645] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf2
    2020-12-20 19:32:44.526 INFO  [6807645] [GSurfaceLib::dumpSurfaces@749] X4PhysicalVolume::convertSurfaces num_surfaces 48
     index :  0 is_sensor : N type :        bordersurface name :                                   ESRAirSurfaceTop bpv1 /dd/Geometry/AdDetails/lvTopReflector#pvTopRefGap0xc2664680x3eeae20 bpv2 /dd/Geometry/AdDetails/lvTopRefGap#pvTopESR0xc4110d00x3eeab80 .
     index :  1 is_sensor : N type :        bordersurface name :                                   ESRAirSurfaceBot bpv1 /dd/Geometry/AdDetails/lvBotReflector#pvBotRefGap0xbfa64580x3eeb320 bpv2 /dd/Geometry/AdDetails/lvBotRefGap#pvBotESR0xbf9bd080x3eeb080 .
     index :  2 is_sensor : N type :        bordersurface name :                                      SSTOilSurface bpv1 /dd/Geometry/AD/lvSST#pvOIL0xc2415100x3f0b6a0 bpv2 /dd/Geometry/AD/lvADE#pvSST0xc128d900x3ef9100 .
     index :  3 is_sensor : N type :        bordersurface name :                               SSTWaterSurfaceNear1 bpv1 /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE10xc2cf5280x3efb9c0 bpv2 /dd/Geometry/AD/lvADE#pvSST0xc128d900x3ef9100 .
     index :  4 is_sensor : N type :        bordersurface name :                               SSTWaterSurfaceNear2 bpv1 /dd/Geometry/Pool/lvNearPoolIWS#pvNearADE20xc0479c80x3efbb80 bpv2 /dd/Geometry/AD/lvADE#pvSST0xc128d900x3ef9100 .
     index :  5 is_sensor : N type :        bordersurface name :                              NearIWSCurtainSurface bpv1 /dd/Geometry/Pool/lvNearPoolCurtain#pvNearPoolIWS0xc15a4980x3fa6c80 bpv2 /dd/Geometry/Pool/lvNearPoolOWS#pvNearPoolCurtain0xc5c5f200x3fa9070 .
     index :  6 is_sensor : N type :        bordersurface name :                                NearOWSLinerSurface bpv1 /dd/Geometry/Pool/lvNearPoolLiner#pvNearPoolOWS0xbf55b100x4128cf0 bpv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 .
     index :  7 is_sensor : N type :        bordersurface name :                               NearDeadLinerSurface bpv1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c0180x412b090 bpv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 .
     index :  8 is_sensor : Y type :        bordersurface name :                          SCB_photocathode_logsurf1 bpv1 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 bpv2 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 .
     index :  9 is_sensor : Y type :        bordersurface name :                          SCB_photocathode_logsurf2 bpv1 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 bpv2 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 .
     index : 10 is_sensor : N type :          skinsurface name :                               NearPoolCoverSurface sslv lvNearTopCover0xc1370600x3ebf2d0 .
     index : 11 is_sensor : N type :          skinsurface name :                                       RSOilSurface sslv lvRadialShieldUnit0xc3d7ec00x3eea9d0 .
     index : 12 is_sensor : N type :          skinsurface name :                                 AdCableTraySurface sslv lvAdVertiCableTray0xc3a27f00x3f2ce70 .



Darwin.charles.1062 is_sensor not set for SCB_photocathode_logsurf1 whereas is is in 1042::

    2020-12-20 20:24:14.929 INFO  [6861666] [GSurfaceLib::add@345]  GBorderSurface  name NearDeadLinerSurface pv1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c0180x412b090 pv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 keys REFLECTIVITY EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:24:14.929 INFO  [6861666] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    2020-12-20 20:24:14.930 INFO  [6861666] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x118c87e80
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:24:14.930 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:24:14.931 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:24:14.931 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@86]  pns 23 pns_null 22
    2020-12-20 20:24:14.931 INFO  [6861666] [X4MaterialPropertiesTable::AddProperties@122]  cpns 33 cpns_null 33
    2020-12-20 20:24:14.931 INFO  [6861666] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:24:14.931 INFO  [6861666] [*X4LogicalBorderSurface::Convert@61] SCB_photocathode_logsurf1 is_sensor 0
    2020-12-20 20:24:14.931 INFO  [6861666] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf1 pv1 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 pv2 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 keys EFFICIENCY has_EFFICIENCY 1
    Assertion failed: (_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "), function createStandardSurface, file /Users/charles/opticks/ggeo/GSurfaceLib.cc, line 597.
    Process 81029 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff77d74b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff77d74b66 <+10>: jae    0x7fff77d74b70            ; <+20>
        0x7fff77d74b68 <+12>: movq   %rax, %rdi
        0x7fff77d74b6b <+15>: jmp    0x7fff77d6bae9            ; cerror_nocancel
        0x7fff77d74b70 <+20>: retq   
    Target 0: (OKX4Test) stopped.



Darwin.blyth.1042::


    X4PhysicalVolume=INFO X4LogicalBorderSurfaceTable=INFO X4LogicalBorderSurface=INFO GSurfaceLib=INFO X4LogicalSurface=INFO X4MaterialPropertiesTable=INFO  geocache-create -D 

    2020-12-20 20:24:45.281 INFO  [6862364] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:24:45.281 INFO  [6862364] [*X4LogicalBorderSurface::Convert@61] SCB_photocathode_logsurf1 is_sensor 1
    2020-12-20 20:24:45.281 INFO  [6862364] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf1 pv1 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 pv2 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 keys EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:24:45.282 INFO  [6862364] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf2
    2020-12-20 20:24:45.282 INFO  [6862364] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x115c74c30
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@86]  pns 23 pns_null 22
    2020-12-20 20:24:45.282 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@122]  cpns 33 cpns_null 33
    2020-12-20 20:24:45.282 INFO  [6862364] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:24:45.282 INFO  [6862364] [*X4LogicalBorderSurface::Convert@61] SCB_photocathode_logsurf2 is_sensor 1
    2020-12-20 20:24:45.282 INFO  [6862364] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf2 pv1 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 pv2 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 keys EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:24:45.283 INFO  [6862364] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :      0x115c589d0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x115c58cd0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:24:45.283 INFO  [6862364] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0




Darwin.charles.1062::

    2020-12-20 20:01:15.835 INFO  [6835068] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:01:15.835 INFO  [6835068] [GSurfaceLib::add@345]  GBorderSurface  name NearDeadLinerSurface pv1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c0180x412b090 pv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 keys REFLECTIVITY EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:01:15.835 INFO  [6835068] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    2020-12-20 20:01:15.835 INFO  [6835068] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:01:15.835 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:01:15.835 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:01:15.835 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:01:15.835 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:01:15.835 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x118df7880
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@86]  pns 23 pns_null 22
    2020-12-20 20:01:15.836 INFO  [6835068] [X4MaterialPropertiesTable::AddProperties@122]  cpns 33 cpns_null 33
    2020-12-20 20:01:15.836 INFO  [6835068] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:01:15.836 INFO  [6835068] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf1 pv1 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 pv2 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 keys EFFICIENCY has_EFFICIENCY 1
    Assertion failed: (_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "), function createStandardSurface, file /Users/charles/opticks/ggeo/GSurfaceLib.cc, line 595.
    Process 77034 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff77d74b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff77d74b66 <+10>: jae    0x7fff77d74b70            ; <+20>
        0x7fff77d74b68 <+12>: movq   %rax, %rdi
        0x7fff77d74b6b <+15>: jmp    0x7fff77d6bae9            ; cerror_nocancel
        0x7fff77d74b70 <+20>: retq   
    Target 0: (OKX4Test) stopped.

    Process 77034 launched: '/Users/charles/local/opticks/lib/OKX4Test' (x86_64)


Possibly is_sensor is what is different.


1062 zero EFFICIENCY::

    2020-12-20 20:44:10.015 INFO  [6882139] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    2020-12-20 20:44:10.015 INFO  [6882139] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x118498a30
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@87] X4MaterialPropertiesTable::AddProperties.EFFICIENCY zero  constant: 0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:44:10.015 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 22
    2020-12-20 20:44:10.016 INFO  [6882139] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2020-12-20 20:44:10.016 INFO  [6882139] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:44:10.016 INFO  [6882139] [*X4LogicalBorderSurface::Convert@61] SCB_photocathode_logsurf1 is_sensor 0
    2020-12-20 20:44:10.016 INFO  [6882139] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf1 pv1 /


1042 non-zero EFFICIENCY::

    2020-12-20 20:47:26.773 INFO  [6886110] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:47:26.773 INFO  [6886110] [*X4LogicalBorderSurface::Convert@61] NearDeadLinerSurface is_sensor 0
    2020-12-20 20:47:26.773 INFO  [6886110] [GSurfaceLib::add@345]  GBorderSurface  name NearDeadLinerSurface pv1 /dd/Geometry/Sites/lvNearHallBot#pvNearPoolDead0xc13c0180x412b090 pv2 /dd/Geometry/Pool/lvNearPoolDead#pvNearPoolLiner0xbf4b2700x4129b20 keys REFLECTIVITY EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:47:26.773 INFO  [6886110] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf1
    2020-12-20 20:47:26.773 INFO  [6886110] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x115d22780
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@87] X4MaterialPropertiesTable::AddProperties.EFFICIENCY range: 0 : 0.24
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:47:26.773 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 22
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2020-12-20 20:47:26.774 INFO  [6886110] [X4LogicalSurface::Convert@43] ]
    2020-12-20 20:47:26.774 INFO  [6886110] [*X4LogicalBorderSurface::Convert@61] SCB_photocathode_logsurf1 is_sensor 1
    2020-12-20 20:47:26.774 INFO  [6886110] [GSurfaceLib::add@345]  GBorderSurface  name SCB_photocathode_logsurf1 pv1 /dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0 pv2 /dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720 keys EFFICIENCY has_EFFICIENCY 1
    2020-12-20 20:47:26.774 INFO  [6886110] [X4LogicalBorderSurfaceTable::init@60] SCB_photocathode_logsurf2
    2020-12-20 20:47:26.774 INFO  [6886110] [X4LogicalSurface::Convert@36] [
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                         RINDEX pidx :     0 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   REFLECTIVITY pidx :     1 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     REALRINDEX pidx :     2 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                IMAGINARYRINDEX pidx :     3 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                     EFFICIENCY pidx :     4 pvec :      0x115d22780
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@87] X4MaterialPropertiesTable::AddProperties.EFFICIENCY range: 0 : 0.24
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  TRANSMITTANCE pidx :     5 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :           SPECULARLOBECONSTANT pidx :     6 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :          SPECULARSPIKECONSTANT pidx :     7 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :            BACKSCATTERCONSTANT pidx :     8 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       GROUPVEL pidx :     9 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                          MIEHG pidx :    10 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                       RAYLEIGH pidx :    11 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSCOMPONENT pidx :    12 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                   WLSABSLENGTH pidx :    13 pvec :              0x0
    2020-12-20 20:47:26.774 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                      ABSLENGTH pidx :    14 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  FASTCOMPONENT pidx :    15 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :                  SLOWCOMPONENT pidx :    16 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :       PROTONSCINTILLATIONYIELD pidx :    17 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :     DEUTERONSCINTILLATIONYIELD pidx :    18 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :       TRITONSCINTILLATIONYIELD pidx :    19 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :        ALPHASCINTILLATIONYIELD pidx :    20 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :          IONSCINTILLATIONYIELD pidx :    21 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@68]  pname :     ELECTRONSCINTILLATIONYIELD pidx :    22 pvec :              0x0
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@93]  pns 23 pns_null 22
    2020-12-20 20:47:26.775 INFO  [6886110] [X4MaterialPropertiesTable::AddProperties@129]  cpns 33 cpns_null 33
    2020-12-20 20:47:26.775 INFO  [6886110] [X4LogicalSurface::Convert@43] ]




Added test CGDMLPropertyTest the 1042 vs 1062 difference is plain
-------------------------------------------------------------------

The new test loads the GDML and dumps properties using mostly pure Geant4 code.

1042 has some values::

    epsilon:tests blyth$ om-;TEST=CGDMLPropertyTest;om-t 
    === om-mk : bdir /usr/local/opticks/build/cfg4/tests rdir cfg4/tests : make CGDMLPropertyTest && ./CGDMLPropertyTest
    [ 98%] Built target CFG4
    Scanning dependencies of target CGDMLPropertyTest
    [ 98%] Building CXX object tests/CMakeFiles/CGDMLPropertyTest.dir/CGDMLPropertyTest.cc.o
    [100%] Linking CXX executable CGDMLPropertyTest
    [100%] Built target CGDMLPropertyTest
    2020-12-21 12:36:59.293 INFO  [7364433] [main@36] OKConf::Geant4VersionInteger() : 1042
    2020-12-21 12:36:59.293 INFO  [7364433] [main@42]  parsing /tmp/v1.gdml
    G4GDML: Reading '/tmp/v1.gdml'...
    G4GDML: Reading userinfo...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/v1.gdml' done!
    2020-12-21 12:36:59.716 INFO  [7364433] [main@47]  nmat 36
    2020-12-21 12:36:59.716 INFO  [7364433] [main@50]  nlbs 10
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] ESRAirSurfaceTop 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd97d11a0b0 plen       39
    0.98505 0.98505 0.984548 0.983585 0.968716 0.970241 0.971068 0.965398 0.951738 0.981663 0.980112 0.988567 0.985178 0.965753 0.975675 0.977987 0.975159 0.965276 0.966203 0.961376 0.958306 0.957332 0.726586 0.11559 0.104132 0.116531 0.142322 0.118947 0.179949 0.173176 0.09159 0.0100036 0.0099 0.0099 0.0099 0.0099 0.0099 0.0099 0.0099 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd97d11a840 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] ESRAirSurfaceBot 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd97d11db90 plen       39
    0.98505 0.98505 0.984548 0.983585 0.968716 0.970241 0.971068 0.965398 0.951738 0.981663 0.980112 0.988567 0.985178 0.965753 0.975675 0.977987 0.975159 0.965276 0.966203 0.961376 0.958306 0.957332 0.726586 0.11559 0.104132 0.116531 0.142322 0.118947 0.179949 0.173176 0.09159 0.0100036 0.0099 0.0099 0.0099 0.0099 0.0099 0.0099 0.0099 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd97d11de90 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] SSTOilSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd97d128550 plen       39
    0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 0.1 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd97d128d20 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] SSTWaterSurfaceNear1 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd979c1c3a0 plen       39
    0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979c1c6a0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] SSTWaterSurfaceNear2 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd979c1c4d0 plen       39
    0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 0.8 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979c1c560 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] NearIWSCurtainSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd979c300e0 plen       39
    0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.982963 0.987913 0.99 0.995 0.99 0.960524 0.940209 0.91045 0.870413 0.800692 0.760381 0.720299 0.680216 0.640134 0.600051 0.6 0.6 0.6 0.6 0.6 0.6 0.6 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979c303e0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] NearOWSLinerSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd979f853a0 plen       39
    0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.982963 0.987913 0.99 0.995 0.99 0.960524 0.940209 0.91045 0.870413 0.800692 0.760381 0.720299 0.680216 0.640134 0.600051 0.6 0.6 0.6 0.6 0.6 0.6 0.6 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979f856a0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] NearDeadLinerSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7fd979f88930 plen       39
    0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.98 0.982963 0.987913 0.99 0.995 0.99 0.960524 0.940209 0.91045 0.870413 0.800692 0.760381 0.720299 0.680216 0.640134 0.600051 0.6 0.6 0.6 0.6 0.6 0.6 0.6 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979f89100 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:36:59.716 INFO  [7364433] [main@70] SCB_photocathode_logsurf1 23
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979f8ac90 plen       39
    0.0001 0.0001 0.000440306 0.000782349 0.00112439 0.00146644 0.00180848 0.00272834 0.00438339 0.00692303 0.00998793 0.0190265 0.027468 0.0460445 0.0652553 0.0849149 0.104962 0.139298 0.170217 0.19469 0.214631 0.225015 0.24 0.235045 0.21478 0.154862 0.031507 0.00478915 0.00242326 0.000850572 0.000475524 0.000100476 7.50165e-05 5.00012e-05 2.49859e-05 0 0 0 0 
    2020-12-21 12:36:59.717 INFO  [7364433] [main@70] SCB_photocathode_logsurf2 23
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fd979f8ac90 plen       39
    0.0001 0.0001 0.000440306 0.000782349 0.00112439 0.00146644 0.00180848 0.00272834 0.00438339 0.00692303 0.00998793 0.0190265 0.027468 0.0460445 0.0652553 0.0849149 0.104962 0.139298 0.170217 0.19469 0.214631 0.225015 0.24 0.235045 0.21478 0.154862 0.031507 0.00478915 0.00242326 0.000850572 0.000475524 0.000100476 7.50165e-05 5.00012e-05 2.49859e-05 0 0 0 0 
    epsilon:tests blyth$ 
    epsilon:tests blyth$ 
    epsilon:tests blyth$ 



1062 gives all zeros from the same GDML::

    2020-12-21 12:37:29.267 INFO  [7365045] [main@36] OKConf::Geant4VersionInteger() : 1062
    2020-12-21 12:37:29.267 INFO  [7365045] [main@42]  parsing /tmp/v1.gdml
    G4GDML: Reading '/tmp/v1.gdml'...
    G4GDML: Reading userinfo...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/v1.gdml' done!
    2020-12-21 12:37:29.683 INFO  [7365045] [main@47]  nmat 36
    2020-12-21 12:37:29.683 INFO  [7365045] [main@50]  nlbs 10
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] ESRAirSurfaceTop 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] ESRAirSurfaceBot 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] SSTOilSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] SSTWaterSurfaceNear1 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] SSTWaterSurfaceNear2 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.683 INFO  [7365045] [main@70] NearIWSCurtainSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.684 INFO  [7365045] [main@70] NearOWSLinerSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.684 INFO  [7365045] [main@70] NearDeadLinerSurface 23
     i     1 pidx     1 pname                  REFLECTIVITY pvec 0x7ffdca116e80 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.684 INFO  [7365045] [main@70] SCB_photocathode_logsurf1 23
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    2020-12-21 12:37:29.684 INFO  [7365045] [main@70] SCB_photocathode_logsurf2 23
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7ffdca116dc0 plen       39
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
    epsilon:cfg4 charles$ 


See no changes to Geant4 code for property handling with::

   g4n-cls G4MaterialPropertiesTable
   g4n-cls G4PhysicsVector 

So suspicion now pointing to a GDML parsing difference. 

::

    g4n-cd source/persistency/gdml/src

    epsilon:src blyth$ g4n-cls G4GDMLReadStructure



Need a piecemeal way to test. Exploit protected methods with X4GDMLReadStructure ?::

     54 class G4GDMLReadStructure : public G4GDMLReadParamvol
     55 {
     56 
     57  public:
     58 
     59    G4GDMLReadStructure();
     60    virtual ~G4GDMLReadStructure();
     61 
     62    G4VPhysicalVolume* GetPhysvol(const G4String&) const;
     63    G4LogicalVolume* GetVolume(const G4String&) const;
     64    G4AssemblyVolume* GetAssembly(const G4String&) const;
     65    G4GDMLAuxListType GetVolumeAuxiliaryInformation(G4LogicalVolume*) const;
     66    G4VPhysicalVolume* GetWorldVolume(const G4String&);
     67    const G4GDMLAuxMapType* GetAuxMap() const {return &auxMap;}
     68    void Clear();   // Clears internal map and evaluator
     69 
     70    virtual void VolumeRead(const xercesc::DOMElement* const);
     71    virtual void Volume_contentRead(const xercesc::DOMElement* const);
     72    virtual void StructureRead(const xercesc::DOMElement* const);
     73 
     74  protected:
     75 
     76    void AssemblyRead(const xercesc::DOMElement* const);
     77    void DivisionvolRead(const xercesc::DOMElement* const);
     78    G4LogicalVolume* FileRead(const xercesc::DOMElement* const);
     79    void PhysvolRead(const xercesc::DOMElement* const,
     80                     G4AssemblyVolume* assembly=0);
     81    void ReplicavolRead(const xercesc::DOMElement* const, G4int number);
     82    void ReplicaRead(const xercesc::DOMElement* const replicaElement,
     83                     G4LogicalVolume* logvol,G4int number);
     84    EAxis AxisRead(const xercesc::DOMElement* const axisElement);
     85    G4double QuantityRead(const xercesc::DOMElement* const readElement);
     86    void BorderSurfaceRead(const xercesc::DOMElement* const);
     87    void SkinSurfaceRead(const xercesc::DOMElement* const);
     88 
     89  protected:
     90 
     91    G4GDMLAuxMapType auxMap;
     92    G4GDMLAssemblyMapType assemblyMap;
     93    G4LogicalVolume *pMotherLogical;
     94    std::map<std::string, G4VPhysicalVolume*> setuptoPV;
     95    G4bool strip;
     96 };




::


   ..3164     <!-- SCB manual addition start : see notes/issues/sensor-gdml-review.rst -->
     3165     <!-- see bordersurface referencing at tail of structure -->
     3166 
     3167     <opticalsurface finish="0" model="0" name="SCB_photocathode_opsurf" type="0" value="1">
     3168          <property name="EFFICIENCY" ref="EFFICIENCY0x1d79780"/>   <!-- the non-zero efficiency-->
     3169     </opticalsurface>
     3170     <!-- SCB manual addition end : see notes/issues/sensor-gdml-review.rst -->
     3171 
     3172   </solids>


    31971     <!-- SCB manual addition start : see notes/issues/sensor-gdml-review.rst -->
    31972     <!-- see opticalsurface at tail of solids -->
    31973 
    31974     <bordersurface name="SCB_photocathode_logsurf1" surfaceproperty="SCB_photocathode_opsurf">
    31975        <physvolref ref="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0" />
    31976        <physvolref ref="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720" />
    31977     </bordersurface>
    31978 
    31979     <bordersurface name="SCB_photocathode_logsurf2" surfaceproperty="SCB_photocathode_opsurf">
    31980        <physvolref ref="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720" />
    31981        <physvolref ref="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0" />
    31982     </bordersurface>
    31983     <!-- SCB manual addition end : see notes/issues/sensor-gdml-review.rst -->
    31984   </structure>



::

    X4GDMLReadStructureTest /tmp/t.gdml   # writes the GDML string literal in the code to the path provided and parses the file into Geant4 geometry

    CGDMLPropertyTest /tmp/t.gdml         #  


Darwin.charles.1062::

    epsilon:opticks charles$ X4GDMLReadStructureTest /tmp/s.gdml
    2020-12-21 18:59:48.609 INFO  [7924624] [test_readString@149] writing GDMLString to path /tmp/s.gdml
    G4GDML: Reading '/tmp/s.gdml'...
    G4GDML: Reading userinfo...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/s.gdml' done!

    epsilon:opticks charles$ CGDMLPropertyTest /tmp/s.gdml
    2020-12-21 19:00:11.324 INFO  [7928334] [main@36] OKConf::Geant4VersionInteger() : 1062
    2020-12-21 19:00:11.324 INFO  [7928334] [main@43]  parsing /tmp/s.gdml
    G4GDML: Reading '/tmp/s.gdml'...
    G4GDML: Reading userinfo...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/tmp/s.gdml' done!
    2020-12-21 19:00:11.342 INFO  [7928334] [main@48]  nmat 2
    2020-12-21 19:00:11.342 INFO  [7928334] [main@51]  nlbs 1
    2020-12-21 19:00:11.342 INFO  [7928334] [main@71] SCB_photocathode_logsurf1 23
     i     4 pidx     4 pname                    EFFICIENCY pvec 0x7fdc81f048a0 plen       39
    0.0001 0.0001 0.000440306 0.000782349 0.00112439 0.00146644 0.00180848 0.00272834 0.00438339 0.00692303 0.00998793 0.0190265 0.027468 0.0460445 0.0652553 0.0849149 0.104962 0.139298 0.170217 0.19469 0.214631 0.225015 0.24 0.235045 0.21478 0.154862 0.031507 0.00478915 0.00242326 0.000850572 0.000475524 0.000100476 7.50165e-05 5.00012e-05 2.49859e-05 0 0 0 0 
    epsilon:opticks charles$ 


The problem with the full geometry is not manifesting in the partial one.



Rebuild my changed g4_1062
---------------------------

Change ~/.opticks_config setting OPTICKS_GEANT4_VER=1062 and removing the "opticks-prepend-prefix /usr/local/opticks_externals/g4_1042"::

     22 # temporary setting whilst building g4_1062 for use from charles account 
     23 export OPTICKS_GEANT4_VER=1062
     24 
     25 ## hookup paths to access "foreign" externals 
     26 opticks-prepend-prefix /usr/local/opticks_externals/boost
     27 
     28 opticks-prepend-prefix /usr/local/opticks_externals/clhep
     29 opticks-prepend-prefix /usr/local/opticks_externals/xercesc
     30 #opticks-prepend-prefix /usr/local/opticks_externals/g4_1042
     31 

Now in a fresh "blyth" session::

    epsilon:opticks blyth$ g4-
    epsilon:opticks blyth$ g4-prefix
    /usr/local/opticks_externals/g4_1062

::

    g4-build
    ...
    [ 85%] Built target G4interfaces
    [ 86%] Built target G4parmodels
    Scanning dependencies of target G4persistency
    [ 86%] Building CXX object source/persistency/CMakeFiles/G4persistency.dir/gdml/src/G4GDMLRead.cc.o
    [ 86%] Building CXX object source/persistency/CMakeFiles/G4persistency.dir/gdml/src/G4GDMLReadSolids.cc.o
    [ 86%] Linking CXX shared library ../../BuildProducts/lib/libG4persistency.dylib
    [ 88%] Built target G4persistency
    [ 94%] Built target G4physicslists
    [ 95%] Built target G4readout



File a Geant4 bug : 2305
--------------------------

Problem 2305 - All GDML read properties of skinsurface and bordersurface
elements yields only the G4MaterialPropertyVector of the first occurrence of
each property name (edit)

* https://bugzilla-geant4.kek.jp/show_bug.cgi?id=2305




