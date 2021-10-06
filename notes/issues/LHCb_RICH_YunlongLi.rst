LHCb_RICH_YunlongLi
======================




> Hi Simon,
>
> Thank you very much for your reply. 
>
> Since the gdml and log files are too large to deliver as attachments, I
> uploaded the gdml file of the RICH detector and the log files to bitbucket
> ( https://bitbucket.org/yl-li/opticks_lhcb_rich/src/master/), so that you can
> reproduce the problems I mentioned before.
> ...
> I executed the command: 
>
>
>     OKX4Test --deletegeocache \
>              --gdmlpath ~/liyu/geometry/rich1_new.gdml \
>              --cvd 1 --rtx 1 \
>              --envkey --xanalytic \
>              --timemax 400 --animtimemax 400 \
>              --target 1 --eye -1,-1,-1 \
>              --SYSRAP debug
>
>
> and from the OKX4Test_SAbbrev.log (as attached to this email) in the last few
> lines, you can see [PipeBeTV56] and [PipeTitaniumG5] give the same abbrievation
> as [P5]. And after changing the code a bit as I wrote before, this problem can
> be solved.


Regarding your commandline notice that controlling the logging level 
at the project level with "--SYSRAP debug" tends to yield huge amounts of output.
Instead of doing that you can control the logging level for each class/struct 
by setting envvars named after the class/struct.  For example::

    export SAbbrev=INFO

Also note that the option "--cvd 1" is internally setting CUDA_VISIBLE_DEVICES envvar 
to 1 which will only work if you have more than one GPU attached.  
In this case with OKX4Test that does not matter, as the GPU is not being used, 
but in other cases using an inappropriate "--cvd" will cause crashes.  

Regarding the SAbbrev assert I added SAbbrevTest.cc test_3 for the issue::

void test_3()
{
     SAbbrev::FromString(R"LITERAL(
Copper
PipeAl6061
C4F10
PipeAl2219F
VeloStainlessSteel
Vacuum
PipeBeTV56
PipeSteel316LN
PipeBe
Celazole
PipeTitaniumG5
AW7075
PipeAl6082
FutureFibre
Technora
Brass
PipeSteel
BakeOutAerogel
Rich2CarbonFibre
RichSoftIron
Rich1GasWindowQuartz
Kovar
HpdIndium
HpdWindowQuartz
HpdS20PhCathode
HpdChromium
HpdKapton
Supra36Hpd
RichHpdSilicon
RichHpdVacuum
Rich1Nitrogen
Rich1MirrorCarbonFibre
R1RadiatorGas
Rich1MirrorGlassSimex
Rich1Mirror2SupportMaterial
Rich1G10
Rich1PMI
Rich1DiaphramMaterial
Air
)LITERAL")->dump() ; 

} 

I fixed the issue by using random abbreviations when the usual attempts 
to abbreviate fail to come up with something unique.  See the commit:

https://bitbucket.org/simoncblyth/opticks/commits/574be3f0366be3f0c94a6a9edd1a43d2039e2d1c




> 2. In this file only the polished, polished front-painted and ground mirrors
> are considered, other cases will cause the assertion in line 239 failed. Are
> you planning to handle other types of mirrors?
>
>
>   I have no plan to implement more surface types until I need them.
>
>   I am very willing to incorporate your pull requests with more surface types added.
>   However I suggest you discuss with me how you plan to do that first to ensure your
>   work can be incorporated into Opticks.
>
> The main reason why I asked this problem is that in this gdml file, there are some ground frontpainted mirrors (type 4), which can cause the command
>
> OKX4Test --deletegeocache --gdmlpath ~/liyu/geometry/rich1_new.gdml --cvd 1 --rtx 1 --envkey --xanalytic --timemax 400 --animtimemax 400 --target 1 --eye -1,-1,-1
>
> failed (See OKX4Test_GOpticalSurface.log). Right now, in order to make the test running, we just added
> if(strncmp(m_finish,"4",strlen(m_finish))==0)  return false ;
> to GOpticalSurface::isSpecular() function.

In this case is does not matter as you provided the GDML, but in general you will need to provide stack traces 
of problems using the gdb debugger and the "bt" command.  The below is from lldb on macOS however its simular with gdb on Linux::

    ...
    2021-10-06 16:14:03.770 INFO  [17614518] [X4PhysicalVolume::convertMaterials@322]  used_materials.size 39 num_material_with_efficiency 0
    2021-10-06 16:14:03.770 INFO  [17614518] [GMaterialLib::dumpSensitiveMaterials@1257] X4PhysicalVolume::convertMaterials num_sensitive_materials 0
    2021-10-06 16:14:03.770 NONE  [17614518] [*X4::MakeSurfaceIndexCache@330] [  num_lbs 1984 num_sks 0
    2021-10-06 16:14:03.771 NONE  [17614518] [*X4::MakeSurfaceIndexCache@350] ]
    2021-10-06 16:14:03.773 INFO  [17614518] [GOpticalSurface::isSpecular@234] GOpticalSurface::isSpecular  m_shortname RichHPDEnvLargeTubeMetalSurface0000x110f3550 m_finish 4
    Assertion failed: (0 && "expecting m_finish to be 0:polished or 3:ground "), function isSpecular, file /Users/blyth/opticks/ggeo/GOpticalSurface.cc, line 239.
    Process 38346 stopped

    Process 38346 launched: '/usr/local/opticks/lib/OKX4Test' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #3: 0x00007fff7101f1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001097fb6fc libGGeo.dylib`GOpticalSurface::isSpecular(this=0x000000010ed45f30) const at GOpticalSurface.cc:239
        frame #5: 0x00000001098a5d97 libGGeo.dylib`GSurfaceLib::createStandardSurface(this=0x000000011440a630, src=0x000000010ed464c0) at GSurfaceLib.cc:524
        frame #6: 0x00000001098a4ec2 libGGeo.dylib`GSurfaceLib::addStandardized(this=0x000000011440a630, surf=0x000000010ed464c0) at GSurfaceLib.cc:441
        frame #7: 0x00000001098a4e04 libGGeo.dylib`GSurfaceLib::addBorderSurface(this=0x000000011440a630, surf=0x000000010ed464c0, pv1="_dd_Geometry_BeforeMagnetRegion_Rich1_RichHPDMasterLogList_lvRich1HPDMaster000_pvRich1HPDSMaster0000x1120b9d0", pv2="_dd_Geometry_BeforeMagnetRegion_Rich1_RichHPDSMasterLogList_lvRich1HPDSMaster000_pvRichHPDEnvLargeTub0xd090ff0", direct=false) at GSurfaceLib.cc:373
        frame #8: 0x00000001098a4ac7 libGGeo.dylib`GSurfaceLib::add(this=0x000000011440a630, raw=0x000000010ed464c0, implicit=false, direct=false) at GSurfaceLib.cc:346
        frame #9: 0x00000001037aef86 libExtG4.dylib`X4LogicalBorderSurfaceTable::init(this=<unavailable>) at X4LogicalBorderSurfaceTable.cc:128 [opt]
        frame #10: 0x00000001037aecc9 libExtG4.dylib`X4LogicalBorderSurfaceTable::Convert(GSurfaceLib*, char) [inlined] X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=<unavailable>, dst=<unavailable>, mode=<unavailable>) at X4LogicalBorderSurfaceTable.cc:107 [opt]
        frame #11: 0x00000001037aecaf libExtG4.dylib`X4LogicalBorderSurfaceTable::Convert(GSurfaceLib*, char) [inlined] X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=<unavailable>, dst=<unavailable>, mode=<unavailable>) at X4LogicalBorderSurfaceTable.cc:106 [opt]
        frame #12: 0x00000001037aecaf libExtG4.dylib`X4LogicalBorderSurfaceTable::Convert(dst=<unavailable>, mode='\x10') at X4LogicalBorderSurfaceTable.cc:43 [opt]
        frame #13: 0x00000001037c3e42 libExtG4.dylib`X4PhysicalVolume::convertSurfaces(this=<unavailable>) at X4PhysicalVolume.cc:662 [opt]
        frame #14: 0x00000001037c3445 libExtG4.dylib`X4PhysicalVolume::init(this=<unavailable>) at X4PhysicalVolume.cc:201 [opt]
        frame #15: 0x00000001037c2fc0 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=<unavailable>, ggeo=<unavailable>, top=<unavailable>) at X4PhysicalVolume.cc:182 [opt]
        frame #16: 0x0000000100015736 OKX4Test`main(argc=12, argv=0x00007ffeefbfcec8) at OKX4Test.cc:108
    (lldb) 

    (lldb) f 5
    frame #5: 0x00000001098a5d97 libGGeo.dylib`GSurfaceLib::createStandardSurface(this=0x000000011440a630, src=0x000000010ed464c0) at GSurfaceLib.cc:524
       521 	            }
       522 	            assert(_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity " );
       523 	
    -> 524 	            if(os->isSpecular())
       525 	            {
       526 	                _detect  = makeConstantProperty(0.0) ;    
       527 	                _reflect_specular = _REFLECTIVITY ;
    (lldb) 

    (lldb) f 4
    frame #4: 0x00000001097fb6fc libGGeo.dylib`GOpticalSurface::isSpecular(this=0x000000010ed45f30) const at GOpticalSurface.cc:239
       236 	              << " m_finish "    << ( m_finish ? m_finish : "-" ) 
       237 	              ;
       238 	   
    -> 239 	    assert(0 && "expecting m_finish to be 0:polished or 3:ground ");
       240 	    return false ; 
       241 	}
       242 	
    (lldb) 


The assert is avoided with::

    288 /**
    289 GOpticalSurface::isSpecular
    290 ---------------------------
    291 
    292 Now returns true for all three polished finishes : polished, polishedfrontpainted, polishedbackpainted
    293 Opticks treats all these three finishes as a specular surface. 
    294 
    295 **/
    296 bool GOpticalSurface::isSpecular() const { return isPolished() ; }
    297 


see: https://bitbucket.org/simoncblyth/opticks/commits/ae7f3607c1ee774a24d78811fe68a8f3abb5b1ce




> https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/extg4/X4Solid.cc#lines-1105,
>
> 3. In this file why are the startphi and deltaphi not allowed to be 0 and 360
> at the same time? I see in G4Polycone class, such case is allowed.
>
>
>   1091 void X4Solid::convertPolycone()
>   1092 {
>   1093     // G4GDMLWriteSolids::PolyconeWrite
>   1094     // G4GDMLWriteSolids::ZplaneWrite
>   1095     // ../analytic/gdml.py
>   1096
>   1097     //LOG(error) << "START" ;
>   1098
>   1099     const G4Polycone* const solid = static_cast<const G4Polycone*>(m_solid);
>   1100     assert(solid);
>   1101     const G4PolyconeHistorical* ph = solid->GetOriginalParameters() ;
>   1102
>   1103     float startphi = ph->Start_angle/degree ;
>   1104     float deltaphi = ph->Opening_angle/degree ;
>   1105     assert( startphi == 0.f && deltaphi == 360.f );
>   1106
>
>
>
>   The assertion on line 1105 is requiring that startphi=0 and deltaphi=360 constraining that
>   there is no phi segment applied to the polycone.
>
>   The assert is there just because that has not been needed in the geometries so far faced.
>   You are very welcome to do the development work of adding that in a pull request. Make
>   sure to include a unit test that tests the new functionality you are adding.
>
>
>
> This case exists in this gdml file. if you correct all the things above and run the command:
> OKX4Test --deletegeocache --gdmlpath ~/liyu/geometry/rich1_new.gdml --cvd 1 --rtx 1 --envkey --xanalytic --timemax 400 --animtimemax 400 --target 1 --eye -1,-1,-1 --X4 debug
> the assertion here will fail (see OKX4Test_X4Solid.log file).
>

> At present, we just remove this assertion 


There is no easy fix to extend the solid implementation to handle phi segmented polycones::

    2021-10-06 17:47:59.178 INFO  [17852095] [GPropertyLib::dumpSensorIndices@1066] X4PhysicalVolume::convertSurfaces  NumSensorIndices 1 ( 1990  ) 
    Assertion failed: (startphi == 0.f && deltaphi == 360.f), function convertPolycone, file /Users/blyth/opticks/extg4/X4Solid.cc, line 1105.

    Process 72914 launched: '/usr/local/opticks/lib/OKX4Test' (x86_64)
    (lldb) bt
        frame #3: 0x00007fff7101f1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010379dc8f libExtG4.dylib`X4Solid::convertPolycone(this=<unavailable>) at X4Solid.cc:1105 [opt]
        frame #5: 0x000000010379ae87 libExtG4.dylib`X4Solid::init(this=<unavailable>) at X4Solid.cc:170 [opt]
        frame #6: 0x000000010379a92b libExtG4.dylib`X4Solid::Convert(G4VSolid const*, Opticks*, char const*) [inlined] X4Solid::X4Solid(this=<unavailable>, solid=<unavailable>, ok=<unavailable>, top=<unavailable>) at X4Solid.cc:132 [opt]
        frame #7: 0x000000010379a905 libExtG4.dylib`X4Solid::Convert(G4VSolid const*, Opticks*, char const*) [inlined] X4Solid::X4Solid(this=<unavailable>, solid=<unavailable>, ok=<unavailable>, top=<unavailable>) at X4Solid.cc:131 [opt]
        frame #8: 0x000000010379a905 libExtG4.dylib`X4Solid::Convert(solid=<unavailable>, ok=<unavailable>, boundary=<unavailable>) at X4Solid.cc:95 [opt]
        frame #9: 0x00000001037c813e libExtG4.dylib`X4PhysicalVolume::convertSolid(this=<unavailable>, lvIdx=<unavailable>, soIdx=<unavailable>, solid=<unavailable>, lvname=<unavailable>, balance_deep_tree=<unavailable>) const at X4PhysicalVolume.cc:1087 [opt]
        frame #10: 0x00000001037c6e7e libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=<unavailable>, pv=<unavailable>, depth=<unavailable>) at X4PhysicalVolume.cc:980 [opt]
        frame #11: 0x00000001037c6bf6 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=<unavailable>, pv=<unavailable>, depth=<unavailable>) at X4PhysicalVolume.cc:964 [opt]
        frame #12: 0x00000001037c4151 libExtG4.dylib`X4PhysicalVolume::convertSolids(this=<unavailable>) at X4PhysicalVolume.cc:926 [opt]
        frame #13: 0x00000001037c3466 libExtG4.dylib`X4PhysicalVolume::init(this=<unavailable>) at X4PhysicalVolume.cc:203 [opt]
        frame #14: 0x00000001037c2fc0 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=<unavailable>, ggeo=<unavailable>, top=<unavailable>) at X4PhysicalVolume.cc:182 [opt]
        frame #15: 0x0000000100015736 OKX4Test`main(argc=12, argv=0x00007ffeefbfcec8) at OKX4Test.cc:108
        frame #16: 0x00007fff70fab015 libdyld.dylib`start + 1
        frame #17: 0x00007fff70fab015 libdyld.dylib`start + 1
    (lldb) 
    (lldb) f 4
    libExtG4.dylib was compiled with optimization - stepping may behave oddly; variables may not be available.
    frame #4: 0x000000010379dc8f libExtG4.dylib`X4Solid::convertPolycone(this=<unavailable>) at X4Solid.cc:1105 [opt]
       1102	
       1103	    float startphi = ph->Start_angle/degree ;  
       1104	    float deltaphi = ph->Opening_angle/degree ;
    -> 1105	    assert( startphi == 0.f && deltaphi == 360.f ); 
       1106	
       1107	    unsigned nz = ph->Num_z_planes ; 
       1108	
    (lldb) 


> and I am willing to find a better solution here.
>
> https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/extg4/X4PhysicalVolume.cc#lines-1398,


The place to implement this is in X4Solid::convertPolycone
using X4Solid::intersectWithPhiSegment as other shapes do already.
The phi segment shape is a prism described by a set of planes
to form the convex polyhedron. 

Although using X4Solid::intersectWithPhiSegment can be done very easily
with only a few lines of code following the example of other shapes
that use intersectWithPhiSegment the performance and correctness 
of such segmented shapes has not been well tested.  

So most if the work would be in validation and comparison with Geant4. 
Also the performance would need to be measured as the segment that 
is intersected with is implemented using a CSG convexpolyhedron   
implemented with a set of planes. 

If performance or correctness is poor the next thing I would try 
is to intersect with a segment formed from some other shape
that does not use the plane defined convex polyhedron.   

Whether it is worthwhile for you to do this implementation depends on 
how optically important the shape is within your geometry. 


Some complicated solid (lvIdx 74) is hanging the conversion.  Using "export X4Solid=INFO"::

    2021-10-06 20:26:38.199 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.199 INFO  [18037644] [*X4Solid::Convert@116] ]
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::Banner@80]  lvIdx    74 soIdx    74 soname UX85-2-CollarAttMainSub0xdca8d50 lvname _dd_Geometry_MagnetRegion_PipeSupportsInMagnet_lvUX852CollarAtt0xdca8f80
    2021-10-06 20:26:38.206 INFO  [18037644] [*X4Solid::Convert@104] [ convert UX85-2-CollarAttMainSub0xdca8d50
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier a entityType                    3 entityName   G4SubtractionSolid name         UX85-2-CollarAttMainSub0xdca8d50 root 0x0
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::convertBooleanSolid@300]  _operator 3 CSG::Name difference
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier b entityType                    3 entityName   G4SubtractionSolid name UX85-2-CollarAttMain-Child_For_UX85-2-CollarAttMainSub0xdca89e0 root 0x0
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::convertBooleanSolid@300]  _operator 3 CSG::Name difference
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier c entityType                    3 entityName   G4SubtractionSolid name UX85-2-CollarAttMain-Child_For_UX85-2-CollarAttMainSub0xdca8670 root 0x0
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::convertBooleanSolid@300]  _operator 3 CSG::Name difference
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier d entityType                    3 entityName   G4SubtractionSolid name UX85-2-CollarAttMain-Child_For_UX85-2-CollarAttMainSub0xdca8300 root 0x0
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::convertBooleanSolid@300]  _operator 3 CSG::Name difference
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier e entityType                    3 entityName   G4SubtractionSolid name UX85-2-CollarAttMain-Child_For_UX85-2-CollarAttMainSub0xdca7f90 root 0x0
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::convertBooleanSolid@300]  _operator 3 CSG::Name difference
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier f entityType                    3 entityName   G4SubtractionSolid name UX85-2-CollarAttMain-Child_For_UX85-2-CollarAttMainSub0xdca7c20 root 0x0
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::convertBooleanSolid@300]  _operator 3 CSG::Name difference
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier g entityType                    3 entityName   G4SubtractionSolid name UX85-2-CollarAttMain-Child_For_UX85-2-CollarAttMainSub0xdca7940 root 0x0
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::convertBooleanSolid@300]  _operator 3 CSG::Name difference
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier h entityType                    3 entityName   G4SubtractionSolid name UX85-2-CollarAttMain-Child_For_UX85-2-CollarAttMainSub0xdca7660 root 0x0
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::convertBooleanSolid@300]  _operator 3 CSG::Name difference
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier i entityType                    5 entityName                G4Box name            UX85-2-CollarAttMain0xdca2120 root 0x0
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier j entityType                    0 entityName     G4DisplacedSolid name                                  placedB root 0x0
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier k entityType                    5 entityName                G4Box name        UX85-2-CollarAttMainSub10xdca7510 root 0x0
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.206 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier l entityType                    0 entityName     G4DisplacedSolid name                                  placedB root 0x0
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier m entityType                    5 entityName                G4Box name        UX85-2-CollarAttMainSub20xdca7890 root 0x0
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier n entityType                    0 entityName     G4DisplacedSolid name                                  placedB root 0x0
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier o entityType                    5 entityName                G4Box name        UX85-2-CollarAttMainSub30xdca7b70 root 0x0
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier p entityType                    0 entityName     G4DisplacedSolid name                                  placedB root 0x0
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier q entityType                   25 entityName               G4Tubs name      UX85-2-CollarAttMain-Hole10xdca7e50 root 0x0
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier r entityType                    0 entityName     G4DisplacedSolid name                                  placedB root 0x0
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier s entityType                   25 entityName               G4Tubs name      UX85-2-CollarAttMain-Hole20xdca81c0 root 0x0
    2021-10-06 20:26:38.207 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier t entityType                    0 entityName     G4DisplacedSolid name                                  placedB root 0x0
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier u entityType                   25 entityName               G4Tubs name UX85-2-CollarAttMain-RoundEdge10xdca8530 root 0x0
    2021-10-06 20:26:38.208 ERROR [18037644] [*X4Solid::intersectWithPhiSegment@736]  special cased startPhi == 0.f && deltaPhi == 180.f 
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier v entityType                    0 entityName     G4DisplacedSolid name                                  placedB root 0x0
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier w entityType                   25 entityName               G4Tubs name UX85-2-CollarAttMain-RoundEdge20xdca88a0 root 0x0
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier x entityType                    0 entityName     G4DisplacedSolid name                                  placedB root 0x0
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@163] [ X4SolidBase identifier y entityType                   25 entityName               G4Tubs name UX85-2-CollarAttMain-RoundEdge2b0xdca8c10 root 0x0
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.208 INFO  [18037644] [X4Solid::init@199] ]
    2021-10-06 20:26:38.209 INFO  [18037644] [*X4Solid::Convert@116] ]
    2021-10-06 20:26:38.209 FATAL [18037644] [*NTreeBalance<nnode>::create_balanced@101] balancing trees of this structure not implemented
    ^C^C^C^C^CProcess 84649 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGKILL
        frame #0: 0x000000010a208b67 libNPY.dylib`glm::vec<4, float, (glm::qualifier)0>& glm::vec<4, float, (this=0x00007ffeefbf96a0, v=0x00007ffeefbf9750)0>::operator*=<float>(glm::vec<4, float, (glm::qualifier)0> const&) at type_vec4.inl:597
       594 		template<typename U>
       595 		GLM_FUNC_QUALIFIER GLM_CONSTEXPR vec<4, T, Q> & vec<4, T, Q>::operator*=(vec<4, U, Q> const& v)
       596 		{
    -> 597 			return (*this = detail::compute_vec4_mul<T, Q, detail::is_aligned<Q>::value>::call(*this, vec<4, T, Q>(v)));
       598 		}
       599 	
       600 		template<typename T, qualifier Q>
    Target 0: (OKX4Test) stopped.

    Process 84649 launched: '/usr/local/opticks/lib/OKX4Test' (x86_64)
    (lldb) 
    error: No auto repeat.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGKILL
      * frame #0: 0x000000010a208b67 libNPY.dylib`glm::vec<4, float, (glm::qualifier)0>& glm::vec<4, float, (this=0x00007ffeefbf96a0, v=0x00007ffeefbf9750)0>::operator*=<float>(glm::vec<4, float, (glm::qualifier)0> const&) at type_vec4.inl:597
        frame #1: 0x000000010a208a90 libNPY.dylib`glm::vec<4, float, (glm::qualifier)0> glm::operator*<float, (v1=0x0000000117576d44, v2=0x00007ffeefbf9750)0>(glm::vec<4, float, (glm::qualifier)0> const&, glm::vec<4, float, (glm::qualifier)0> const&) at type_vec4.inl:890
        frame #2: 0x000000010a0c51f0 libNPY.dylib`glm::mat<4, 4, float, (glm::qualifier)0>::col_type glm::operator*<float, (m=0x0000000117576d14, v=0x00007ffeefbf9858)0>(glm::mat<4, 4, float, (glm::qualifier)0> const&, glm::mat<4, 4, float, (glm::qualifier)0>::row_type const&) at type_mat4x4.inl:569
        frame #3: 0x000000010a345b80 libNPY.dylib`ncylinder::operator(this=0x00000001175767b0, x_=10.1814461, y_=-11.9151802, z_=3.02999997)(float, float, float) const at NCylinder.cpp:74
        frame #4: 0x000000010a2e1f63 libNPY.dylib`ndifference::operator(this=0x0000000117575f90, x=10.1814461, y=-11.9151802, z=3.02999997)(float, float, float) const at NNode.cpp:1152
        frame #5: 0x000000010a2e1e3c libNPY.dylib`nintersection::operator(this=0x0000000117576b90, x=10.1814461, y=-11.9151802, z=3.02999997)(float, float, float) const at NNode.cpp:1144
        frame #6: 0x000000010a2e1e63 libNPY.dylib`nintersection::operator(this=0x000000011941cde8, x=10.1814461, y=-11.9151802, z=3.02999997)(float, float, float) const at NNode.cpp:1145
        frame #7: 0x000000010a2f47e2 libNPY.dylib`float std::__1::__invoke_void_return_wrapper<float>::__call<nintersection&, float, float, float>(nintersection&&&, float&&, float&&, float&&) [inlined] decltype(__f=0x000000011941cde8, __args=0x00007ffeefbf9a6c, __args=0x00007ffeefbf9a68, __args=0x00007ffeefbf9a64)(std::__1::forward<float, float, float>(fp0))) std::__1::__invoke<nintersection&, float, float, float>(nintersection&&&, float&&, float&&, float&&) at type_traits:4291
        frame #8: 0x000000010a2f479b libNPY.dylib`float std::__1::__invoke_void_return_wrapper<float>::__call<nintersection&, float, float, float>(__args=0x000000011941cde8, __args=0x00007ffeefbf9a6c, __args=0x00007ffeefbf9a68, __args=0x00007ffeefbf9a64) at __functional_base:328
        frame #9: 0x000000010a2f4599 libNPY.dylib`std::__1::__function::__func<nintersection, std::__1::allocator<nintersection>, float (float, float, float)>::operator(this=0x000000011941cde0, __arg=0x00007ffeefbf9a6c, __arg=0x00007ffeefbf9a68, __arg=0x00007ffeefbf9a64)(float&&, float&&, float&&) at functional:1552
        frame #10: 0x000000010a32058f libNPY.dylib`std::__1::function<float (float, float, float)>::operator(this=0x000000011941cde0, __arg=10.1814461, __arg=-11.9151802, __arg=3.02999997)(float, float, float) const at functional:1903
        frame #11: 0x000000010a31e84b libNPY.dylib`NNodePoints::selectBySDF(this=0x000000011757b2d0, prim=0x000000011941be10, prim_idx=10, pointmask=2) at NNodePoints.cpp:271
        frame #12: 0x000000010a31db25 libNPY.dylib`NNodePoints::collectCompositePoints(this=0x000000011757b2d0, level=9, margin=0, pointmask=2) at NNodePoints.cpp:207
        frame #13: 0x000000010a31cac0 libNPY.dylib`NNodePoints::collect_surface_points(this=0x000000011757b2d0) at NNodePoints.cpp:155
        frame #14: 0x000000010a3b519f libNPY.dylib`NCSG::collect_surface_points(this=0x0000000117577b20) at NCSG.cpp:1170
        frame #15: 0x000000010a3b40ce libNPY.dylib`NCSG::postchange(this=0x0000000117577b20) at NCSG.cpp:204
        frame #16: 0x000000010a3b457b libNPY.dylib`NCSG::Adopt(root=0x00000001175770d0, config=0x0000000000000000, soIdx=74, lvIdx=74) at NCSG.cpp:173
        frame #17: 0x00000001037c80b4 libExtG4.dylib`X4PhysicalVolume::convertSolid(this=<unavailable>, lvIdx=<unavailable>, soIdx=<unavailable>, solid=<unavailable>, lvname=<unavailable>, balance_deep_tree=<unavailable>) const at X4PhysicalVolume.cc:1094 [opt]
        frame #18: 0x00000001037c6d7e libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=<unavailable>, pv=<unavailable>, depth=<unavailable>) at X4PhysicalVolume.cc:980 [opt]
        frame #19: 0x00000001037c6af6 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=<unavailable>, pv=<unavailable>, depth=<unavailable>) at X4PhysicalVolume.cc:964 [opt]
        frame #20: 0x00000001037c4051 libExtG4.dylib`X4PhysicalVolume::convertSolids(this=<unavailable>) at X4PhysicalVolume.cc:926 [opt]
        frame #21: 0x00000001037c3366 libExtG4.dylib`X4PhysicalVolume::init(this=<unavailable>) at X4PhysicalVolume.cc:203 [opt]
        frame #22: 0x00000001037c2ec0 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=<unavailable>, ggeo=<unavailable>, top=<unavailable>) at X4PhysicalVolume.cc:182 [opt]
        frame #23: 0x0000000100015736 OKX4Test`main(argc=12, argv=0x00007ffeefbfce38) at OKX4Test.cc:108
        frame #24: 0x00007fff70fab015 libdyld.dylib`start + 1
        frame #25: 0x00007fff70fab015 libdyld.dylib`start + 1
    (lldb) 



    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGKILL
      * frame #0: 0x000000010a209aa9 libNPY.dylib`glm::vec<4, float, (glm::qualifier)0> glm::operator*<float, (v1=0x0000000118094a24, v2=0x00007ffeefbf9690)0>(glm::vec<4, float, (glm::qualifier)0> const&, glm::vec<4, float, (glm::qualifier)0> const&) at type_vec4.inl:890
        frame #1: 0x000000010a0c6106 libNPY.dylib`glm::mat<4, 4, float, (glm::qualifier)0>::col_type glm::operator*<float, (m=0x0000000118094a24, v=0x00007ffeefbf97b8)0>(glm::mat<4, 4, float, (glm::qualifier)0> const&, glm::mat<4, 4, float, (glm::qualifier)0>::row_type const&) at type_mat4x4.inl:563
        frame #2: 0x000000010a3422a9 libNPY.dylib`nbox::sdf_(this=0x000000011808f550, pos=0x00007ffeefbf9800, triple=0x00000001180949e0)0> const&, nmat4triple const*) const at NBox.cpp:85
        frame #3: 0x000000010a342242 libNPY.dylib`nbox::operator(this=0x000000011808f550, x_=-122.745415, y_=43.5319595, z_=-3.38671875)(float, float, float) const at NBox.cpp:52
        frame #4: 0x000000010a2e2f3c libNPY.dylib`ndifference::operator(this=0x000000011808feb0, x=-122.745415, y=43.5319595, z=-3.38671875)(float, float, float) const at NNode.cpp:1151
        frame #5: 0x000000010a2e2f3c libNPY.dylib`ndifference::operator(this=0x0000000118090520, x=-122.745415, y=43.5319595, z=-3.38671875)(float, float, float) const at NNode.cpp:1151
        frame #6: 0x000000010a2e2f3c libNPY.dylib`ndifference::operator(this=0x0000000118091430, x=-122.745415, y=43.5319595, z=-3.38671875)(float, float, float) const at NNode.cpp:1151
        frame #7: 0x000000010a2e2f3c libNPY.dylib`ndifference::operator(this=0x0000000118091eb0, x=-122.745415, y=43.5319595, z=-3.38671875)(float, float, float) const at NNode.cpp:1151
        frame #8: 0x000000010a2e2f3c libNPY.dylib`ndifference::operator(this=0x0000000118092f10, x=-122.745415, y=43.5319595, z=-3.38671875)(float, float, float) const at NNode.cpp:1151
        frame #9: 0x000000010a2e2e3c libNPY.dylib`nintersection::operator(this=0x0000000119c03008, x=-122.745415, y=43.5319595, z=-3.38671875)(float, float, float) const at NNode.cpp:1144
        frame #10: 0x000000010a2f57e2 libNPY.dylib`float std::__1::__invoke_void_return_wrapper<float>::__call<nintersection&, float, float, float>(nintersection&&&, float&&, float&&, float&&) [inlined] decltype(__f=0x0000000119c03008, __args=0x00007ffeefbf9acc, __args=0x00007ffeefbf9ac8, __args=0x00007ffeefbf9ac4)(std::__1::forward<float, float, float>(fp0))) std::__1::__invoke<nintersection&, float, float, float>(nintersection&&&, float&&, float&&, float&&) at type_traits:4291
        frame #11: 0x000000010a2f579b libNPY.dylib`float std::__1::__invoke_void_return_wrapper<float>::__call<nintersection&, float, float, float>(__args=0x0000000119c03008, __args=0x00007ffeefbf9acc, __args=0x00007ffeefbf9ac8, __args=0x00007ffeefbf9ac4) at __functional_base:328
        frame #12: 0x000000010a2f5599 libNPY.dylib`std::__1::__function::__func<nintersection, std::__1::allocator<nintersection>, float (float, float, float)>::operator(this=0x0000000119c03000, __arg=0x00007ffeefbf9acc, __arg=0x00007ffeefbf9ac8, __arg=0x00007ffeefbf9ac4)(float&&, float&&, float&&) at functional:1552
        frame #13: 0x000000010a32158f libNPY.dylib`std::__1::function<float (float, float, float)>::operator(this=0x0000000119c03000, __arg=-122.745415, __arg=43.5319595, __arg=-3.38671875)(float, float, float) const at functional:1903
        frame #14: 0x000000010a31f84b libNPY.dylib`NNodePoints::selectBySDF(this=0x0000000118097b40, prim=0x000000011808fab0, prim_idx=1, pointmask=2) at NNodePoints.cpp:271
        frame #15: 0x000000010a31eb25 libNPY.dylib`NNodePoints::collectCompositePoints(this=0x0000000118097b40, level=9, margin=0, pointmask=2) at NNodePoints.cpp:207
        frame #16: 0x000000010a31dac0 libNPY.dylib`NNodePoints::collect_surface_points(this=0x0000000118097b40) at NNodePoints.cpp:155
        frame #17: 0x000000010a3b619f libNPY.dylib`NCSG::collect_surface_points(this=0x00000001180943a0) at NCSG.cpp:1170
        frame #18: 0x000000010a3b50ce libNPY.dylib`NCSG::postchange(this=0x00000001180943a0) at NCSG.cpp:204
        frame #19: 0x000000010a3b557b libNPY.dylib`NCSG::Adopt(root=0x0000000118094190, config=0x0000000000000000, soIdx=94, lvIdx=94) at NCSG.cpp:173
        frame #20: 0x00000001037c808a libExtG4.dylib`X4PhysicalVolume::convertSolid(this=<unavailable>, lvIdx=<unavailable>, soIdx=<unavailable>, solid=<unavailable>, lvname=<unavailable>, balance_deep_tree=<unavailable>) const at X4PhysicalVolume.cc:1098 [opt]
        frame #21: 0x00000001037c6c4e libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=<unavailable>, pv=<unavailable>, depth=<unavailable>) at X4PhysicalVolume.cc:980 [opt]
        frame #22: 0x00000001037c69c6 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=<unavailable>, pv=<unavailable>, depth=<unavailable>) at X4PhysicalVolume.cc:964 [opt]
        frame #23: 0x00000001037c3f21 libExtG4.dylib`X4PhysicalVolume::convertSolids(this=<unavailable>) at X4PhysicalVolume.cc:926 [opt]
        frame #24: 0x00000001037c3236 libExtG4.dylib`X4PhysicalVolume::init(this=<unavailable>) at X4PhysicalVolume.cc:203 [opt]
        frame #25: 0x00000001037c2d90 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=<unavailable>, ggeo=<unavailable>, top=<unavailable>) at X4PhysicalVolume.cc:182 [opt]
        frame #26: 0x0000000100015736 OKX4Test`main(argc=6, argv=0x00007ffeefbfce90) at OKX4Test.cc:108
        frame #27: 0x00007fff70fab015 libdyld.dylib`start + 1
    (lldb) 




    2021-10-06 20:59:24.040 INFO  [18087330] [X4Solid::Banner@80]  lvIdx   867 soIdx   867 soname Rich1MasterWithSubtract0xcf514d0 lvname _dd_Geometry_BeforeMagnetRegion_Rich1_lvRich1Master0xcf516c0
    2021-10-06 20:59:24.040 INFO  [18087330] [*X4Solid::Convert@104] [ convert Rich1MasterWithSubtract0xcf514d0
    2021-10-06 20:59:24.040 INFO  [18087330] [X4Solid::init@163] [ X4SolidBase identifier a entityType                    3 entityName   G4SubtractionSolid name         Rich1MasterWithSubtract0xcf514d0 root 0x0
    ...
    2021-10-06 20:59:24.041 INFO  [18087330] [*X4Solid::Convert@116] ]
    2021-10-06 20:59:24.041 FATAL [18087330] [nnode::get_primitive_bbox@1060] Need to add upcasting for type: 0 name zero
    Assertion failed: (0), function get_primitive_bbox, file /Users/blyth/opticks/npy/NNode.cpp, line 1061.

    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff710fbb66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff712c6080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff710571ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff7101f1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010a2e285b libNPY.dylib`nnode::get_primitive_bbox(this=0x00000001350c7d20, bb=0x00007ffeefbfa340) const at NNode.cpp:1061
        frame #5: 0x000000010a2e2bb3 libNPY.dylib`nnode::bbox(this=0x00000001350c7d20) const at NNode.cpp:1099
        frame #6: 0x000000010a32ab6d libNPY.dylib`NNodeNudger::update_prim_bb(this=0x00000001350c80b0) at NNodeNudger.cpp:105
        frame #7: 0x000000010a32a208 libNPY.dylib`NNodeNudger::init(this=0x00000001350c80b0) at NNodeNudger.cpp:82
        frame #8: 0x000000010a329d36 libNPY.dylib`NNodeNudger::NNodeNudger(this=0x00000001350c80b0, root_=0x00000001350c7320, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:67
        frame #9: 0x000000010a32a57d libNPY.dylib`NNodeNudger::NNodeNudger(this=0x00000001350c80b0, root_=0x00000001350c7320, epsilon_=0.00000999999974, (null)=0) at NNodeNudger.cpp:65
        frame #10: 0x000000010a3b6bcf libNPY.dylib`NCSG::make_nudger(this=0x00000001350c8000, msg="Adopt root ctor") const at NCSG.cpp:1412
        frame #11: 0x000000010a3b64e7 libNPY.dylib`NCSG::NCSG(this=0x00000001350c8000, root=0x00000001350c7320) at NCSG.cpp:282
        frame #12: 0x000000010a3b55fd libNPY.dylib`NCSG::NCSG(this=0x00000001350c8000, root=0x00000001350c7320) at NCSG.cpp:297
        frame #13: 0x000000010a3b551f libNPY.dylib`NCSG::Adopt(root=0x00000001350c7320, config=0x0000000000000000, soIdx=867, lvIdx=867) at NCSG.cpp:166
        frame #14: 0x00000001037c808a libExtG4.dylib`X4PhysicalVolume::convertSolid(this=<unavailable>, lvIdx=<unavailable>, soIdx=<unavailable>, solid=<unavailable>, lvname=<unavailable>, balance_deep_tree=<unavailable>) const at X4PhysicalVolume.cc:1098 [opt]
        frame #15: 0x00000001037c6c4e libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=<unavailable>, pv=<unavailable>, depth=<unavailable>) at X4PhysicalVolume.cc:980 [opt]
        frame #16: 0x00000001037c69c6 libExtG4.dylib`X4PhysicalVolume::convertSolids_r(this=<unavailable>, pv=<unavailable>, depth=<unavailable>) at X4PhysicalVolume.cc:964 [opt]
        frame #17: 0x00000001037c3f21 libExtG4.dylib`X4PhysicalVolume::convertSolids(this=<unavailable>) at X4PhysicalVolume.cc:926 [opt]
        frame #18: 0x00000001037c3236 libExtG4.dylib`X4PhysicalVolume::init(this=<unavailable>) at X4PhysicalVolume.cc:203 [opt]
        frame #19: 0x00000001037c2d90 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=<unavailable>, ggeo=<unavailable>, top=<unavailable>) at X4PhysicalVolume.cc:182 [opt]
        frame #20: 0x0000000100015736 OKX4Test`main(argc=6, argv=0x00007ffeefbfce88) at OKX4Test.cc:108
        frame #21: 0x00007fff70fab015 libdyld.dylib`start + 1
        frame #22: 0x00007fff70fab015 libdyld.dylib`start + 1
    (lldb) 



2021-10-06 20:54:46.195 INFO  [18081422] [*X4Solid::Convert@116] ]
2021-10-06 20:54:46.198 INFO  [18081422] [X4Solid::Banner@80]  lvIdx    90 soIdx    90 soname UX85-3-BigRingQuarter-Sub0xdcefce0 lvname _dd_Geometry_MagnetRegion_PipeSupportsInMagnet_lvUX853BigRingQuarter0xdceff10
2021-10-06 20:54:46.198 INFO  [18081422] [*X4Solid::Convert@104] [ convert UX85-3-BigRingQuarter-Sub0xdcefce0
2021-10-06 20:54:46.198 INFO  [18081422] [X4Solid::init@163] [ X4SolidBase identifier a entityType                



Add "--x4balanceskip" option

>
> 4. In this file the names of the inner material and outer material are
> extracted and then used in line 1524, 1530, 1536 for GBndLib->addBoundary
> function.  In extg4/X4PhysicalVolume.cc, omat and imat are directly extracted
> from logical volumes, and may follow this style "_dd_Materials_Air",
> "_dd_Materials_Vacuum" But in GBndLib::add function, omat and imat are
> extracted from GMaterialLib according to their indexes, and follow this style
> "Air", "Vacuum".  Such difference can cause an assertion failed.
>
>
>   The geometries I work with currently do not have prefixes such as "/dd/Material/"
>   on material names, so there could well be a missing X4::BaseName or equivalent somewhere ?
>   However the way you reported the issue makes me unsure of what the issue is !
>
> Sorry if my description confuses you. You can refer to OKX4Test_GBndLIb.log file, which are generated by this command
> OKX4Test --deletegeocache --gdmlpath ~/liyu/geometry/rich1_new.gdml --cvd 1 --rtx 1 --envkey --xanalytic --timemax 400 --animtimemax 400 --target 1 --eye -1,-1,-1 --X4 debug.
> In line 126191, you can see the names of omat and imat with prefixed as "_dd_Materials".
> 
> Let's see if you can reproduce these problems and then we can deal with others.
>
> Thank you very much for your help and patience.
>
> Best wishes,
> 
> Yunlong
>
>













Hi Yunlong, 

> I hope all is well with you. 

Thanks, I'm well. I hope all is well with you too. 

> From our recent studies about Opticks using LHCb RICH detector and other
> simplied geometries, we found some issues and would like to seek for your help.
> Sorry I don't put these issues on groups.io, because they are related to
> different topics.
>
> https://bitbucket.org/simoncblyth/opticks/src/48b41f66c8b0c821e9458e36568d9daf4350bf29/sysrap/SAbbrev.cc#lines-44, 
> 
> 1. In this file it gives the abbreviations of material names which are used by
> GPropertyLib.  But if names are, i.e, "PipeSteel" and “PipeStainlessSteel”,
> which give the same abbreviations, the assertion in line 106 will fail.


See my update to the test sysrap/tests/SAbbrevTest.cc:test_2, that shows that different abbreviations 
are obtained and there is no assert.::

    sysrap/tests/SAbbrevTest.cc:test_2

    111 void test_2()
    112 {
    113     LOG(info);
    114     std::vector<std::string> ss = {
    115         "PipeSteel",
    116         "PipeStainlessSteel"
    117     };
    118     SAbbrev ab(ss);
    119     ab.dump();
    120 }

Running that test::

    SAbbrevTest 

    2021-09-30 19:56:16.207 INFO  [12432035] [test_2@113] 
                         PipeSteel : PS
                PipeStainlessSteel : Pl


I guess your set of material names has a problem but your idea of what the problem is, 
is not correct. 

The best way to investigate and report issues is to add a test to the unit test 
for the relevant class that captures the issue that you are seeing.

Runnable code provides a much more precise, effective and faster way to communicate issues than words. 
Also it is the best way to investigate issues.
 
When I can see the actual problem you are facing via a failing test, 
I can then consider how to fix it.

> But why do we need to use the abbreviations instead of full names?


The OpenGL GUI and also analysis python provides material history sequence tables 
with the material at every step of the photon presented. 
For those tables to be readable a 2 character abbreviation is needed. 

The abbreviation code could definitely be improved to avoid asserts, 
provide me with the set of names in a test that asserts and I will do so.
For example by doing something like you suggest below or even by forming 
random two character abbreviations until a unique one is found.

> A possible way is to change lines 73~86 to::
>
>       if( n->upper == 1 && n->number > 0 ) // 1 or more upper and number
>       {
>           int iu = n->first_upper_index ;
>           int in = n->first_number_index ;
>           ab = n->getTwoChar( iu < in ? iu : in ,  iu < in ? in : iu  );
>       }
>       else if( n->upper >= 2 ) // more than one uppercase : form abbrev from first two uppercase chars
>       {
>           ab = n->getFirstUpper(n->upper) ;
>       }
>       else
>       {
>           ab = n->getFirst(2) ;
>       }




> https://bitbucket.org/simoncblyth/opticks/src/7ebbd54d88ded3b5b713b3133c653012656dc582/ggeo/GOpticalSurface.cc#lines-228, 
> 
> 2. In this file only the polished, polished front-painted and ground mirrors
> are considered, other cases will cause the assertion in line 239 failed. Are
> you planning to handle other types of mirrors?
>

I have no plan to implement more surface types until I need them. 

I am very willing to incorporate your pull requests with more surface types added.  
However I suggest you discuss with me how you plan to do that first to ensure your 
work can be incorporated into Opticks.

However note that Opticks will soon undergo an enormous transition for compatibility 
with the all new NVIDIA OptiX 7 API. 
This transition  means that all GPU code must be re-architected. It is far from 
being a simple transition, the OptiX 7 API is totally different to OptiX 6.5 
As a result the below packages will be removed::

   cudarap
   thrustrap
   optixrap
   okop

With the below packages added::

   QUDARap  : pure CUDA photon generation, no OptiX dependency 
   CSG      : shared CPU/GPU geometry model 
   CSG_GGeo : conversion of GGeo geometry model into CSG 
   CSGOptiX : OptiX 7 ray tracing 
  
A focus for the new architecture is to provide fine-grained modular testing of GPU code. 

Given the tectonic shifts that Opticks will soon undergo, I think it makes
more sense to do things like implement more surface types after the 
dust has settled in the new architecture. 



> https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/extg4/X4Solid.cc#lines-1105, 
>
> 3. In this file why are the startphi and deltaphi not allowed to be 0 and 360
> at the same time? I see in G4Polycone class, such case is allowed.  


    1091 void X4Solid::convertPolycone()
    1092 {
    1093     // G4GDMLWriteSolids::PolyconeWrite
    1094     // G4GDMLWriteSolids::ZplaneWrite
    1095     // ../analytic/gdml.py 
    1096 
    1097     //LOG(error) << "START" ; 
    1098 
    1099     const G4Polycone* const solid = static_cast<const G4Polycone*>(m_solid);
    1100     assert(solid);
    1101     const G4PolyconeHistorical* ph = solid->GetOriginalParameters() ;
    1102 
    1103     float startphi = ph->Start_angle/degree ;
    1104     float deltaphi = ph->Opening_angle/degree ;
    1105     assert( startphi == 0.f && deltaphi == 360.f );
    1106 


The assertion on line 1105 is requiring that startphi=0 and deltaphi=360 constraining that 
there is no phi segment applied to the polycone.

The assert is there just because that has not been needed in the geometries so far faced.  
You are very welcome to do the development work of adding that in a pull request. Make 
sure to include a unit test that tests the new functionality you are adding. 

Again after you have thought about how you want to implement this and done
some preliminary development make sure to discuss your approach with me to 
ensure that your work can be incorporated into Opticks.
I think I have implemented similar things somewhere via CSG intersection with a phi 
segment shape.

The sample problem with the impending shift in Opticks applies however. There is 
little point in doing any developments in the packages that do not have long to live.



> https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/extg4/X4PhysicalVolume.cc#lines-1398, 

>
> 4. In this file the names of the inner material and outer material are
> extracted and then used in line 1524, 1530, 1536 for GBndLib->addBoundary
> function.  In extg4/X4PhysicalVolume.cc, omat and imat are directly extracted
> from logical volumes, and may follow this style "_dd_Materials_Air",
> "_dd_Materials_Vacuum" But in GBndLib::add function, omat and imat are
> extracted from GMaterialLib according to their indexes, and follow this style
> "Air", "Vacuum".  Such difference can cause an assertion failed. 


The geometries I work with currently do not have prefixes such as "/dd/Material/"
on material names : so your problem suggests there is a missing X4::BaseName somewhere ? 
Tell me where and I will add it. 

1384 unsigned X4PhysicalVolume::addBoundary(const G4VPhysicalVolume* const pv, const G4VPhysicalVolume* const pv_p )
1385 {
1386     const G4LogicalVolume* const lv   = pv->GetLogicalVolume() ;
1387     const G4LogicalVolume* const lv_p = pv_p ? pv_p->GetLogicalVolume() : NULL ;
1388 
1389     // GDMLName adds pointer suffix to the object name, returns null when object is null : eg parent of world 
1390 
1391     const char* _pv = X4::GDMLName(pv) ;
1392     const char* _pv_p = X4::GDMLName(pv_p) ;
1393 
1394 
1395     const G4Material* const imat_ = lv->GetMaterial() ;
1396     const G4Material* const omat_ = lv_p ? lv_p->GetMaterial() : imat_ ;  // top omat -> imat 
1397 
1398     const char* omat = X4::BaseName(omat_) ;
1399     const char* imat = X4::BaseName(imat_) ;
1400 
....
1513     unsigned boundary = 0 ;
1514     if( g_sslv == NULL && g_sslv_p == NULL  )   // no skin surface on this or parent volume, just use bordersurface if there are any
1515     {
1516 
1517 #ifdef OLD_ADD_BOUNDARY
1518         const char* osur = X4::BaseName( osur_ );
1519         const char* isur = X4::BaseName( isur_ );
1520 #else
1521         const char* osur = osur_ ? osur_->getName() : nullptr ;
1522         const char* isur = isur_ ? isur_->getName() : nullptr ;
1523 #endif
1524         boundary = m_blib->addBoundary( omat, osur, isur, imat );
1525     }
1526     else if( g_sslv && !g_sslv_p )   // skin surface on this volume but not parent : set both osur and isur to this 
1527     {
1528         const char* osur = g_sslv->getName();
1529         const char* isur = osur ;
1530         boundary = m_blib->addBoundary( omat, osur, isur, imat );
1531     }
1532     else if( g_sslv_p && !g_sslv )  // skin surface on parent volume but not this : set both osur and isur to this
1533     {
1534         const char* osur = g_sslv_p->getName();
1535         const char* isur = osur ;
1536         boundary = m_blib->addBoundary( omat, osur, isur, imat );
1537     }
1538     else if( g_sslv_p && g_sslv )
1539     {
1540         assert( 0 && "fabled double skin found : see notes/issues/ab-blib.rst  " );
1541     }
1542 
1543     return boundary ;
1544 }

>
>
> A possible way is to deal with omat and imat in the same way as GPropertyMap::FindShortName, change lines 1398~1399 in extg4/X4PhysicalVolume.cc to::
>
>       const char* omat_name = X4::BaseName(omat_);
>       const char* imat_name = X4::BaseName(imat_);
>       const char* omat = NULL;
>       const char* imat = NULL;
>       if( omat_name[0] == '_')
>       {
>           const char* p = strrchr(omat_name, '_') ; 
>           omat = strdup(p+1) ;
>       }
>       else
>       {
>           omat = strdup(omat_name);
>       }
>       if( imat_name[0] == '_')
>       {
>           const char* p = strrchr(imat_name, '_') ; 
>           imat = strdup(p+1) ;
>       }
>       else
>       {
>            imat = strdup(imat_name);
>       }


This way is special casing prefixed names. 

It would be simpler to regularize the names by stripping the prefixes first, 
which is easier to understand and better because it takes less code. 

>
> The same issue exist in 
>
> * https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/extg4/X4MaterialLib.cc#lines-135,

Whats the issue here ? m4_name_base is the name with prefix removed 

::

    129     for(unsigned i=0 ; i < num_materials ; i++)
    130     {
    131         GMaterial*  pmap = m_mlib->getMaterial(i);
    132         G4Material* m4 = (*m_mtab)[i] ;
    133         assert( pmap && m4 );
    134 
    135         const char* pmap_name = pmap->getName();
    136         const std::string& m4_name = m4->GetName();
    137 
    138         bool has_prefix = strncmp( m4_name.c_str(), DD_MATERIALS_PREFIX, strlen(DD_MATERIALS_PREFIX) ) == 0 ;
    139         const char* m4_name_base = has_prefix ? m4_name.c_str() + strlen(DD_MATERIALS_PREFIX) : m4_name.c_str() ;
    140         bool name_match = strcmp( m4_name_base, pmap_name) == 0 ;
    141 
    142         LOG(info)
    143              << std::setw(5) << i
    144              << " ok pmap_name " << std::setw(30) << pmap_name
    145              << " g4 m4_name  " << std::setw(30) << m4_name
    146              << " g4 m4_name_base  " << std::setw(30) << m4_name_base
    147              << " has_prefix " << has_prefix
    148              ;




> * https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/cfg4/CGDMLDetector.cc#lines-206
> * https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/cfg4/CGDMLDetector.cc#lines-206.

Line 206 strips the prefix from the G4Material name if there is one and the lookup 
for the GMaterial is using that unprefixed shortname. What is the issue ?

::

    201     for(unsigned int i=0 ; i < nmat_without_mpt ; i++)
    202     {
    203         G4Material* g4mat = m_traverser->getMaterialWithoutMPT(i) ;
    204         const char* name = g4mat->GetName() ;
    205 
    206         const std::string base = BFile::Name(name);
    207         const char* shortname = base.c_str();
    208 
    209         const GMaterial* ggmat = m_mlib->getMaterial(shortname);
    210         assert(ggmat && strcmp(ggmat->getShortName(), shortname)==0 && "failed to find corresponding G4DAE material") ;
    211 
    212         LOG(verbose)
    213             << " g4mat " << std::setw(45) << name
    214             << " shortname " << std::setw(25) << shortname
    215             ;
    216 

    421 std::string BFile::Name(const char* path)
    422 {
    423     fs::path fsp(path);
    424     std::string name = fsp.filename().string() ;
    425     return name ;
    426 }



Using X4::BaseName on the original material name should get rid of the prefix, see X4Test::

    epsilon:extg4 blyth$ X4Test 
    2021-09-30 20:31:06.725 INFO  [12460728] [test_Name@31] 
     name      : /dd/material/Water
     Name      : /dd/material/Water
     ShortName : /dd/material/Water
     BaseName  : Water

 75 template<typename T>
 76 const char* X4::BaseName( const T* const obj )
 77 {
 78     if(obj == NULL) return NULL ;
 79     const std::string& name = obj->GetName();
 80     return BaseName(name);
 81 }


 40 const char* X4::ShortName( const std::string& name )
 41 {
 42     char* shortname = BStr::trimPointerSuffixPrefix(name.c_str(), NULL) ;
 43     return strdup( shortname );
 44 }
 45 
 46 const char* X4::Name( const std::string& name )
 47 {
 48     return strdup( name.c_str() );
 49 }
 50 
 51 const char* X4::BaseName( const std::string& name)
 52 {
 53     const std::string base = BFile::Name(name.c_str());
 54     return ShortName(base) ;
 55 }


>
>
> https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/ggeo/GMeshLib.cc#lines-193, 
>
> 5. In this file mesh->getAlt can be NULL because it's allowed in line 159, but
> it can cause the following assertion failed. A possible way is to add one line
> after line 193::
>
>       if( mesh->getAlt()==NULL ) continue ; // To be consistent with GMeshLib::saveAltReferences() 
>
> These are some problems we found until now. 


Thank you for working with Opticks.

Life is too short to worry about "theoretical" problems with code, 
there are more than enough real problems.  

So if you have a real issues please report them in a way that I can reproduce them.

Making changes based on code "reading" and possibly incomplete ideas 
of what is happening (or what might happen) is an unwise way to 
direct development efforts. 

I prefer a more traditional approach:

1. you exercise the code and find issues
2. you share the issues in a way that enables me to reproduce them
3. I (or you) try to fix them, preferably by writing simple tests that exercises the code 

For simple issues you could add a unit test that captures the problem, if more complex
you can share some GDML (preferably simplified) that tickles the issue.


> And we are glad to share you some
> pictures of the visualizations of LHCb RICH I geometry and the simplified
> geometry, as attached to this email.

Thank you for sharing the images. Those are very useful to include in presentations 
to enable me to demonstrate all the experiements that are evaluating Opticks
and encourage more adoption.

If you create any more detector geometry and photon path images or movies 
created with Opticks please remember to share them with me.  

>
> Thank you very much for building such an excellent software and look forward to your comments.
>

You are very welcome. 

Simon


