LHCb_RICH_YunlongLi
======================

Hi Yunlong,

> Since the gdml and log files are too large to deliver as attachments, I
> uploaded the gdml file of the RICH detector and the log files to bitbucket
> ( https://bitbucket.org/yl-li/opticks_lhcb_rich/src/master/), so that you can
> reproduce the problems I mentioned before.

Thank you, that had enabled me to reproduce the issues you describe. 

> ...
> I executed the command: 
>
>     OKX4Test --deletegeocache \
>              --gdmlpath ~/liyu/geometry/rich1_new.gdml \
>              --cvd 1 --rtx 1 \
>              --envkey --xanalytic \
>              --timemax 400 --animtimemax 400 \
>              --target 1 --eye -1,-1,-1 \
>              --SYSRAP debug
>
> and from the OKX4Test_SAbbrev.log (as attached to this email) in the last few
> lines, you can see [PipeBeTV56] and [PipeTitaniumG5] give the same abbrievation
> as [P5]. And after changing the code a bit as I wrote before, this problem can
> be solved.


Regarding your commandline notice that controlling the logging level at the project level 
with "--SYSRAP debug" tends to yield huge amounts of output.
Instead of doing that you can control the logging LEVEL for each class/struct 
by setting envvars named after the class/struct.  For example::

    export SAbbrev=INFO

Also note that the option "--cvd 1" is internally setting CUDA_VISIBLE_DEVICES envvar 
to 1 which will only work if you have more than one GPU attached.  
In this case with OKX4Test that does not matter, as CUDA is not being used for the translation.
However in other cases using an inappropriate "--cvd" will cause crashes.  

Regarding the SAbbrev assert I added SAbbrevTest.cc test_3 to look into the issue
with your material names::

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

I fixed the issue by using random abbreviations when the usual approaches
to abbreviate fail to come up with something unique.  See the commit:

* https://bitbucket.org/simoncblyth/opticks/commits/574be3f0366be3f0c94a6a9edd1a43d2039e2d1c



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
of problems using the gdb debugger and the "bt" command.  The below is from lldb on macOS however it
behaves in a simular way to gdb on Linux::

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


The assert is avoided via a change to::

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
> This case exists in this gdml file. if you correct all the things above and run the command:
> OKX4Test --deletegeocache --gdmlpath ~/liyu/geometry/rich1_new.gdml --cvd 1 --rtx 1 --envkey --xanalytic --timemax 400 --animtimemax 400 --target 1 --eye -1,-1,-1 --X4 debug
> the assertion here will fail (see OKX4Test_X4Solid.log file).
>

> At present, we just remove this assertion 
> and I am willing to find a better solution here.
>
> https://bitbucket.org/simoncblyth/opticks/src/02b098569330585dc6303275b1c84a1855a7e1f9/extg4/X4PhysicalVolume.cc#lines-1398,


The place to implement this is in X4Solid::convertPolycone
using X4Solid::intersectWithPhiSegment as other shapes do already.
The phi segment shape is a prism described by a set of planes
to form the convex polyhedron. 

My recent commits implement this  but it is currently disabled as
it needs debugging, and as your geometry seems to have numerous 
other problems with the translation. 

Note that the performance and correctness of shapes using 
intersectWithPhiSegment for such phi segmented shapes has not been well tested.  

So if it is essential for you, then you will need work on 
validation and comparison with Geant4. 
Also the performance would need to be measured as the segment that 
is intersected with is implemented using a CSG convexpolyhedron   
implemented with a set of planes. 

If performance/correctness is poor the next thing I would try 
is to intersect with a segment formed from some other shape
that does not use the plane defined convex polyhedron.   

Whether it is worthwhile for you to do this implementation depends on 
how optically important the shape is within your geometry. 

Regarding the numerous other problems, I have added several --x4*skip 
options to skip parts of the conversion in order to try and assess 
how many of your solid are having problems.

The below script uses these options to skip problems with some solids, 
that are identified by lvIdx (logical volume indices, which match the soIdx solid indices)::

    #!/bin/bash -l 

    # more verbose logging LEVEL for these classes/structs
    export GBndLib=INFO
    export X4PhysicalVolume=INFO
    export X4Solid=INFO
    export NCSG=INFO

    PFX=""
    case $(uname) in
       Darwin) PFX=lldb__ ;;
    esac

    $PFX \
        OKX4Test \
            --deletegeocache \
            --gdmlpath \
                $PWD/rich1_new.gdml \
            --x4balanceskip 74,90,94 \
            --x4nudgeskip 857,867 \
            --x4pointskip 74,867


There are severe problems with the conversion of around 5 of 869 solids. 
Examples of the backtraces and logging from problem solids are in notes/issues/LHCb_RICH_YunlongLi_backtraces.rst

After getting through solid conversion the next issue I found was::

    2021-10-07 12:29:00.213 INFO  [18602665] [GGeo::prepareVolumes@1301] ]
    2021-10-07 12:29:00.966 INFO  [18602665] [GGeo::prepare@678] ]
    Assertion failed: (imat && omat), function fillMaterialLineMap, file /Users/blyth/opticks/ggeo/GBndLib.cc, line 823.
    Process 85577 stopped

    Process 85577 launched: '/usr/local/opticks/lib/OKX4Test' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff710fbb66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff712c6080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff710571ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff7101f1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x00000001098c7c6b libGGeo.dylib`GBndLib::fillMaterialLineMap(this=0x000000010eb3fcd0, msu=size=0) at GBndLib.cc:823
        frame #5: 0x00000001098c82f1 libGGeo.dylib`GBndLib::fillMaterialLineMap(this=0x000000010eb3fcd0) at GBndLib.cc:842
        frame #6: 0x000000010995e08f libGGeo.dylib`GGeo::postDirectTranslation(this=0x000000010eb3fb00) at GGeo.cc:586
        frame #7: 0x000000010001575a OKX4Test`main(argc=10, argv=0x00007ffeefbfce30) at OKX4Test.cc:113
        frame #8: 0x00007fff70fab015 libdyld.dylib`start + 1

    (lldb) f 4
    frame #4: 0x00000001098c7c6b libGGeo.dylib`GBndLib::fillMaterialLineMap(this=0x000000010eb3fcd0, msu=size=0) at GBndLib.cc:823
       820 	        const guint4& bnd = m_bnd[i] ;
       821 	        const char* omat = m_mlib->getName(bnd[OMAT]);
       822 	        const char* imat = m_mlib->getName(bnd[IMAT]);
    -> 823 	        assert(imat && omat);
       824 	        if(msu.count(imat) == 0) msu[imat] = getLine(i, IMAT) ;
       825 	        if(msu.count(omat) == 0) msu[omat] = getLine(i, OMAT) ; 
       826 	    }

    (lldb) p bnd
    (const guint4) $0 = (x = 4294967295, y = 4294967295, z = 4294967295, w = 4294967295)
    (lldb) p i
    (unsigned int) $1 = 0
    (lldb) 
    (lldb) p getNumBnd()
    (unsigned int) $2 = 7
    (lldb) 
    (lldb) p m_bnd
    (std::__1::vector<guint4, std::__1::allocator<guint4> >) $3 = size=7 {
      [0] = (x = 4294967295, y = 4294967295, z = 4294967295, w = 4294967295)
      [1] = (x = 4294967295, y = 4294967295, z = 1984, w = 4294967295)
      [2] = (x = 4294967295, y = 4294967295, z = 1985, w = 4294967295)
      [3] = (x = 4294967295, y = 4294967295, z = 1986, w = 4294967295)
      [4] = (x = 4294967295, y = 4294967295, z = 1987, w = 4294967295)
      [5] = (x = 4294967295, y = 4294967295, z = 1988, w = 4294967295)
      [6] = (x = 4294967295, y = 4294967295, z = 1989, w = 4294967295)
    }


The boundaries are stored via sets of 4 ints, (omat,osur,isur,imat) 
so the above shows that only isur is ever being set. 


> 4. In this file the names of the inner material and outer material are
> extracted and then used in line 1524, 1530, 1536 for GBndLib->addBoundary
> function.  In extg4/X4PhysicalVolume.cc, omat and imat are directly extracted
> from logical volumes, and may follow this style "_dd_Materials_Air",
> "_dd_Materials_Vacuum" But in GBndLib::add function, omat and imat are
> extracted from GMaterialLib according to their indexes, and follow this style
> "Air", "Vacuum".  Such difference can cause an assertion failed.
>
>   The geometries I work with currently do not have prefixes such as "/dd/Material/"
>   on material names, so there could well be a missing X4::BaseName or equivalent somewhere ?
>   However the way you reported the issue makes me unsure of what the issue is !
>
> Sorry if my description confuses you. You can refer to OKX4Test_GBndLIb.log file, which are generated by this command
> OKX4Test --deletegeocache --gdmlpath ~/liyu/geometry/rich1_new.gdml --cvd 1 --rtx 1 --envkey --xanalytic --timemax 400 --animtimemax 400 --target 1 --eye -1,-1,-1 --X4 debug.
> In line 126191, you can see the names of omat and imat with prefixed as "_dd_Materials".
>

Trying to fix an issue from a description alone is akin to trying to type with both hands 
tied behind your back. Reproducing an issue is almost always the essential 
first step to being able to fix it.

My recent commits fix the inconsistent handling of G4Material name prefix 
handling between ggeo/GPropertyMap and extg4/X4 by using sysrap/SGDML 
to provide the common G4Material name prefix. 

> Let's see if you can reproduce these problems and then we can deal with others.
> Thank you very much for your help and patience.

Updating to the lastest Opticks should avoid most of the issues. However parts
of the translation are being skipped for some solids and there are many 
error and warning logs from other solids.

    #!/bin/bash -l 

    # logging for material prefixes and boundary issue
    export GBndLib=INFO
    export X4PhysicalVolume=INFO
    export X4MaterialTable=INFO
    export X4Material=INFO

    # logging for problem solids
    #export X4Solid=INFO
    #export NCSG=INFO

    PFX=""
    case $(uname) in
       Darwin) PFX=lldb__ ;;
    esac

    $PFX \
        OKX4Test \
            --deletegeocache \
            --gdmlpath \
                $PWD/rich1_new.gdml \
            --x4balanceskip 74,90,94 \
            --x4nudgeskip 857,867 \
            --x4pointskip 74,867


Also I note that setting the OPTICKS_KEY envvar as reported 
by the OKX4Test and running other executables, such as OTracerTest, 
to load that geometry is currently asserting::

    epsilon:~ blyth$ lldb__ OTracerTest 
    /Applications/Xcode/Xcode_10_1.app/Contents/Developer/usr/bin/lldb -f OTracerTest -o r --
    (lldb) target create "/usr/local/opticks/lib/OTracerTest"
    Current executable set to '/usr/local/opticks/lib/OTracerTest' (x86_64).
    (lldb) r
    2021-10-07 19:52:35.724 INFO  [19810746] [OpticksHub::loadGeometry@283] [ /usr/local/opticks/geocache/OKX4Test_World0x11431010_PV_g4live/g4ok_gltf/788769803760b2e287e492ade2bc5a3c/1
    Assertion failed: (unsigned(altindex) < m_meshes.size()), function loadAltReferences, file /Users/blyth/opticks/ggeo/GMeshLib.cc, line 198.
    Process 51176 stopped

    Process 51176 launched: '/usr/local/opticks/lib/OTracerTest' (x86_64)
    (lldb) bt
        frame #3: 0x00007fff7101f1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000100a64777 libGGeo.dylib`GMeshLib::loadAltReferences(this=0x000000010a7655e0) at GMeshLib.cc:198
        frame #5: 0x0000000100a63315 libGGeo.dylib`GMeshLib::loadFromCache(this=0x000000010a7655e0) at GMeshLib.cc:79
        frame #6: 0x0000000100a63258 libGGeo.dylib`GMeshLib::Load(ok=0x000000010834b550) at GMeshLib.cc:67
        frame #7: 0x0000000100a42256 libGGeo.dylib`GGeo::loadFromCache(this=0x0000000108201380) at GGeo.cc:543
        frame #8: 0x0000000100a45094 libGGeo.dylib`GGeo::loadGeometry(this=0x0000000108201380) at GGeo.cc:510
        frame #9: 0x0000000100830c9a libOpticksGeo.dylib`OpticksHub::loadGeometry(this=0x0000000108369350) at OpticksHub.cc:287
        frame #10: 0x000000010082fd29 libOpticksGeo.dylib`OpticksHub::init(this=0x0000000108369350) at OpticksHub.cc:250
        frame #11: 0x000000010082fb1c libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x0000000108369350, ok=0x000000010834b550) at OpticksHub.cc:217
        frame #12: 0x000000010082ff5d libOpticksGeo.dylib`OpticksHub::OpticksHub(this=0x0000000108369350, ok=0x000000010834b550) at OpticksHub.cc:216
        frame #13: 0x00000001000d1034 libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfcc98, argc=1, argv=0x00007ffeefbfcd50, argforced="--tracer") at OKMgr.cc:57
        frame #14: 0x00000001000d162b libOK.dylib`OKMgr::OKMgr(this=0x00007ffeefbfcc98, argc=1, argv=0x00007ffeefbfcd50, argforced="--tracer") at OKMgr.cc:65
        frame #15: 0x0000000100009a0a OTracerTest`main(argc=1, argv=0x00007ffeefbfcd50) at OTracerTest.cc:38
        frame #16: 0x00007fff70fab015 libdyld.dylib`start + 1
    (lldb) 


That is the first thing to investigate when I can spare some more cycles 
to look into issues with your geometry. 

Simon













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


