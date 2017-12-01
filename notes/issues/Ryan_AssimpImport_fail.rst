Ryan_AssimpImport_fail
========================

Failed geocache creation
----------------------------

::

    Jui-Jens-MacBook-Pro:opticks wangbtc$ op.sh -G
    296 -rwxr-xr-x  1 wangbtc  staff  147496 Nov 28 10:37 /Users/wangbtc/local/opticks/lib/OKTest
    proceeding.. : /Users/wangbtc/local/opticks/lib/OKTest -G
    2017-11-28 13:03:34.911 WARN  [205648] [OpticksResource::readG4Environment@493] OpticksResource::readG4Environment MISSING FILE externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2017-11-28 13:03:34.914 INFO  [205648] [Opticks::dumpArgs@958] Opticks::configure argc 2
      0 : /Users/wangbtc/local/opticks/lib/OKTest
      1 : -G
    2017-11-28 13:03:34.914 INFO  [205648] [OpticksHub::configure@236] OpticksHub::configure m_gltf 0
    2017-11-28 13:03:34.916 INFO  [205648] [OpticksHub::loadGeometry@366] OpticksHub::loadGeometry START
    2017-11-28 13:03:34.930 INFO  [205648] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    2017-11-28 13:03:34.933 INFO  [205648] [OpticksGeometry::loadGeometry@102] OpticksGeometry::loadGeometry START 
    2017-11-28 13:03:34.933 INFO  [205648] [OpticksGeometry::loadGeometryBase@134] OpticksGeometry::loadGeometryBase START 
    2017-11-28 13:03:34.933 INFO  [205648] [GGeo::loadGeometry@522] GGeo::loadGeometry START loaded 0 gltf 0
    2017-11-28 13:03:34.934 INFO  [205648] [AssimpGGeo::load@135] AssimpGGeo::load  path /Users/wangbtc/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae query range:3153:12221 ctrl  verbosity 0
    2017-11-28 13:03:34.946 INFO  [205648] [AssimpImporter::import@195] AssimpImporter::import path /Users/wangbtc/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae flags 32779
    myStream Error, T0: Collada: File came out empty. Something is wrong here.
    AssimpImporter::import ERROR : "Collada: File came out empty. Something is wrong here." 
    2017-11-28 13:03:35.119 INFO  [205648] [AssimpGGeo::load@150] AssimpGGeo::load select START 
    AssimpImporter::select no tree 
    2017-11-28 13:03:35.119 INFO  [205648] [AssimpGGeo::load@154] AssimpGGeo::load select DONE 
    2017-11-28 13:03:35.326 INFO  [205648] [*OpticksResource::getSensorList@1138] OpticksResource::getSensorList NSensorList:  NSensor count 6888 distinct identier count 684
    2017-11-28 13:03:35.326 INFO  [205648] [AssimpGGeo::convert@172] AssimpGGeo::convert ctrl 
    /Users/wangbtc/opticks/bin/op.sh: line 787:  8610 Segmentation fault: 11  /Users/wangbtc/local/opticks/lib/OKTest -G
    /Users/wangbtc/opticks/bin/op.sh RC 139
    Jui-Jens-MacBook-Pro:opticks wangbtc$ echo $IDPATH


AssimpRapTest with verbosity pumped up
----------------------------------------

Ryans AssimpRapTest appears to be using a different assimp ?

::

    Jui-Jens-MacBook-Pro:opticks wangbtc$ AssimpRapTest --importverbosity 3  —-loadverbosity 3 
    2017-11-30 12:49:19.564 INFO  [515433] [main@71] ok
    2017-11-30 12:49:19.564 INFO  [515433] [Opticks::dumpArgs@980] Opticks::configure argc 5
      0 : AssimpRapTest
      1 : --importverbosity
      2 : 3
      3 : —-loadverbosity
      4 : 3
    2017-11-30 12:49:19.566 INFO  [515433] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    after gg
    2017-11-30 12:49:19.567 ERROR [515433] [GGeo::loadFromG4DAE@560] GGeo::loadFromG4DAE START
    2017-11-30 12:49:19.568 INFO  [515433] [AssimpGGeo::load@137] AssimpGGeo::load  path /Users/wangbtc/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae query range:3153:12221 ctrl  importVerbosity 3 loaderVerbosity 0
    AssimpImporter::init verbosity 3 severity.Err Err severity.Warn Warn severity.Info Info severity.Debugging Debugging
    myStream Debug, T0: debug
    myStream Info,  T0: info
    myStream Warn,  T0: warn
    myStream Error, T0: error
    2017-11-30 12:49:19.568 INFO  [515433] [AssimpImporter::import@216] AssimpImporter::import path /Users/wangbtc/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae flags 32779
    myStream Info,  T0: Load /Users/wangbtc/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    myStream Debug, T0: Assimp 4.0.0 amd64 gcc shared singlethreaded
    myStream Info,  T0: Found a matching importer for this file format: Collada Importer.
    myStream Info,  T0: Import root directory is '/Users/wangbtc/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/'
    myStream Debug, T0: Collada schema version is 1.4.n
    myStream Debug, T0: Ignoring global element <opticalsurface>.
    myStream Skipping one or more lines with the same contents
    myStream Debug, T0: Ignoring global element <skinsurface>.
    myStream Skipping one or more lines with the same contents
    myStream Debug, T0: Ignoring global element <bordersurface>.
    myStream Skipping one or more lines with the same contents
    myStream Debug, T0: Ignoring global element <meta>.
    myStream Debug, T0: Ignoring global element <library_visual_scenes>.
    myStream Debug, T0: Ignoring global element <scene>.
    myStream Error, T0: Collada: File came out empty. Something is wrong here.
    AssimpImporter::import ERROR : "Collada: File came out empty. Something is wrong here." 
    2017-11-30 12:49:19.733 INFO  [515433] [AssimpGGeo::load@161] AssimpGGeo::load select START 
    AssimpImporter::select no tree 
    2017-11-30 12:49:19.733 INFO  [515433] [AssimpGGeo::load@165] AssimpGGeo::load select DONE  
    2017-11-30 12:49:19.733 ERROR [515433] [NSensorList::load@77] NSensorList::load 
     idmpath:   /Users/wangbtc/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.idmap
    2017-11-30 12:49:19.936 INFO  [515433] [*OpticksResource::getSensorList@1049] OpticksResource::getSensorList NSensorList:  NSensor count 6888 distinct identier count 684
    2017-11-30 12:49:19.937 INFO  [515433] [AssimpGGeo::convert@183] AssimpGGeo::convert ctrl 
    Segmentation fault: 11


Successful AssimpRapTest
--------------------------


::


    simon:issues blyth$ AssimpRapTest --importverbosity 3  —-loadverbosity 3 
    2017-12-01 10:40:42.183 INFO  [768371] [main@71] ok
    2017-12-01 10:40:42.183 INFO  [768371] [Opticks::dumpArgs@980] Opticks::configure argc 5
      0 : AssimpRapTest
      1 : --importverbosity
      2 : 3
      3 : —-loadverbosity
      4 : 3
    2017-12-01 10:40:42.206 INFO  [768371] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    after gg
    2017-12-01 10:40:42.210 ERROR [768371] [GGeo::loadFromG4DAE@560] GGeo::loadFromG4DAE START
    2017-12-01 10:40:42.211 INFO  [768371] [AssimpGGeo::load@137] AssimpGGeo::load  path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae query range:3153:12221 ctrl  importVerbosity 3 loaderVerbosity 0
    AssimpImporter::init verbosity 3 severity.Err Err severity.Warn Warn severity.Info Info severity.Debugging Debugging
    myStream Debug, T0: debug
    myStream Info,  T0: info
    myStream Warn,  T0: warn
    myStream Error, T0: error
    2017-12-01 10:40:42.212 INFO  [768371] [AssimpImporter::import@216] AssimpImporter::import path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae flags 32779
    myStream Info,  T0: Load /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    myStream Debug, T0: Assimp 3.1.222162994 amd64 gcc debug noboost shared singlethreaded
    myStream Info,  T0: Found a matching importer for this file format
    myStream Info,  T0: Import root directory is '/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/'
    myStream Debug, T0: ColladaParser::ReadContents <>.
    myStream Debug, T0: ColladaParser::ReadContents <COLLADA>.
    myStream Debug, T0: Collada schema version is 1.4.n
    myStream Debug, T0: ColladaParser::ReadStructure <

      >.
    myStream Debug, T0: ColladaParser::ReadStructure <asset>.
    myStream Debug, T0: ColladaParser::ReadStructure <

      >.
    myStream Debug, T0: ColladaParser::ReadStructure <library_effects>.
    myStream Debug, T0: ColladaParser::ReadStructure <

      >.
    myStream Debug, T0: ColladaParser::ReadStructure <library_geometries>.
    myStream Debug, T0: ColladaParser::ReadStructure <

      >.
    myStream Debug, T0: ColladaParser::ReadStructure <library_materials>.
    myStream Debug, T0: ColladaParser::ReadStructure <

      >.
    myStream Debug, T0: ColladaParser::ReadStructure <library_nodes>.
    myStream Debug, T0: ColladaParser::ReadExtraSceneNode START <extra>.
    myStream Skipping one or more lines with the same contents
    myStream Debug, T0: ColladaParser::ReadStructure <

      >.
    myStream Debug, T0: ColladaParser::ReadStructure <library_visual_scenes>.
    myStream Debug, T0: ColladaParser::ReadStructure <

      >.
    myStream Debug, T0: ColladaParser::ReadStructure <scene>.
    myStream Debug, T0: ColladaParser::ReadStructure <COLLADA>.
    myStream Debug, T0: ColladaParser::ReadContents <COLLADA>.
    myStream Info,  T0: Entering post processing pipeline
    myStream Debug, T0: TriangulateProcess begin
    myStream Info,  T0: TriangulateProcess finished. All polygons have been triangulated.
    myStream Debug, T0: SortByPTypeProcess begin
    myStream Info,  T0: Points: 0, Lines: 0, Triangles: 249, Polygons: 0 (Meshes, X = removed)
    myStream Debug, T0: SortByPTypeProcess finished
    myStream Debug, T0: Generate spatially-sorted vertex cache
    myStream Debug, T0: CalcTangentsProcess begin
    myStream Info,  T0: CalcTangentsProcess finished. Tangents have been calculated
    myStream Debug, T0: JoinVerticesProcess begin
    myStream Debug, T0: Mesh 0 (near_top_cover_box0xc23f970) | Verts in: 228 out: 183 | ~19.7368%
    myStream Debug, T0: Mesh 1 (RPCStrip0xc04bcb0) | Verts in: 24 out: 24 | ~0%
    myStream Debug, T0: Mesh 2 (RPCGasgap140xbf4c660) | Verts in: 24 out: 24 | ~0%
    myStream Debug, T0: Mesh 3 (RPCBarCham140xc2ba760) | Verts in: 24 out: 24 | ~0%
    myStream Debug, T0: Mesh 4 (RPCGasgap230xbf50468) | Verts in: 24 out: 24 | ~0%
    myStream Debug, T0: Mesh 5 (RPCBarCham230xc125900) | Verts in: 24 out: 24 | ~0%
    myStream Debug, T0: Mesh 6 (RPCFoam0xc21f3f8) | Verts in: 24 out: 24 | ~0%
    myStream Debug, T0: Mesh 7 (RPCMod0xc13bfd8) | Verts in: 24 out: 24 | ~0%
    myStream Debug, T0: Mesh 8 (NearRPCRoof0xc135b28) | Verts in: 24 out: 24 | ~0%
    myStream Debug, T0: Mesh 9 (near_span_hbeam0xc2a27d8) | Verts in: 72 out: 72 | ~0%
    myStream Debug, T0: Mesh 10 (near_side_short_hbeam0xc2b1ea8) | Verts in: 72 out: 72 | ~0%
    myStream Debug, T0: Mesh 11 (near_thwart_long_angle_iron0xc21e058) | Verts in: 48 out: 48 | ~0%
    myStream Debug, T0: Mesh 12 (near_diagonal_angle_iron0xc04a0e8) | Verts in: 96 out: 94 | ~2.08333%




Perhaps missing macro G4DAE_EXTRAS in the assimp build ?
-------------------------------------------------------------

::

    simon:code blyth$ grep G4DAE_EXTRAS *.*
    ColladaHelper.h:#ifdef G4DAE_EXTRAS
    ColladaHelper.h:#ifdef G4DAE_EXTRAS
    ColladaLoader.cpp:#ifdef G4DAE_EXTRAS
    ColladaLoader.cpp:#ifdef G4DAE_EXTRAS
    ColladaLoader.cpp:#ifdef G4DAE_EXTRAS
    ColladaLoader.cpp:#ifdef G4DAE_EXTRAS
    ColladaLoader.h:#ifdef G4DAE_EXTRAS
    ColladaParser.cpp:#ifdef G4DAE_EXTRAS
    ColladaParser.cpp:#ifdef G4DAE_EXTRAS
    ColladaParser.cpp:#ifdef G4DAE_EXTRAS
    ColladaParser.cpp:#ifdef G4DAE_EXTRAS
    ColladaParser.h:#define G4DAE_EXTRAS
    ColladaParser.h:#ifdef G4DAE_EXTRAS
    ColladaParser.h:#ifdef G4DAE_EXTRAS
    simon:code blyth$ 



::

    simon:assimp-fork blyth$ hash_define_without_value 
    2017-12-01 11:29:32.636 INFO  [783274] [main@13] G4DAE_EXTRAS_NO_VALUE
    2017-12-01 11:29:32.636 INFO  [783274] [main@19] G4DAE_EXTRAS_WITH_ONE
    2017-12-01 11:29:32.636 INFO  [783274] [main@26] G4DAE_EXTRAS_WITH_ZERO
    simon:assimp-fork blyth$ 




::

    0056 
      57 
      58 #ifdef G4DAE_EXTRAS
      59 const std::string ColladaParser::g4dae_bordersurface_physvolume1 = "g4dae_bordersurface_physvolume1" ;
      60 const std::string ColladaParser::g4dae_bordersurface_physvolume2 = "g4dae_bordersurface_physvolume2" ;
      61 const std::string ColladaParser::g4dae_skinsurface_volume = "g4dae_skinsurface_volume" ;
      62 
      63 const std::string ColladaParser::g4dae_opticalsurface_name   = "g4dae_opticalsurface_name" ;
      64 const std::string ColladaParser::g4dae_opticalsurface_finish = "g4dae_opticalsurface_finish" ;
      65 const std::string ColladaParser::g4dae_opticalsurface_model  = "g4dae_opticalsurface_model" ;
      66 const std::string ColladaParser::g4dae_opticalsurface_type   = "g4dae_opticalsurface_type" ;
      67 const std::string ColladaParser::g4dae_opticalsurface_value  = "g4dae_opticalsurface_value" ;
      68 #endif
      69 


    1017 void ColladaParser::ReadMaterial( Collada::Material& pMaterial)
    1018 {
    1019     while( mReader->read())
    1020     {
    1021         if( mReader->getNodeType() == irr::io::EXN_ELEMENT) {
    1022             if (IsElement("material")) {
    1023                 SkipElement();
    1024             }
    1025             else if( IsElement( "instance_effect"))
    1026             {
    1027                 // referred effect by URL
    1028                 int attrUrl = GetAttribute( "url");
    1029                 const char* url = mReader->getAttributeValue( attrUrl);
    1030                 if( url[0] != '#')
    1031                     ThrowException( "Unknown reference format");
    1032 
    1033                 pMaterial.mEffect = url+1;
    1034 
    1035                 SkipElement();
    1036             }
    1037 #ifdef G4DAE_EXTRAS
    1038             else if( IsElement( "extra"))
    1039             {
    1040                 if(!pMaterial.mExtra )
    1041                      pMaterial.mExtra = new Collada::ExtraProperties();
    1042 
    1043                 ReadExtraProperties( *pMaterial.mExtra , "extra" );
    1044             }
    1045 #endif
    1046             else
    1047             {
    1048                 // ignore the rest
    1049                 SkipElement();
    1050             }
    1051         }
    1052         else if( mReader->getNodeType() == irr::io::EXN_ELEMENT_END) {
    1053             if( strcmp( mReader->getNodeName(), "material") != 0)
    1054                 ThrowException( "Expected end of <material> element.");
    1055 
    1056             break;
    1057         }
    1058     }
    1059 }




    simon:code blyth$ grep g4dae *.*
    ColladaLoader.cpp:            const char* prefix = "g4dae_" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_bordersurface_physvolume1 = "g4dae_bordersurface_physvolume1" ; 
    ColladaParser.cpp:const std::string ColladaParser::g4dae_bordersurface_physvolume2 = "g4dae_bordersurface_physvolume2" ; 
    ColladaParser.cpp:const std::string ColladaParser::g4dae_skinsurface_volume = "g4dae_skinsurface_volume" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_opticalsurface_name   = "g4dae_opticalsurface_name" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_opticalsurface_finish = "g4dae_opticalsurface_finish" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_opticalsurface_model  = "g4dae_opticalsurface_model" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_opticalsurface_type   = "g4dae_opticalsurface_type" ;
    ColladaParser.cpp:const std::string ColladaParser::g4dae_opticalsurface_value  = "g4dae_opticalsurface_value" ;
    ColladaParser.cpp:    pProperties[g4dae_opticalsurface_name]   = pOpticalSurface.mName ; 
    ColladaParser.cpp:    pProperties[g4dae_opticalsurface_model]  = pOpticalSurface.mModel ; 
    ColladaParser.cpp:    pProperties[g4dae_opticalsurface_type]   = pOpticalSurface.mType ; 
    ColladaParser.cpp:    pProperties[g4dae_opticalsurface_finish] = pOpticalSurface.mFinish ; 
    ColladaParser.cpp:    pProperties[g4dae_opticalsurface_value]  = pOpticalSurface.mValue ; 
    ColladaParser.cpp:        pMaterial.mExtra->mProperties[g4dae_skinsurface_volume] = pSkinSurface.mVolume ; 
    ColladaParser.cpp:        pMaterial.mExtra->mProperties[g4dae_bordersurface_physvolume1] = pBorderSurface.mPhysVolume1 ; 
    ColladaParser.cpp:        pMaterial.mExtra->mProperties[g4dae_bordersurface_physvolume2] = pBorderSurface.mPhysVolume2 ; 
    ColladaParser.h:    static const std::string g4dae_bordersurface_physvolume1 ; 
    ColladaParser.h:    static const std::string g4dae_bordersurface_physvolume2 ;
    ColladaParser.h:    static const std::string g4dae_skinsurface_volume ;
    ColladaParser.h:    static const std::string g4dae_opticalsurface_name ;
    ColladaParser.h:    static const std::string g4dae_opticalsurface_finish ;
    ColladaParser.h:    static const std::string g4dae_opticalsurface_model ;
    ColladaParser.h:    static const std::string g4dae_opticalsurface_type ;
    ColladaParser.h:    static const std::string g4dae_opticalsurface_value ;
    simon:code blyth$ 



Extracts of G4DAE file : all elements inside extra are being skipped
----------------------------------------------------------------------


::

    152905     <node id="World0xc15cfc0">
    152906       <instance_geometry url="#WorldBox0xc15cf40">
    152907         <bind_material>
    152908           <technique_common>
    152909             <instance_material symbol="Vacuum" target="#__dd__Materials__Vacuum0xbf9fcc0"/>
    152910           </technique_common>
    152911         </bind_material>
    152912       </instance_geometry>
    152913       <node id="__dd__Structure__Sites__db-rock0xc15d358">
    152914         <matrix>
    152915                 -0.543174 -0.83962 0 -16520
    152916 0.83962 -0.543174 0 -802110
    152917 0 0 1 -2110
    152918 0.0 0.0 0.0 1.0
    152919 </matrix>
    152920         <instance_node url="#__dd__Geometry__Sites__lvNearSiteRock0xc030350"/>
    152921         <extra>
    152922           <meta id="/dd/Structure/Sites/db-rock0xc15d358">
    152923             <copyNo>1000</copyNo>
    152924             <ModuleName></ModuleName>
    152925           </meta>
    152926         </extra>
    152927       </node>
    152928     </node>
    152929     <extra>
    152930       <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" type="0" value="1">
    152931         <matrix coldim="2" name="REFLECTIVITY0xc04f6a8">1.5e-06 0 6.5e-06 0</matrix>
    152932         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc04f6a8"/>
    152933         <matrix coldim="2" name="RINDEX0xc33da70">1.5e-06 0 6.5e-06 0</matrix>
    152934         <property name="RINDEX" ref="RINDEX0xc33da70"/>
    152935       </opticalsurface>
    152936       <opticalsurface finish="3" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface" type="0" value="1">
    152937         <matrix coldim="2" name="BACKSCATTERCONSTANT0xc28d340">1.55e-06 0 6.2e-06 0 1.033e-05 0 1.55e-05 0</matrix>
    152938         <property name="BACKSCATTERCONSTANT" ref="BACKSCATTERCONSTANT0xc28d340"/>
    152939         <matrix coldim="2" name="REFLECTIVITY0xc563328">1.55e-06 0.0393 1.771e-06 0.0393 2.066e-06 0.0394 2.48e-06 0.03975 2.755e-06 0.04045 3.01e-06 0.04135 3.542e-06 0.0432 4.133e-06 0.04655        4.959e-06 0.0538 6.2e-06 0.067 1.033e-05 0.114 1.55e-05 0.173</matrix>
    152940         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc563328"/>
    152941         <matrix coldim="2" name="SPECULARLOBECONSTANT0xbfa85d0">1.55e-06 0 6.2e-06 0 1.033e-05 0 1.55e-05 0</matrix>
    152942         <property name="SPECULARLOBECONSTANT" ref="SPECULARLOBECONSTANT0xbfa85d0"/>
    152943         <matrix coldim="2" name="SPECULARSPIKECONSTANT0xc03fc20">1.55e-06 0 6.2e-06 0 1.033e-05 0 1.55e-05 0</matrix>
    152944         <property name="SPECULARSPIKECONSTANT" ref="SPECULARSPIKECONSTANT0xc03fc20"/>
    152945       </opticalsurface>
    152946       <opticalsurface finish="0" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceTop" type="0" value="0">
    152947         <matrix coldim="2" name="REFLECTIVITY0xc359d00">1.55e-06 0.98505 1.63e-06 0.98406 1.68e-06 0.96723 1.72e-06 0.9702 1.77e-06 0.97119 1.82e-06 0.96624 1.88e-06 0.95139 1.94e-06 0.98307 2e       -06 0.9801 2.07e-06 0.98901 2.14e-06 0.98505 2.21e-06 0.96525 2.3e-06 0.97614 2.38e-06 0.97812 2.48e-06 0.97515 2.58e-06 0.96525 2.7e-06 0.96624 2.82e-06 0.96129 2.95e-06 0.95832 3.1e-06 0.9573       3 3.26e-06 0.73656 3.44e-06 0.11583 3.65e-06 0.10395 3.88e-06 0.11682 4.13e-06 0.14256 4.43e-06 0.1188 4.77e-06 0.18018 4.96e-06 0.21384 6.2e-06 0.0099 1.033e-05 0.0099 1.55e-05 0.0099</matrix>
    152948         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc359d00"/>
    152949       </opticalsurface>
    152950       <opticalsurface finish="0" model="1" name="__dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot" type="0" value="0">
    152951         <matrix coldim="2" name="REFLECTIVITY0xc04e480">1.55e-06 0.98505 1.63e-06 0.98406 1.68e-06 0.96723 1.72e-06 0.9702 1.77e-06 0.97119 1.82e-06 0.96624 1.88e-06 0.95139 1.94e-06 0.98307 2e       -06 0.9801 2.07e-06 0.98901 2.14e-06 0.98505 2.21e-06 0.96525 2.3e-06 0.97614 2.38e-06 0.97812 2.48e-06 0.97515 2.58e-06 0.96525 2.7e-06 0.96624 2.82e-06 0.96129 2.95e-06 0.95832 3.1e-06 0.9573       3 3.26e-06 0.73656 3.44e-06 0.11583 3.65e-06 0.10395 3.88e-06 0.11682 4.13e-06 0.14256 4.43e-06 0.1188 4.77e-06 0.18018 4.96e-06 0.21384 6.2e-06 0.0099 1.033e-05 0.0099 1.55e-05 0.0099</matrix>
    152952         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc04e480"/>
    152953       </opticalsurface>
    ......
    153178       <opticalsurface finish="3" model="1" name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearDeadLinerSurface" type="0" value="0.2">
    153179         <matrix coldim="2" name="BACKSCATTERCONSTANT0xc04efd8">1.5e-06 0 6.5e-06 0</matrix>
    153180         <property name="BACKSCATTERCONSTANT" ref="BACKSCATTERCONSTANT0xc04efd8"/>
    153181         <matrix coldim="2" name="REFLECTIVITY0xc3485a0">1.55e-06 0.98 2.034e-06 0.98 2.068e-06 0.98 2.103e-06 0.98 2.139e-06 0.98 2.177e-06 0.98 2.216e-06 0.98 2.256e-06 0.98 2.298e-06 0.98 2.3       41e-06 0.98 2.386e-06 0.98 2.433e-06 0.98 2.481e-06 0.98 2.532e-06 0.982 2.585e-06 0.983 2.64e-06 0.985 2.697e-06 0.988 2.757e-06 0.99 2.82e-06 0.99 2.885e-06 0.995 2.954e-06 0.995 3.026e-06 0.       99 3.102e-06 0.99 3.181e-06 0.98 3.265e-06 0.96 3.353e-06 0.95 3.446e-06 0.94 3.545e-06 0.93 3.649e-06 0.91 3.76e-06 0.89 3.877e-06 0.87 4.002e-06 0.83 4.136e-06 0.8 6.2e-06 0.6</matrix>
    153182         <property name="REFLECTIVITY" ref="REFLECTIVITY0xc3485a0"/>
    153183         <matrix coldim="2" name="SPECULARLOBECONSTANT0xc33cb10">1.5e-06 0.85 6.5e-06 0.85</matrix>
    153184         <property name="SPECULARLOBECONSTANT" ref="SPECULARLOBECONSTANT0xc33cb10"/>
    153185         <matrix coldim="2" name="SPECULARSPIKECONSTANT0xc33cb38">1.5e-06 0 6.5e-06 0</matrix>
    153186         <property name="SPECULARSPIKECONSTANT" ref="SPECULARSPIKECONSTANT0xc33cb38"/>
    153187       </opticalsurface>
    153188       <skinsurface name="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface" surfaceproperty="__dd__Geometry__PoolDetails__NearPoolSurfaces__NearPoolCoverSurface">
    153189         <volumeref ref="__dd__Geometry__PoolDetails__lvNearTopCover0xc137060"/>
    153190       </skinsurface>
    153191       <skinsurface name="__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__RSOilSurface">
    153192         <volumeref ref="__dd__Geometry__AdDetails__lvRadialShieldUnit0xc3d7ec0"/>
    153193       </skinsurface>
    153194       <skinsurface name="__dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface" surfaceproperty="__dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface">
    153195         <volumeref ref="__dd__Geometry__AdDetails__lvAdVertiCableTray0xc3a27f0"/>
    153196       </skinsurface>


::

    simon:issues blyth$ 
    simon:issues blyth$ assimp-;assimp-c code
    simon:code blyth$ 
    simon:code blyth$ grep Ignoring\ global\ element *.*
    ColladaParser.cpp:              DefaultLogger::get()->debug( boost::str( boost::format( "Ignoring global element <%s>.") % mReader->getNodeName()));


::

     126 // Reads the contents of the file
     127 void ColladaParser::ReadContents()
     128 {
     129     while( mReader->read())
     130     {
     131         DefaultLogger::get()->debug( boost::str( boost::format( "ColladaParser::ReadContents <%s>.") % mReader->getNodeName()));
     132         // handle the root element "COLLADA"
     133         if( mReader->getNodeType() == irr::io::EXN_ELEMENT)
     134         {
     135             if( IsElement( "COLLADA"))
     136             {
     137                 // check for 'version' attribute
     138                 const int attrib = TestAttribute("version");
     139                 if (attrib != -1) {
     140                     const char* version = mReader->getAttributeValue(attrib);
     141 
     142                     if (!::strncmp(version,"1.5",3)) {
     143                         mFormat =  FV_1_5_n;
     144                         DefaultLogger::get()->debug("Collada schema version is 1.5.n");
     145                     }
     146                     else if (!::strncmp(version,"1.4",3)) {
     147                         mFormat =  FV_1_4_n;
     148                         DefaultLogger::get()->debug("Collada schema version is 1.4.n");
     149                     }
     150                     else if (!::strncmp(version,"1.3",3)) {
     151                         mFormat =  FV_1_3_n;
     152                         DefaultLogger::get()->debug("Collada schema version is 1.3.n");
     153                     }
     154                 }
     155 
     156                 ReadStructure();
     157             } else
     158             {
     159                 DefaultLogger::get()->debug( boost::str( boost::format( "Ignoring global element <%s>.") % mReader->getNodeName()));
     160                 SkipElement();
     161             }
     162         } else
     163         {
     164             // skip everything else silently
     165         }
     166     }
     167 }
     168 











::

    193 void AssimpImporter::import(unsigned int flags)
    194 {
    195     LOG(info) << "AssimpImporter::import path " << m_path << " flags " << flags ;
    196     m_process_flags = flags ;
    197 
    198     assert(m_path);
    199     m_aiscene = m_importer->ReadFile( m_path, flags );
    200 
    201     if(!m_aiscene)
    202     {
    203         printf("AssimpImporter::import ERROR : \"%s\" \n", m_importer->GetErrorString() );
    204         return ;
    205     }
    206 
    207     //dumpProcessFlags("AssimpImporter::import", flags);
    208     //dumpSceneFlags("AssimpImporter::import", m_aiscene->mFlags);
    209 
    210     Summary("AssimpImporter::import DONE");
    211 
    212     m_tree = new AssimpTree(m_aiscene);
    213 }



::

    simon:issues blyth$ assimp-
    simon:issues blyth$ assimp-c
    simon:assimp-fork blyth$ 

    simon:code blyth$ grep File\ came\ out\ empty  *.cpp
    ColladaLoader.cpp:      throw DeadlyImportError( "Collada: File came out empty. Something is wrong here.");
    simon:code blyth$ pwd
    /usr/local/opticks/externals/assimp/assimp-fork/code
    simon:code blyth$ 




::

     126 // Imports the given file into the given scene structure. 
     127 void ColladaLoader::InternReadFile( const std::string& pFile, aiScene* pScene, IOSystem* pIOHandler)
     128 {
     129     mFileName = pFile;
     130 
     131     // clean all member arrays - just for safety, it should work even if we did not
     132     mMeshIndexByID.clear();
     133     mMaterialIndexByName.clear();
     134     mMeshes.clear();
     135     newMats.clear();
     136     mLights.clear();
     137     mCameras.clear();
     138     mTextures.clear();
     139     mAnims.clear();
     140 
     141     // parse the input file
     142     ColladaParser parser( pIOHandler, pFile);
     143 
     144     if( !parser.mRootNode)
     145         throw DeadlyImportError( "Collada: File came out empty. Something is wrong here.");
     146 


::

    simon:code blyth$ l Collada*
    -rw-r--r--  1 blyth  staff   65345 Aug 30 13:33 ColladaLoader.cpp
    -rw-r--r--  1 blyth  staff    9676 Aug 30 13:26 ColladaLoader.h
    -rw-r--r--  1 blyth  staff   33508 Jun 14 13:10 ColladaExporter.cpp
    -rw-r--r--  1 blyth  staff    5987 Jun 14 13:10 ColladaExporter.h
    -rw-r--r--  1 blyth  staff   18318 Jun 14 13:10 ColladaHelper.h
    -rw-r--r--  1 blyth  staff  109145 Jun 14 13:10 ColladaParser.cpp
    -rw-r--r--  1 blyth  staff   15807 Jun 14 13:10 ColladaParser.h
    simon:code blyth$ 



Hmm how to switch on debug in the ColladaParser ?

::

     126 // Reads the contents of the file
     127 void ColladaParser::ReadContents()
     128 {
     129     while( mReader->read())
     130     {
     131         DefaultLogger::get()->debug( boost::str( boost::format( "ColladaParser::ReadContents <%s>.") % mReader->getNodeName()));
     132         // handle the root element "COLLADA"
     133         if( mReader->getNodeType() == irr::io::EXN_ELEMENT)
     134         {
     135             if( IsElement( "COLLADA"))
     136             {



* https://github.com/simoncblyth/assimp/commit/caa047509302a5d9d4f0fcb3fe736332330ef1af






Try assimp build without the macro : but doesnt compile, so red herring
------------------------------------------------------------------------

From some minor mistake assimp doesnt compile without the macro, 
suggesting that the macro is a red herring.
 


::

    simon:assimp-fork blyth$ git status
    On branch master
    Your branch is up-to-date with 'origin/master'.

    nothing to commit, working directory clean
    simon:assimp-fork blyth$ 
    simon:assimp-fork blyth$ 
    simon:assimp-fork blyth$ 
    simon:assimp-fork blyth$ ls
    AssimpBuildTreeSettings.cmake.in    CodeConventions.txt         assimp-config.cmake.in          include                 test
    AssimpConfig.cmake.in           INSTALL                 assimp.pc.in                packaging               tools
    AssimpConfigVersion.cmake.in        LICENSE                 cmake-modules               port                    workspaces
    CHANGES                 README                  code                    revision.h.in
    CMakeLists.txt              Readme.md               contrib                 samples
    CREDITS                 assimp-config-version.cmake.in      doc                 scripts
    simon:assimp-fork blyth$ vi code/ColladaParser.*
    2 files to edit
    simon:assimp-fork blyth$ 

    simon:assimp-fork blyth$ git diff
    diff --git a/code/ColladaParser.h b/code/ColladaParser.h
    index 4c81d8a..10cdb1b 100644
    --- a/code/ColladaParser.h
    +++ b/code/ColladaParser.h
    @@ -45,7 +45,7 @@ OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
     #ifndef AI_COLLADAPARSER_H_INC
     #define AI_COLLADAPARSER_H_INC
     
    -#define G4DAE_EXTRAS
    +//#define G4DAE_EXTRAS^M
     
     #include "irrXMLWrapper.h"
     #include "ColladaHelper.h"
    simon:assimp-fork blyth$ 



::

    simon:assimp-fork blyth$ assimp-
    simon:assimp-fork blyth$ assimp--
    === assimp-get : already did "git clone http://github.com/simoncblyth/assimp.git assimp-fork" from /usr/local/opticks/externals/assimp
    === assimp-cmake : configured already : use assimp-configure to reconfigure
    Scanning dependencies of target assimp
    [  1%] Building CXX object code/CMakeFiles/assimp.dir/ImporterRegistry.cpp.o
    [  2%] Building CXX object code/CMakeFiles/assimp.dir/ColladaLoader.cpp.o
    [  3%] Building CXX object code/CMakeFiles/assimp.dir/ColladaParser.cpp.o
    /usr/local/opticks/externals/assimp/assimp-fork/code/ColladaParser.cpp:2797:17: error: use of undeclared identifier 'ReadExtraSceneNode'
                    ReadExtraSceneNode() ;  
                    ^
    1 error generated.
    make[2]: *** [code/CMakeFiles/assimp.dir/ColladaParser.cpp.o] Error 1
    make[1]: *** [code/CMakeFiles/assimp.dir/all] Error 2
    make: *** [all] Error 2
    [  1%] Building CXX object code/CMakeFiles/assimp.dir/ColladaParser.cpp.o
    /usr/local/opticks/externals/assimp/assimp-fork/code/ColladaParser.cpp:2797:17: error: use of undeclared identifier 'ReadExtraSceneNode'
                    ReadExtraSceneNode() ;  
                    ^
    1 error generated.
    make[2]: *** [code/CMakeFiles/assimp.dir/ColladaParser.cpp.o] Error 1
    make[1]: *** [code/CMakeFiles/assimp.dir/all] Error 2
    make: *** [all] Error 2
    === assimp-rpath-kludge : already present : libassimp.3.dylib
    simon:assimp-fork blyth$ 
    simon:assimp-fork blyth$ 


::

    imon:assimp-fork blyth$ git checkout code/ColladaParser.h
    simon:assimp-fork blyth$ git status
    On branch master
    Your branch is up-to-date with 'origin/master'.

    nothing to commit, working directory clean
    simon:assimp-fork blyth$ 



Perhaps Ryans build is linking against the wrong assimp ?
-----------------------------------------------------------

* rejig FindAssimp.cmake into FindOpticksAssimp.cmake to
  avoid CMake using standard file mechanism 


