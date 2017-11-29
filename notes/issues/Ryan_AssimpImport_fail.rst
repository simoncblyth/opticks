Ryan_AssimpImport_fail
========================

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






::

    simon:assimprap blyth$ AssimpRapTest --importverbosity 3
    2017-11-29 12:31:21.209 INFO  [332263] [main@71] ok
    2017-11-29 12:31:21.209 INFO  [332263] [Opticks::dumpArgs@978] Opticks::configure argc 3
      0 : AssimpRapTest
      1 : --importverbosity
      2 : 3
    2017-11-29 12:31:21.213 INFO  [332263] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    after gg
    2017-11-29 12:31:21.217 ERROR [332263] [GGeo::loadFromG4DAE@560] GGeo::loadFromG4DAE START
    2017-11-29 12:31:21.217 INFO  [332263] [AssimpGGeo::load@137] AssimpGGeo::load  path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae query range:3153:12221 ctrl  importVerbosity 3 loaderVerbosity 0
    AssimpImporter::init verbosity 3 severity.Err Err severity.Warn Warn severity.Info Info severity.Debugging Debugging
    myStream Debug, T0: debug
    myStream Info,  T0: info
    myStream Warn,  T0: warn
    myStream Error, T0: error
    2017-11-29 12:31:21.217 INFO  [332263] [AssimpImporter::import@216] AssimpImporter::import path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae flags 32779
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


