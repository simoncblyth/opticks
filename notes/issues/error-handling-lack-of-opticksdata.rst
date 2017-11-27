error-handling-lack-of-geant4ini
===================================


::

    ur-Jens-MacBook-Pro:opticks wangbtc$ AssimpRapTest
    2017-11-26 17:41:08.358 WARN  [130356] [OpticksResource::readG4Environment@493] OpticksResource::readG4Environment MISSING FILE externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 
    2017-11-26 17:41:08.359 INFO  [130356] [main@69] ok
    2017-11-26 17:41:08.359 INFO  [130356] [Opticks::dumpArgs@958] Opticks::configure argc 1
      0 : AssimpRapTest
    2017-11-26 17:41:08.362 INFO  [130356] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    after gg
    2017-11-26 17:41:08.363 INFO  [130356] [AssimpGGeo::load@135] AssimpGGeo::load  path /Users/wangbtc/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae query range:3153:12221 ctrl  verbosity 0
    2017-11-26 17:41:08.363 INFO  [130356] [AssimpImporter::import@195] AssimpImporter::import path /Users/wangbtc/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae flags 32779
    myStream Error, T0: Collada: File came out empty. Something is wrong here.
    AssimpImporter::import ERROR : "Collada: File came out empty. Something is wrong here." 
    2017-11-26 17:41:08.517 INFO  [130356] [AssimpGGeo::load@150] AssimpGGeo::load select START 
    AssimpImporter::select no tree 
    2017-11-26 17:41:08.517 INFO  [130356] [AssimpGGeo::load@154] AssimpGGeo::load select DONE  
    2017-11-26 17:41:08.713 INFO  [130356] [*OpticksResource::getSensorList@1138] OpticksResource::getSensorList NSensorList:  NSensor count 6888 distinct identier count 684
    2017-11-26 17:41:08.713 INFO  [130356] [AssimpGGeo::convert@172] AssimpGGeo::convert ctrl 
    Segmentation fault: 11

