volume-counts
==================


GGeoLib::dump
-------------------

::

    2019-05-17 16:48:06.317 INFO  [209436] [GGeoLib::dump@322] Scene::uploadGeometry GGeoLib GGeoLib TRIANGULATED  numMergedMesh 6 ptr 0x1ac0690
    mm index   0 geocode   T                  numVolumes     366697 numFaces        6648 numITransforms           1 numITransforms*numVolumes      366697
    mm index   1 geocode   T                  numVolumes          5 numFaces        1584 numITransforms       36572 numITransforms*numVolumes      182860
    mm index   2 geocode   T                  numVolumes          6 numFaces        3648 numITransforms       20046 numITransforms*numVolumes      120276
    mm index   3 geocode   T                  numVolumes        130 numFaces        1560 numITransforms         480 numITransforms*numVolumes       62400
    mm index   4 geocode   T                  numVolumes          1 numFaces         192 numITransforms         480 numITransforms*numVolumes         480
    mm index   5 geocode   T                  numVolumes          1 numFaces        1856 numITransforms         480 numITransforms*numVolumes         480
     num_total_volumes 366697 num_instanced_volumes 366496 num_global_volumes 201
    2019-05-17 16:48:06.317 INFO  [209436] [RContext::initUniformBuffer@36] RContext::initUniformBuffer


    echo $(( 182860 + 120276 + 62400 + 480 + 480 ))


    [blyth@localhost issues]$ echo $(( 182860 + 120276 + 62400 + 480 + 480 ))
    366496


GUI check
-------------

::

    geocache-gui


Flipping the toggles to see which is which.

===============   =================  ================
mm index            gui label          notes
===============   =================  ================
   1                  in0              small PMT
   2                  in1              large PMT
   3                  in2              some TT plate, that manages to be 130 volumes 
   4                  in3              support stick
   5                  in4              support temple
===============   =================  ================



All volumes are listed in mm0 (for absolute indexing)
-------------------------------------------------------
  
* TODO: check only the non-instanced (201) are landing on GPU 

::

    320 void GGeoLib::dump(const char* msg)
    321 {
    322     LOG(info) << msg << " " << desc() ;
    323 
    324     unsigned nmm = getNumMergedMesh();
    325     unsigned num_total_volumes = 0 ;
    326     unsigned num_instanced_volumes = 0 ;
    327 
    328     for(unsigned i=0 ; i < nmm ; i++)
    329     {
    330         GMergedMesh* mm = getMergedMesh(i);
    331 
    332         unsigned numVolumes = mm ? mm->getNumVolumes() : -1 ;
    333         unsigned numITransforms = mm ? mm->getNumITransforms() : -1 ;
    334         if( i == 0 ) num_total_volumes = numVolumes ;
    335         std::cout << GMergedMesh::Desc(mm) << std::endl ;
    336         num_instanced_volumes += i > 0 ? numITransforms*numVolumes : 0 ;
    337     }
    338     std::cout
    339                 << " num_total_volumes " << num_total_volumes
    340                 << " num_instanced_volumes " << num_instanced_volumes
    341                 << " num_global_volumes " << num_total_volumes - num_instanced_volumes
    342                 << std::endl
    343                 ;
    344 
    345 }



