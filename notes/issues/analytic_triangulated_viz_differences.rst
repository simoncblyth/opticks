Analytic Triangulated Viz Differences
========================================

DONE : Mostly fixed by move to absolute tree approach
--------------------------------------------------------

Investigate and resolve differences between the branches, aiming for the      
the only significant difference to be more precise geometry intersections : everything 
else should arranged to be the same or very similar.

* far too much debug output in analytic

* view point
* source position
* mesh coloration


DONE
------

Take analogous approach to analytic GMergedMesh creation by GScene 
as that done by standard GGeo/GMergedMesh
(ie full geometry but with nodes masked according to volume range selection).

* working on this in op-pygdml by arranging for python to be run from 
  within the opticks environment




LEAVING AS IS : mesh ordering different in gdml and dae branches
---------------------------------------------------------------------

Forced to do solid name mapping to establish correspondence... why ?

::

    2017-06-23 16:55:44.638 INFO  [525994] [GScene::importMeshes@185] GScene::importMeshes START num_meshes 249
    2017-06-23 16:55:44.638 INFO  [525994] [GScene::importMeshes@200]  mesh_idx    0 tri_mesh_idx  248 soname WorldBox0xc15cf40
    2017-06-23 16:55:44.639 INFO  [525994] [GScene::importMeshes@200]  mesh_idx    1 tri_mesh_idx  247 soname near_rock0xc04ba08
    2017-06-23 16:55:44.640 INFO  [525994] [GScene::importMeshes@200]  mesh_idx    2 tri_mesh_idx   21 soname near_hall_top_dwarf0xc0316c8
    2017-06-23 16:55:44.641 INFO  [525994] [GScene::importMeshes@200]  mesh_idx    3 tri_mesh_idx    0 soname near_top_cover_box0xc23f970
    2017-06-23 16:55:44.641 INFO  [525994] [GScene::importMeshes@200]  mesh_idx    4 tri_mesh_idx    7 soname RPCMod0xc13bfd8
    2017-06-23 16:55:44.643 INFO  [525994] [GScene::importMeshes@200]  mesh_idx    5 tri_mesh_idx    6 soname RPCFoam0xc21f3f8
    2017-06-23 16:55:44.644 INFO  [525994] [GScene::importMeshes@200]  mesh_idx    6 tri_mesh_idx    3 soname RPCBarCham140xc2ba760
    2017-06-23 16:55:44.644 INFO  [525994] [GScene::importMeshes@200]  mesh_idx    7 tri_mesh_idx    2 soname RPCGasgap140xbf4c660
    2017-06-23 16:55:44.644 INFO  [525994] [GScene::importMeshes@200]  mesh_idx    8 tri_mesh_idx    1 soname RPCStrip0xc04bcb0
    2017-06-23 16:55:44.644 INFO  [525994] [GScene::importMeshes@200]  mesh_idx    9 tri_mesh_idx    5 soname RPCBarCham230xc125900
    2017-06-23 16:55:44.644 INFO  [525994] [GScene::importMeshes@200]  mesh_idx   10 tri_mesh_idx    4 soname RPCGasgap230xbf50468
    2017-06-23 16:55:44.644 INFO  [525994] [GScene::importMeshes@200]  mesh_idx   11 tri_mesh_idx    8 soname NearRPCRoof0xc135b28
    2017-06-23 16:55:44.644 INFO  [525994] [GScene::importMeshes@200]  mesh_idx   12 tri_mesh_idx   20 soname NearRPCSptRoof0xc052bc0




assimprap/AssimpGeo
----------------------

OpticksQuery volume range selection feeds into GSolid/GNode isSelected flags.



Partial Geometry Control And Analytic Geometry ?
-----------------------------------------------------



::

    045 const char* OpticksResource::DEFAULT_GEOKEY = "OPTICKSDATA_DAEPATH_DYB" ;
    046 const char* OpticksResource::DEFAULT_QUERY = "range:3153:12221" ;
    ...
    507 
    508     m_query_string = SSys::getenvvar(m_envprefix, "QUERY", DEFAULT_QUERY);
    509     m_ctrl         = SSys::getenvvar(m_envprefix, "CTRL", DEFAULT_CTRL);
    510     m_meshfix      = SSys::getenvvar(m_envprefix, "MESHFIX", DEFAULT_MESHFIX);
    511     m_meshfixcfg   = SSys::getenvvar(m_envprefix, "MESHFIX_CFG", DEFAULT_MESHFIX_CFG);
    512 
    513     m_query = new OpticksQuery(m_query_string);
    514     std::string query_digest = SDigest::md5digest( m_query_string, strlen(m_query_string));
    515 
    516     m_digest = strdup(query_digest.c_str());
    517 
    518     // idpath incorporates digest of geometry selection envvar 
    519     // allowing to benefit from caching as vary geometry selection 
    520     // while still only having a single source geometry file.
    521 
    522     if(m_daepath)
    523     {
    524         std::string kfn = BStr::insertField( m_daepath, '.', -1 , m_digest );



Launchers
------------


Default triangulated::

    simon:~ blyth$ t op
    op () 
    { 
        opticks-;
        op.sh $*
    }


Analytic::

    simon:issues blyth$ tgltf-
    simon:issues blyth$ t tgltf-gdml
    tgltf-gdml () 
    { 
        TGLTFPATH=$($FUNCNAME- 2>/dev/null) tgltf-- $*
    }
    simon:issues blyth$ t tgltf--
    tgltf-- () 
    { 
        local msg="=== $FUNCNAME :";
        tgltf-;
        local cmdline=$*;
        local tgltfpath=${TGLTFPATH:-$TMP/nd/scene.gltf};
        local tgltfpretty=${tgltfpath/.gltf}.pretty.gltf;
        cat $tgltfpath | python -m json.tool > $tgltfpretty;
        echo $msg wrote prettified gltf to $tgltfpretty;
        local gltf=1;
        op.sh $cmdline \
                  --debugger \
                  --gltf $gltf \
                  --gltfbase $(dirname $tgltfpath) \
                  --gltfname $(basename $tgltfpath) \ 
                  --gltftarget $(tgltf-target) \
                  --target 3 \
                  --animtimemax 10 \
                  --timemax 10 \
                  --geocenter \
                  --eye 1,0,0 \
                  --dbganalytic \
                  --tag $(tgltf-tag) \
                  --cat $(tgltf-det) \
                  --save
    }






Triangulated::


    simon:~ blyth$ op
    288 -rwxr-xr-x  1 blyth  staff  143804 Jun 21 20:50 /usr/local/opticks/lib/OKTest
    proceeding : /usr/local/opticks/lib/OKTest
    2017-06-22 17:18:17.507 INFO  [302738] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
    2017-06-22 17:18:17.676 INFO  [302738] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-06-22 17:18:17.789 INFO  [302738] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-06-22 17:18:17.875 INFO  [302738] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-06-22 17:18:17.875 INFO  [302738] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-06-22 17:18:17.875 INFO  [302738] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-06-22 17:18:17.875 INFO  [302738] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-06-22 17:18:17.880 INFO  [302738] [GGeo::loadAnalyticPmt@750] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0

    2017-06-22 17:18:17.889 INFO  [302738] [*Opticks::makeSimpleTorchStep@1198] Opticks::makeSimpleTorchStep config  cfg NULL
    2017-06-22 17:18:17.889 INFO  [302738] [OpticksGen::targetGenstep@130] OpticksGen::targetGenstep setting frame 3153 0.5432,-0.8396,0.0000,0.0000 0.8396,0.5432,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 -18079.4531,-799699.4375,-6605.0000,1.0000
    2017-06-22 17:18:17.889 FATAL [302738] [GenstepNPY::setPolarization@221] GenstepNPY::setPolarization pol 0.0000,0.0000,0.0000,0.0000 npol nan,nan,nan,nan m_polw nan,nan,nan,430.0000
    2017-06-22 17:18:17.889 INFO  [302738] [SLog::operator@15] OpticksHub::OpticksHub DONE



    2017-06-22 17:18:17.890 FATAL [302738] [OpticksHub::configureState@196] OpticksHub::configureState NState::description /Users/blyth/.opticks/dayabay/State state dir /Users/blyth/.opticks/dayabay/State
    2017-06-22 17:18:17.894 WARN  [302738] [OpticksViz::prepareScene@176] OpticksViz::prepareScene using non-standard rendermode 
    2017-06-22 17:18:18.655 INFO  [302738] [OpticksViz::uploadGeometry@231] Opticks time 0.0000,200.0000,50.0000,0.0000 space -16520.0000,-802110.0000,-7125.0000,7710.5625 wavelength 60.0000,820.0000,20.0000,760.0000
    2017-06-22 17:18:18.708 INFO  [302738] [OpticksGeometry::setTarget@129] OpticksGeometry::setTarget  based on CenterExtent from m_mesh0  target 0 aim 1 ce  -16520 -802110 -7125 7710.56
    2017-06-22 17:18:18.708 INFO  [302738] [Composition::setCenterExtent@991] Composition::setCenterExtent ce -16520.0000,-802110.0000,-7125.0000,7710.5625
    2017-06-22 17:18:18.708 INFO  [302738] [SLog::operator@15] OpticksViz::OpticksViz DONE
    2017-06-22 17:18:18.951 INFO  [302738] [SLog::operator@15] OScene::OScene DONE
    2017-06-22 17:18:18.951 FATAL [302738] [*OContext::addEntry@44] OContext::addEntry G
    2017-06-22 17:18:18.951 INFO  [302738] [SLog::operator@15] OEvent::OEvent DONE
    2017-06-22 17:18:20.227 INFO  [302738] [SLog::operator@15] OPropagator::OPropagator DONE
    2017-06-22 17:18:20.227 INFO  [302738] [SLog::operator@15] OpEngine::OpEngine DONE
    2017-06-22 17:18:20.245 FATAL [302738] [*OContext::addEntry@44] OContext::addEntry P
    2017-06-22 17:18:20.245 INFO  [302738] [SLog::operator@15] OKGLTracer::OKGLTracer DONE
    2017-06-22 17:18:20.245 INFO  [302738] [SLog::operator@15] OKPropagator::OKPropagator DONE
    OKMgr::init
       OptiXVersion :            3080
    2017-06-22 17:18:20.245 INFO  [302738] [SLog::operator@15] OKMgr::OKMgr DONE
    2017-06-22 17:18:20.246 INFO  [302738] [OpticksRun::setGensteps@81] OpticksRun::setGensteps 1,6,4
    2017-06-22 17:18:20.246 INFO  [302738] [OpticksRun::passBaton@95] OpticksRun::passBaton nopstep 0x7ff3e924e540 genstep 0x7ff3e494d580
    2017-06-22 17:18:20.246 FATAL [302738] [OKPropagator::propagate@65] OKPropagator::propagate(1) OK INTEROP DEVELOPMENT
    2017-06-22 17:18:20.246 INFO  [302738] [Composition::setCenterExtent@991] Composition::setCenterExtent ce -18079.4531,-799699.4375,-6605.0000,1000.0000
    2017-06-22 17:18:20.246 INFO  [302738] [OpticksHub::target@461] OpticksHub::target evt Evt /tmp/blyth/opticks/evt/dayabay/torch/1 20170622_171820 /usr/local/opticks/lib/OKTest gsce -18079.4531,-799699.4375,-6605.0000,1000.0000
    2017-06-22 17:18:20.246 INFO  [302738] [OpticksViz::uploadEvent@269] OpticksViz::uploadEvent (1)
    2017-06-22 17:18:20.248 INFO  [302738] [Rdr::upload@303]       axis_attr vpos cn        3 sh                3,3,4 id    21 dt   0x7ff3e350d780 hd     Y nb        144 GL_STATIC_DRAW
    2017-06-22 17:18:20.249 INFO  [302738] [Rdr::upload@303]    genstep_attr vpos cn        1 sh                1,6,4 id    22 dt   0x7ff3e494e550 hd     Y nb         96 GL_STATIC_DRAW
    2017-06-22 17:18:20.252 INFO  [302738] [Rdr::upload@303]    nopstep_attr vpos cn        0 sh                0,4,4 id    23 dt              0x0 hd     N nb          0 GL_STATIC_DRAW
    2017-06-22 17:18:20.254 INFO  [302738] [Rdr::upload@303]     photon_attr vpos cn   100000 sh           100000,4,4 id    24 dt              0x0 hd     N nb    6400000 GL_DYNAMIC_DRAW
    2017-06-22 17:18:20.265 INFO  [302738] [Rdr::upload@303]     record_attr rpos cn  1000000 sh        100000,10,2,4 id    25 dt              0x0 hd     N nb   16000000 GL_STATIC_DRAW





Analytic source targetting fails to get the correct transform::

    2017-06-22 20:18:13.044 INFO  [398292] [GScene::init@114] GScene::init DONE
    2017-06-22 20:18:13.053 INFO  [398292] [*Opticks::makeSimpleTorchStep@1206] Opticks::makeSimpleTorchStep config  cfg NULL
    2017-06-22 20:18:13.053 WARN  [398292] [*GMesh::getTransform@869] GMesh::getTransform out of bounds  m_num_solids 1660 index 3153
    2017-06-22 20:18:13.053 INFO  [398292] [OpticksGen::targetGenstep@130] OpticksGen::targetGenstep setting frame 3153 1.0000,0.0000,0.0000,0.0000 0.0000,1.0000,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 0.0000,0.0000,0.0000,1.0000
    2017-06-22 20:18:13.053 FATAL [398292] [GenstepNPY::setPolarization@221] GenstepNPY::setPolarization pol 0.0000,0.0000,0.0000,0.0000 npol nan,nan,nan,nan m_polw nan,nan,nan,430.0000
    2017-06-22 20:18:13.053 INFO  [398292] [SLog::operator@15] OpticksHub::OpticksHub DONE
    2017-06-22 20:18:15.810 INFO  [398292] [OpticksGeometry::setTarget@130] OpticksGeometry::setTarget  based on CenterExtent from m_mesh0  target 0 aim 1 ce  2871 0 -41 3005
    2017-06-22 20:18:15.810 INFO  [398292] [Composition::setCenterExtent@991] Composition::setCenterExtent ce 2871.0000,0.0000,-41.0000,3005.0000
    2017-06-22 20:18:15.810 INFO  [398292] [SLog::operator@15] OpticksViz::OpticksViz DONE

Triangulated::


    2017-06-22 17:18:17.889 INFO  [302738] [*Opticks::makeSimpleTorchStep@1198] Opticks::makeSimpleTorchStep config  cfg NULL
    2017-06-22 17:18:17.889 INFO  [302738] [OpticksGen::targetGenstep@130] OpticksGen::targetGenstep setting frame 3153 0.5432,-0.8396,0.0000,0.0000 0.8396,0.5432,0.0000,0.0000 0.0000,0.0000,1.0000,0.0000 -18079.4531,-799699.4375,-6605.0000,1.0000
    2017-06-22 17:18:17.889 FATAL [302738] [GenstepNPY::setPolarization@221] GenstepNPY::setPolarization pol 0.0000,0.0000,0.0000,0.0000 npol nan,nan,nan,nan m_polw nan,nan,nan,430.0000
    2017-06-22 17:18:17.889 INFO  [302738] [SLog::operator@15] OpticksHub::OpticksHub DONE





::

    114 void OpticksGen::targetGenstep( GenstepNPY* gs )
    115 {
    116     // targetted positioning and directioning of the torch requires geometry info, 
    117     // which is not available within npy- so need to externally setFrameTransform
    118     // based on integer frame volume index
    119 
    120     if(gs->isFrameTargetted())
    121     {
    122         LOG(info) << "OpticksGen::targetGenstep frame targetted already  " << gformat(gs->getFrameTransform()) ;
    123     }
    124     else
    125     {
    126         if(m_ggeo)
    127         {
    128             glm::ivec4& iframe = gs->getFrame();
    129             glm::mat4 transform = m_ggeo->getTransform( iframe.x );
    130             LOG(info) << "OpticksGen::targetGenstep setting frame " << iframe.x << " " << gformat(transform) ;
    131             gs->setFrameTransform(transform);
    132         }
    133         else
    134         {
    135             LOG(warning) << "OpticksGen::targetGenstep SKIP AS NO GEOMETRY " ;
    136         }
    137     }
    138 }

    1517 glm::mat4 GGeo::getTransform(int index)
    1518 {
    1519     glm::mat4 vt ;
    1520     if(index > -1)
    1521     {
    1522         GMergedMesh* mesh0 = getMergedMesh(0);
    1523         float* transform = mesh0 ? mesh0->getTransform(index) : NULL ;
    1524         if(transform) vt = glm::make_mat4(transform) ;
    1525     }
    1526     return vt ;
    1527 }

    GLTF mode grabbing the GScene/GGeoLib merged mesh

    Where is partial geometry offsetting handled for tri mode ?
    The target 3153 is a full geometry index ... 

    0480 GGeoLib* GGeo::getGeoLib()
     481 {
     482     return m_gltf > 0 ? m_geolib_analytic : m_geolib ;
     483 }
     484 
     485 unsigned int GGeo::getNumMergedMesh()
     486 {
     487     GGeoLib* geolib = getGeoLib() ;
     488     assert(geolib);
     489     return geolib->getNumMergedMesh();
     490 }
     491 
     492 GMergedMesh* GGeo::getMergedMesh(unsigned int index)
     493 {
     494     GGeoLib* geolib = getGeoLib() ;
     495     assert(geolib);
     496 
     497     GMergedMesh* mm = geolib->getMergedMesh(index);
     498 


     864 float* GMesh::getTransform(unsigned int index)
     865 {
     866     if(index >= m_num_solids)
     867     {
     868        // assert(0);
     869         LOG(warning) << "GMesh::getTransform out of bounds "
     870                      << " m_num_solids " << m_num_solids
     871                      << " index " << index
     872                      ;
     873     }
     874     return index < m_num_solids ? m_transforms + index*16 : NULL  ;
     875 }


As shown by GGeoLibTest mm0 has all transforms for all 12230 volumes are in cache, 
however the nf/nv of ni only switch on within the volume selection range.
So its better to think of geocache volume range selection 
as full geometry with the non-selected mesh faces switched off.

::

     3141 ni[      0      0   3141   2968 ] id[   3141     15     10      0 ]
     3142 ni[      0      0   3142   2968 ] id[   3142     15     10      0 ]
     3143 ni[      0      0   3143   2968 ] id[   3143     15     10      0 ]
     3144 ni[      0      0   3144   2968 ] id[   3144     15     10      0 ]
     3145 ni[      0      0   3145   2968 ] id[   3145     15     10      0 ]
     3146 ni[      0      0   3146   2968 ] id[   3146     15     10      0 ]
     3147 ni[      0      0   3147      1 ] id[   3147    246     11      0 ]
     3148 ni[      0      0   3148   3147 ] id[   3148    236     12      0 ]
     3149 ni[      0      0   3149   3148 ] id[   3149    234     13      0 ]
     3150 ni[      0      0   3150   3149 ] id[   3150    232     14      0 ]
     3151 ni[      0      0   3151   3150 ] id[   3151    213     15      0 ]
     3152 ni[      0      0   3152   3151 ] id[   3152    211     16      0 ]
     3153 ni[     96     50   3153   3152 ] id[   3153    192     17      0 ]
     3154 ni[     96     50   3154   3153 ] id[   3154     94     18      0 ]
     3155 ni[     96     50   3155   3154 ] id[   3155     90     19      0 ]
     3156 ni[    288    146   3156   3155 ] id[   3156     42     20      0 ]
     3157 ni[    332    168   3157   3156 ] id[   3157     37     21      0 ]
     3158 ni[    288    146   3158   3157 ] id[   3158     24     22      0 ]
     3159 ni[    288    146   3159   3158 ] id[   3159     22     23      0 ]
     3160 ni[     92     48   3160   3158 ] id[   3160     23     23      0 ]
     3161 ni[    384    168   3161   3157 ] id[   3161     25     22      0 ]
     3162 ni[    384    168   3162   3157 ] id[   3162     26     22      0 ]
     3163 ni[    192     96   3163   3157 ] id[   3163     27     24      0 ]
     3164 ni[     96     50   3164   3157 ] id[   3164     28     25      0 ]

Instanced geometry has nf/nv of zero despite being within the selected volume range, 
as those are not in global mm0 but rather instanced mm1:: 

     6675 ni[     12      8   6675   3152 ] id[   6675    198     87      0 ]
     6676 ni[     12      8   6676   3152 ] id[   6676    198     87      0 ]
     6677 ni[      0      0   6677   3152 ] id[   6677     47     81      0 ]
     6678 ni[      0      0   6678   6677 ] id[   6678     46     28      0 ]
     6679 ni[      0      0   6679   6678 ] id[   6679     43     29   2199 ]
     6680 ni[      0      0   6680   6678 ] id[   6680     44     30      0 ]
     6681 ni[      0      0   6681   6678 ] id[   6681     45     30      0 ]
     6682 ni[    192     96   6682   3152 ] id[   6682    193     82      0 ]
     6683 ni[    192     96   6683   3152 ] id[   6683    194     83      0 ]
     6684 ni[     12      8   6684   3152 ] id[   6684    195     84      0 ]



