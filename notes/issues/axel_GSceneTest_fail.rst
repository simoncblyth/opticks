axel-GSceneTest-fail
=====================


Expected way to make the analytic cache
------------------------------------------

::

    op --j1707 --gdml2gltf
       # convert the gdml into gltf with a python script

    op --j1707 -G
       # construct the triangulated geocache

    op --j1707 --gltf 3 -G
       # add analytic parts to the geocache


* working to move gdml2gltf into source level, so users
  only need to do that when adding their own geometry, not
  when using other peoples geometry

* actually the two -G are not needed, think just the gltf one will do both


TODO
-----

* slim all the gltf, then commit 

* are delaying the migration due to Ryans debugging 
* make OPTICKS_RESOURCE_LAYOUT 1 the default, and get rid of the envvar see :doc:`resource_review`
* get all users onboard

  * delete opticksdata and pull it again
  * one command to remake geocache




GPmt 
-----

* which to commit ?

::

    simon:export blyth$ hg st .
    ? DayaBay/GPmt/10/GPmt.npy
    ? DayaBay/GPmt/10/GPmt_boundaries.txt
    ? DayaBay/GPmt/10/GPmt_csg.npy
    ? DayaBay/GPmt/10/GPmt_lvnames.txt
    ? DayaBay/GPmt/10/GPmt_materials.txt
    ? DayaBay/GPmt/10/GPmt_pvnames.txt
    ? DayaBay/GPmt/2/GPmt.npy
    ? DayaBay/GPmt/2/GPmt_boundaries.txt
    ? DayaBay/GPmt/2/GPmt_csg.npy
    ? DayaBay/GPmt/2/GPmt_gcsg.npy
    ? DayaBay/GPmt/2/GPmt_lvnames.txt
    ? DayaBay/GPmt/2/GPmt_materials.txt
    ? DayaBay/GPmt/2/GPmt_pvnames.txt



Slimmimg the gltf 
-------------------

::

    opticksdata-export-du juno1707

    simon:juno1707 blyth$ mv g4_00.gltf g4_00.gltf.keep

    op --j1707 --gdml2gltf 
        ## confirmed was able to reproduce the old  g4_00.gltf.keep

    op --j1707 --gdml2gltf 
        ## now with suppressed matrix and selected

    op --j1707 --gltf 3 -G -D
        ## fails from lack of selected, fixed npy- for this
         
    NSceneTest
         ## can more quickly just check the gltf with 
         ## this after setting OPTICKS_SRCPATH envvar

    op --j1707 
         ## this fails from not enough GPU memory 

    op --j1707 --gltf 3 
         ## note stack overflow issue from the propagation
         ## and mostly messed up seqhis, 
         ##
         ## verified that this issues no related to changing 
         ## resource layout 
     
    op --gltf 3 
         ## stack overflow issues and messed up seqhis labels
         ## also present, but much less that with j1707 : just out in the tail



    simon:juno1707 blyth$ opticksdata-export-du juno1707
     81M    g4_00.gltf.keep   
     56M    g4_00.gltf

     ##  Identity suppression + select removal saves ~25M::


    opticksdata-export-du juno1707
    opticks-pretty g4_00.gltf      ## still with the duplicate pvname


    op --j1707 --gdml2gltf 
        ## now with suppressed matrix, removed selected, removed duplicated extras[pvname] 

    simon:juno1707 blyth$ opticksdata-export-du juno1707
    4.0K    BOpticksResourceTest.log
    4.0K    ChromaMaterialMap.json
    1.2M    extras
     24M    g4_00.dae
     20M    g4_00.gdml
     44M    g4_00.gltf          ##  another 12M from removing duplicated name 
     81M    g4_00.gltf.keep


     NSceneTest ## reads it OK

     op --j1707 --gltf 3 -G -D
          ## onwards to make the geocache, runs ok

     op --j1707 --gltf 3
          ## viz ok, still the stackoverflow issue, crazy seqhis

     opticks-pretty g4_00.gltf      
          ## not much low-hanging fruit left, could abbreviate boundary to b ?


Removed this duplication of a name::

        {
            "extras": {
                "boundary": "Pyrex///Vacuum",
                "pvname": "PMT_20inch_inner2_phys0x1821730"
            },
            "mesh": 21,
            "name": "PMT_20inch_inner2_phys0x1821730"
        },




* PERHAPS: also express as translation when no rotation


/usr/local/opticks/externals/yoctogl/yocto-gl/yocto/yocto_gltf.cpp::


    2861 YGLTF_API std::array<float, 16> node_transform(const node_t* node) {
    2862     auto xf = _identity_float4x4;
    2863 
    2864     // matrix
    2865     if (node->matrix != _identity_float4x4) {
    2866         xf = _float4x4_mul(node->matrix, xf);
    2867     }
    2868 




TODO : commit gltf and extras into opticksdata ?
-----------------------------------------------------

* pretty gltf too big to commit, especially as it can be generated from the normal one
* gltf also pretty big, perhaps compress ? 
* Can YoctoGL read compressed gltf ?


Suppress identity matrix is an easy way to tighten it up::

   imon:juno1707 blyth$ opticks-pretty g4_00.gltf



::

    simon:export blyth$ opticksdata-;opticksdata-export-du
    116K    DayaBay
    1.3M    DayaBay_MX_20140916-2050
    2.5M    DayaBay_MX_20141013-1711
     38M    DayaBay_VGDX_20140414-1300
    8.5M    Far_VGDX_20140414-1256
     80K    LXe
    6.8M    Lingao_VGDX_20140414-1247
    440K    dpib
     25M    juno
    325M    juno1707

    simon:export blyth$ opticksdata-;opticksdata-export-du juno1707
    4.0K    ChromaMaterialMap.json
    1.2M    extras
     24M    g4_00.dae
     20M    g4_00.gdml
     81M    g4_00.gltf
    199M    g4_00.pretty.gltf

    simon:juno1707 blyth$ opticksdata-;opticksdata-export-du DayaBay_VGDX_20140414-1300
    8.3M    extras
    6.8M    g4_00.dae
    3.9M    g4_00.gdml
    5.7M    g4_00.gltf
    2.5M    g4_00.idmap
     11M    g4_00.pretty.gltf



::

    simon:opticksdata blyth$ hg st . | grep -v json | grep -v npy | grep -v cc | grep -v bash 
    ? config/opticksdata.ini
    ? export/DayaBay/GPmt/10/GPmt_boundaries.txt
    ? export/DayaBay/GPmt/10/GPmt_lvnames.txt
    ? export/DayaBay/GPmt/10/GPmt_materials.txt
    ? export/DayaBay/GPmt/10/GPmt_pvnames.txt
    ? export/DayaBay/GPmt/2/GPmt_boundaries.txt
    ? export/DayaBay/GPmt/2/GPmt_lvnames.txt
    ? export/DayaBay/GPmt/2/GPmt_materials.txt
    ? export/DayaBay/GPmt/2/GPmt_pvnames.txt
    ? export/DayaBay_VGDX_20140414-1300/extras/csg.txt
    ? export/DayaBay_VGDX_20140414-1300/g4_00.gltf
    ? export/juno1707/extras/csg.txt
    ? export/juno1707/g4_00.gltf

    ? export/DayaBay_VGDX_20140414-1300/g4_00.pretty.gltf
    ? export/juno1707/g4_00.pretty.gltf

    simon:opticksdata blyth$ du -h export/DayaBay_VGDX_20140414-1300/g4_00.gltf
    5.7M    export/DayaBay_VGDX_20140414-1300/g4_00.gltf

    simon:opticksdata blyth$ du -h export/juno1707/g4_00.gltf
     81M    export/juno1707/g4_00.gltf



    simon:opticksdata blyth$ du -h export/DayaBay_VGDX_20140414-1300/g4_00.pretty.gltf
     11M    export/DayaBay_VGDX_20140414-1300/g4_00.pretty.gltf

    simon:opticksdata blyth$ du -h export/juno1707/g4_00.pretty.gltf
    199M    export/juno1707/g4_00.pretty.gltf



DONE
-----

* rearrange where NScene writes into geocache
* test with opticksdata readonly 



CONFIRMED : Missing nodemeta errors just from empty nodes of complete tree iteration
---------------------------------------------------------------------------------------

Not all nodes, but many : suspect from empty nodes of complete tree

::

    op --j1707 --gltf 3 -G

    simon:geocache blyth$ grep nodemeta /usr/local/opticks/geocache/j1707.log | wc -l
         134


::

    2017-11-29 17:25:20.142 INFO  [437369] [NScene::dumpRepeatCount@1477] NScene::dumpRepeatCount totCount 290254
    /usr/local/opticks/opticksdata/export/juno1707/extras/14/5/nodemeta.json
    /usr/local/opticks/opticksdata/export/juno1707/extras/14/6/nodemeta.json
    /usr/local/opticks/opticksdata/export/juno1707/extras/14/7/nodemeta.json
    /usr/local/opticks/opticksdata/export/juno1707/extras/14/8/nodemeta.json
    /usr/local/opticks/opticksdata/export/juno1707/extras/14/11/nodemeta.json
    /usr/local/opticks/opticksdata/export/juno1707/extras/14/12/nodemeta.json
    /usr/local/opticks/opticksdata/export/juno1707/extras/14/13/nodemeta.json
    /usr/local/opticks/opticksdata/export/juno1707/extras/14/14/nodemeta.json   ## these indices are 0-based


     missmeta 1-based (8) :  6 7 8 9 12 13 14 15




    simon:opticksnpy blyth$ opticks-tbool- 14
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/juno1707/extras/14/tbool14.bash
    args: 
    [2017-11-29 17:33:10,607] p50393 {/Users/blyth/opticks/ana/base.py:154} INFO - _opticks_idfilename layout 1 : idpath /usr/local/opticks/geocache/juno1707/g4_00.dae/a181a603769c1f98ad927e7367c7aa51/1 
    [2017-11-29 17:33:10,607] p50393 {/Users/blyth/opticks/ana/base.py:165} INFO - _opticks_idfilename layout 1 : idpath /usr/local/opticks/geocache/juno1707/g4_00.dae/a181a603769c1f98ad927e7367c7aa51/1 -> idfilename g4_00.dae 
    [2017-11-29 17:33:10,607] p50393 {/Users/blyth/opticks/ana/base.py:248} INFO - install_prefix : /usr/local/opticks 
    [2017-11-29 17:33:10,608] p50393 {/Users/blyth/opticks/analytic/csg.py:1092} INFO - raw name:union
    un(un(zs,di(cy,to)),cy) height:3 totnodes:15 

                              1
                             un
              2                   3    
             un                  cy

          4           5        (6)      (7)
         zs          di            
                 cy      to        
      (8)  (9)  10       11    (12) (13) (14) (15)


    [2017-11-29 17:33:10,608] p50393 {/Users/blyth/opticks/analytic/csg.py:1092} INFO - optimized name:union
    un(un(zs,di(cy,to)),cy) height:3 totnodes:15 

                         un    
         un                  cy
     zs          di            
             cy      to        


               1
          2          3
      4     5     6     7
    8  9 10  11 12 13  14  15

    simon:opticks blyth$ cd /usr/local/opticks/opticksdata/export/juno1707/extras/24
    simon:24 blyth$ l
    total 48
    -rw-r--r--  1 blyth  staff  4051 Aug  3 20:09 tbool24.bash
    -rw-r--r--  1 blyth  staff  3061 Aug  3 16:06 NNodeTest_24.cc
    -rw-r--r--  1 blyth  staff   222 Aug  3 16:06 meta.json
    -rw-r--r--  1 blyth  staff  4112 Aug  3 16:06 nodes.npy
    -rw-r--r--  1 blyth  staff   784 Aug  3 16:06 transforms.npy
    drwxr-xr-x  3 blyth  staff   102 Aug  3 12:01 0
    drwxr-xr-x  3 blyth  staff   102 Aug  3 12:01 1
    drwxr-xr-x  3 blyth  staff   102 Aug  3 12:01 15
    drwxr-xr-x  3 blyth  staff   102 Aug  3 12:01 16
    drwxr-xr-x  3 blyth  staff   102 Aug  3 12:01 2
    drwxr-xr-x  3 blyth  staff   102 Aug  3 12:01 3
    drwxr-xr-x  3 blyth  staff   102 Aug  3 12:01 31
    drwxr-xr-x  3 blyth  staff   102 Aug  3 12:01 32
    drwxr-xr-x  3 blyth  staff   102 Aug  3 12:01 4
    drwxr-xr-x  3 blyth  staff   102 Aug  3 12:01 7
    drwxr-xr-x  3 blyth  staff   102 Aug  3 12:01 8
    simon:24 blyth$ 



    2017-11-29 17:54:27.777 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/8 m_height 1 m_num_nodes 3 missmeta 0
    2017-11-29 17:54:27.784 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/9 m_height 2 m_num_nodes 7 missmeta 0
    2017-11-29 17:54:27.788 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/15 m_height 0 m_num_nodes 1 missmeta 0
    2017-11-29 17:54:27.804 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/10 m_height 2 m_num_nodes 7 missmeta 0
    2017-11-29 17:54:27.808 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/14 m_height 3 m_num_nodes 15 missmeta 8
     missmeta 1-based (8) :  6 7 8 9 12 13 14 15
    2017-11-29 17:54:27.840 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/13 m_height 3 m_num_nodes 15 missmeta 8
     missmeta 1-based (8) :  6 7 8 9 12 13 14 15
    2017-11-29 17:54:27.876 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/11 m_height 4 m_num_nodes 31 missmeta 22
     missmeta 1-based (22) :  6 7 10 11 12 13 14 15 16 17 20 21 22 23 24 25 26 27 28 29 30 31
    2017-11-29 17:54:27.923 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/12 m_height 4 m_num_nodes 31 missmeta 22
     missmeta 1-based (22) :  6 7 10 11 12 13 14 15 16 17 20 21 22 23 24 25 26 27 28 29 30 31
    2017-11-29 17:54:27.958 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/20 m_height 1 m_num_nodes 3 missmeta 0
    2017-11-29 17:54:27.980 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/18 m_height 0 m_num_nodes 1 missmeta 0
    2017-11-29 17:54:27.994 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/16 m_height 0 m_num_nodes 1 missmeta 0
    2017-11-29 17:54:27.997 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/17 m_height 0 m_num_nodes 1 missmeta 0
    2017-11-29 17:54:28.010 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/19 m_height 0 m_num_nodes 1 missmeta 0
    2017-11-29 17:54:28.012 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/26 m_height 1 m_num_nodes 3 missmeta 0
    2017-11-29 17:54:28.033 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/21 m_height 3 m_num_nodes 15 missmeta 4
     missmeta 1-based (4) :  10 11 14 15
    2017-11-29 17:54:28.037 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/22 m_height 1 m_num_nodes 3 missmeta 0
    2017-11-29 17:54:28.042 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/23 m_height 4 m_num_nodes 31 missmeta 18
     missmeta 1-based (18) :  10 11 14 15 18 19 20 21 22 23 24 25 26 27 28 29 30 31
    2017-11-29 17:54:28.047 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/25 m_height 1 m_num_nodes 3 missmeta 0
    2017-11-29 17:54:28.062 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/24 m_height 5 m_num_nodes 63 missmeta 52
     missmeta 1-based (52) :  6 7 10 11 12 13 14 15 18 19 20 21 22 23 24 25 26 27 28 29 30 31 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63
    2017-11-29 17:54:28.066 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/28 m_height 0 m_num_nodes 1 missmeta 0
    2017-11-29 17:54:28.072 INFO  [451173] [NCSG::loadNodeMetadata@413] NCSG::loadNodeMetadata m_treedir /usr/local/opticks/opticksdata/export/juno1707/extras/27 m_height 0 m_num_nodes 1 missmeta 0



Its a deep tree union of box3 and cylinder::

    opticks-;opticks-tbool 24
    opticks-;opticks-tbool-vi 24

Actually nope, that using the wrong geocache, as didnt change IDPATH

::

    simon:opticks blyth$ opticks-;opticks-tbool- 24
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/24/tbool24.bash
    args: 
    [2017-11-29 14:08:57,260] p32593 {/Users/blyth/opticks/analytic/csg.py:1092} INFO - raw name:union
    un(cy,un(cy,co)) height:2 totnodes:7 

         un            
     cy          un    
             cy      co
    [2017-11-29 14:08:57,260] p32593 {/Users/blyth/opticks/analytic/csg.py:1092} INFO - optimized name:union
    un(cy,un(cy,co)) height:2 totnodes:7 

         un            
     cy          un    
             cy      co
    [2017-11-29 14:08:57,261] p32593 {/Users/blyth/opticks/analytic/csg.py:446} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/24 
    [2017-11-29 14:08:57,261] p32593 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/24/0/0/nodemeta.json {'containerscale': '2', 'container': '1', 'idx': 0, 'verbosity': '0', 'resolution': '20', 'poly': 'IM'} 
    [2017-11-29 14:08:57,263] p32593 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/24/1/0/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 0, 'poly': 'IM'} 
    [2017-11-29 14:08:57,263] p32593 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/24/1/1/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 1, 'poly': 'IM'} 
    [2017-11-29 14:08:57,263] p32593 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/24/1/2/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 2, 'poly': 'IM'} 
    [2017-11-29 14:08:57,264] p32593 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/24/1/5/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 5, 'poly': 'IM'} 
    [2017-11-29 14:08:57,264] p32593 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/24/1/6/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 6, 'poly': 'IM'} 
    autoseqmap=TO:0,SR:1,SA:0_name=24_outerfirst=1_analytic=1_csgpath=/tmp/blyth/opticks/tbool/24_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x3f,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75_autocontainer=Rock//perfectAbsorbSurface/Vacuum
    simon:opticks blyth$ 


::

    simon:opticks blyth$ op.sh --j1707 --idpath
    === op-cmdline-binary-match : finds 1st argument with associated binary : --idpath
    IDPATH /usr/local/opticks/geocache/juno1707/g4_00.dae/a181a603769c1f98ad927e7367c7aa51/1
    simon:opticks blyth$ 


After changing IDPATH realise should be from SRCFOLD as extras are regarded as sources::

    simon:24 blyth$ opticks-;opticks-tbool-vi 24


So have to add envvar separate from IDPATH now that are decoupling sources from geocache

::

    242 export OPTICKS_SRCPATH_DYB=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    243 export OPTICKS_SRCPATH_J1707=/usr/local/opticks/opticksdata/export/juno1707/g4_00.dae
    244 export OPTICKS_SRCPATH=$OPTICKS_SRCPATH_J1707


    simon:juno1707 blyth$ opticks-tbool-info

    opticks-tbool-info
    ======================

      opticks-srcfold       : /usr/local/opticks/opticksdata/export/juno1707
      opticks-srcextras     : /usr/local/opticks/opticksdata/export/juno1707/extras
      opticks-tbool-path 0  : /usr/local/opticks/opticksdata/export/juno1707/extras/0/tbool0.bash
      opticks-nnt-path 0    : /usr/local/opticks/opticksdata/export/juno1707/extras/0/NNodeTest_0.cc
     


Ahha python expecting old layout

::

    simon:juno1707 blyth$ opticks-;opticks-tbool- 24
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/juno1707/extras/24/tbool24.bash
    args: 
    Traceback (most recent call last):
      File "<stdin>", line 9, in <module>
      File "/Users/blyth/opticks/ana/base.py", line 286, in opticks_main
        opticks_environment()
      File "/Users/blyth/opticks/ana/base.py", line 278, in opticks_environment
        env = OpticksEnv()
      File "/Users/blyth/opticks/ana/base.py", line 229, in __init__
        self.setdefault("OPTICKS_IDFILENAME",      _opticks_idfilename(IDPATH))
      File "/Users/blyth/opticks/ana/base.py", line 122, in _opticks_idfilename
        assert len(elem) == 3
    AssertionError
    simon:juno1707 blyth$ 



lvid 14 is PMT ::

    simon:opticks blyth$ opticks-tbool- 14 
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/juno1707/extras/14/tbool14.bash
    args: 
    [2017-11-29 17:17:37,553] p48700 {/Users/blyth/opticks/ana/base.py:154} INFO - _opticks_idfilename layout 1 : idpath /usr/local/opticks/geocache/juno1707/g4_00.dae/a181a603769c1f98ad927e7367c7aa51/1 
    [2017-11-29 17:17:37,553] p48700 {/Users/blyth/opticks/ana/base.py:165} INFO - _opticks_idfilename layout 1 : idpath /usr/local/opticks/geocache/juno1707/g4_00.dae/a181a603769c1f98ad927e7367c7aa51/1 -> idfilename g4_00.dae 
    [2017-11-29 17:17:37,553] p48700 {/Users/blyth/opticks/ana/base.py:248} INFO - install_prefix : /usr/local/opticks 
    [2017-11-29 17:17:37,554] p48700 {/Users/blyth/opticks/analytic/csg.py:1092} INFO - raw name:union
    un(un(zs,di(cy,to)),cy) height:3 totnodes:15 

                         un    
         un                  cy
     zs          di            
             cy      to        
    [2017-11-29 17:17:37,554] p48700 {/Users/blyth/opticks/analytic/csg.py:1092} INFO - optimized name:union
    un(un(zs,di(cy,to)),cy) height:3 totnodes:15 

                         un    
         un                  cy
     zs          di            
             cy      to        
    [2017-11-29 17:17:37,554] p48700 {/Users/blyth/opticks/analytic/csg.py:446} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/14 
    [2017-11-29 17:17:37,555] p48700 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/14/0/0/nodemeta.json {'containerscale': '2', 'container': '1', 'idx': 0, 'verbosity': '0', 'resolution': '20', 'poly': 'IM'} 
    [2017-11-29 17:17:37,557] p48700 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/14/1/0/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 0, 'poly': 'IM'} 
    [2017-11-29 17:17:37,557] p48700 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/14/1/1/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 1, 'poly': 'IM'} 
    [2017-11-29 17:17:37,558] p48700 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/14/1/3/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 3, 'poly': 'IM'} 
    [2017-11-29 17:17:37,558] p48700 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/14/1/4/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 4, 'poly': 'IM'} 
    [2017-11-29 17:17:37,558] p48700 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/14/1/9/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 9, 'poly': 'IM'} 
    [2017-11-29 17:17:37,558] p48700 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/14/1/10/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 10, 'poly': 'IM'} 
    [2017-11-29 17:17:37,559] p48700 {/Users/blyth/opticks/analytic/csg.py:747} INFO - write nodemeta to /tmp/blyth/opticks/tbool/14/1/2/nodemeta.json {'verbosity': '0', 'resolution': '20', 'idx': 2, 'poly': 'IM'} 
    autoseqmap=TO:0,SR:1,SA:0_name=14_outerfirst=1_analytic=1_csgpath=/tmp/blyth/opticks/tbool/14_mode=PyCsgInBox_autoobject=Vacuum/perfectSpecularSurface//GlassSchottF2_autoemitconfig=photons:600000,wavelength:380,time:0.2,posdelta:0.1,sheetmask:0x3f,umin:0.25,umax:0.75,vmin:0.25,vmax:0.75_autocontainer=Rock//perfectAbsorbSurface/Vacuum
    simon:opticks blyth$ 






Testing With OPTICKS_RESOURCE_LAYOUT 1
----------------------------------------

::

    simon:ggeo blyth$ op.sh -G --gltf 3


Loada missing metadata errors::

    2017-11-29 13:01:51.890 INFO  [351641] [NScene::dumpRepeatCount@1477] NScene::dumpRepeatCount totCount 7744
    2017-11-29 13:01:51.959 ERROR [351641] [NCSG::LoadMetadata@355] NCSG::LoadMetadata missing metadata  treedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/0 idx 5 metapath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/0/5/nodemeta.json


Probably just need to gdml2gltf again as they are present for juno1707

*  /usr/local/opticks/geocache/j1707.log






::

    simon:juno1707 blyth$ hg st .
    ? extras/0/0/nodemeta.json
    ? extras/0/NNodeTest_0.cc
    ? extras/0/meta.json
    ? extras/0/nodes.npy
    ? extras/0/tbool0.bash
    ? extras/0/transforms.npy
    ? extras/1/0/nodemeta.json
    ? extras/1/1/nodemeta.json
    ? extras/1/2/nodemeta.json
    ? extras/1/NNodeTest_1.cc
    ? extras/1/meta.json
    ? extras/1/nodes.npy


Make sure no writing into opticksdata
---------------------------------------

::

    simon:export blyth$ l
    total 0
    drwxr-xr-x  10 blyth  staff  340 Nov 29 13:09 DayaBay_VGDX_20140414-1300
    drwxr-xr-x  11 blyth  staff  374 Nov 14 13:25 juno1707
    drwxr-xr-x   4 blyth  staff  136 Nov 14 11:28 LXe
    drwxr-xr-x   4 blyth  staff  136 Nov 11 17:03 juno
    drwxr-xr-x   6 blyth  staff  204 Sep 11 16:17 DayaBay
    drwxr-xr-x   3 blyth  staff  102 Jun 14 13:13 DayaBay_MX_20140916-2050
    drwxr-xr-x   3 blyth  staff  102 Jun 14 13:13 DayaBay_MX_20141013-1711
    drwxr-xr-x   3 blyth  staff  102 Jun 14 13:13 Far_VGDX_20140414-1256
    drwxr-xr-x   3 blyth  staff  102 Jun 14 13:13 Lingao_VGDX_20140414-1247
    drwxr-xr-x   4 blyth  staff  136 Jun 14 13:13 dpib
    simon:export blyth$ 
    simon:export blyth$ 
    simon:export blyth$ pwd
    /usr/local/opticks/opticksdata/export
    simon:export blyth$ chmod -R u-w DayaBay_VGDX_20140414-1300 
    simon:export blyth$ 

    simon:ggeo blyth$ op.sh -G --gltf 3 -D

* this gives permission denied with layout 0, succeeds with layout 1



DONE : in new layout write NScene lvlists into new idfold (not the old one: srcfold)
-------------------------------------------------------------------------------------

::

       srcfold :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300
       srcbase :  Y :              /usr/local/opticks/opticksdata/export
        idfold :  Y : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300
        idpath :  Y : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1


analytic/sc.py : writing extras+gltf need to be done together
-----------------------------------------------------------------

::

    412     def save(self, path, load_check=True, pretty_also=True):
    413         log.info("saving to %s " % path )
    414         gdir = os.path.dirname(path)
    415         self.save_extras(gdir)    # sets uri for extra external files, so must come before the json gltf save
    416 
    417         gltf = self.gltf
    418         json_save_(path, gltf)



srcfold from opticksdata
---------------------------

::

    simon:DayaBay_VGDX_20140414-1300 blyth$ l
    total 60720
    drwxr-xr-x   19 blyth  staff       646 Aug 29 10:46 g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    drwxr-xr-x   15 blyth  staff       510 Jul 13 16:48 g4_00.7cecd380789815049b2380e5959f811d.dae
    drwxr-xr-x   15 blyth  staff       510 Jul 13 14:04 g4_00.2afdb82667f76de20f0e565546dbe5e1.dae
    drwxr-xr-x   15 blyth  staff       510 Jul 13 13:48 g4_00.4baa7a574c7dd45bfe1aa5c9f622ebb7.dae
    drwxr-xr-x   15 blyth  staff       510 Jul 13 12:44 g4_00.0e689bcb706504f90f700561849028ed.dae
    drwxr-xr-x   15 blyth  staff       510 Jul 13 12:26 g4_00.d00a9521a9a628ced58541d480142b69.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  8 14:08 g4_00.495038eb12ffd551d21f50e05d9b904e.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  7 13:21 g4_00.47461040d4dc1a53a1c220fdff8b0e81.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  7 12:51 g4_00.658867c521b8ae0058a00c516cde4105.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  7 12:06 g4_00.60420969851752cc7f01c61eb6d4ec56.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  7 10:52 g4_00.9f4370cb66a18882488962cd3bcd5b00.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  7 09:42 g4_00.13b28d14fb98f106080ffaa81b291ecf.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  6 20:57 g4_00.7ecded8ae576354131804060af5dd0a1.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  6 19:27 g4_00.a430a192de1f617b85d3bc0c47426faf.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  6 19:25 g4_00.780a488e98526cf78fb14c46ff52bcd3.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  6 19:21 g4_00.05928ea493b6e1e0b6f26beda9eb369b.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  4 13:56 g4_00.450c3b9471accf34fa1e808c6c8a679a.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  4 11:42 g4_00.54dce5b6a7a226fb440eab1c42e16616.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  4 10:55 g4_00.7ed7a5aadccb0f4759f6291842731e70.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  4 09:56 g4_00.31551e658ac453a1f16fa4169b99116f.dae
    drwxr-xr-x   12 blyth  staff       408 Jul  3 21:29 g4_00.0bf1c4270d9131ed90ad6e218af1da34.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  3 15:21 g4_00.48ce6eae7a859d5555e1e21c4bee206e.dae
    drwxr-xr-x   16 blyth  staff       544 Jul  3 13:18 g4_00.4d0ba6665a8a501401e989b108a23ae1.dae
    drwxr-xr-x   15 blyth  staff       510 Jul  3 12:58 g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae.keep
    drwxr-xr-x   12 blyth  staff       408 Jul  1 22:36 g4_00.f3f705f3d7d6bf7f11563167ead1265d.dae      

    ## above all to geocache in new layout 


    drwxr-xr-x    4 blyth  staff       136 Jul  3 18:43 g4_00    

    ## contains a few lvlists (csgskip, placeholder poly) written by NScene 
    ## these need to move to geocache ?


    -rw-r--r--    1 blyth  staff  11172379 Aug  2 20:46 g4_00.pretty.gltf
    -rw-r--r--    1 blyth  staff   6005119 Aug  2 20:46 g4_00.gltf
    drwxr-xr-x  252 blyth  staff      8568 Jul  3 18:26 extras   

    ## written by gdml2dgltf, so probably need to commit em to opticksdata (<10 MB)

    -rw-r--r--    1 blyth  staff   7126305 Jun 14 13:13 g4_00.dae
    -rw-r--r--    1 blyth  staff   4111332 Jun 14 13:13 g4_00.gdml
    -rw-r--r--    1 blyth  staff   2663880 Jun 14 13:13 g4_00.idmap

     ## sources already committed to opticksdata






::

     151 NScene::NScene(const char* base, const char* name, NSceneConfig* config, int dbgnode, int scene_idx)
     152    :
     153     NGLTF(base, name, config, scene_idx),
     154     m_num_gltf_nodes(getNumNodes()),
     155     m_config(config),
     156     m_dbgnode(dbgnode),
     157     m_containment_err(0),
     158     m_verbosity(m_config->verbosity),
     159     m_num_global(0),
     160     m_num_csgskip(0),
     161     m_num_placeholder(0),
     162     m_num_selected(0),
     163     m_csgskip_lvlist(NULL),
     164     m_placeholder_lvlist(NULL),
     165     m_node_count(0),
     166     m_label_count(0),
     167     m_digest_count(new Counts<unsigned>("progenyDigest")),
     168     m_age(NScene::SecondsSinceLastWrite(base, name)),
     169     m_triple_debug(true),
     170     m_triple(NULL),
     171     m_num_triple(0)
     172 {
     173     init_lvlists(base, name);
     174     init();
     175 }





Observations
--------------

* Opticks::configureCheckGeometryFiles complaining about lack of 
  a different path than subsequently actually used ?



This is because of the argforced value 101::

    simon:ggeo blyth$ OpticksTest --gltf 101 2>&1 | cat |  grep GLTF
    2017-11-28 12:01:36.655 FATAL [30378] [Opticks::configureCheckGeometryFiles@830]  GLTFBase $TMP/nd
    2017-11-28 12:01:36.655 FATAL [30378] [Opticks::configureCheckGeometryFiles@831]  GLTFName scene.gltf
    2017-11-28 12:01:36.655 FATAL [30378] [Opticks::configureCheckGeometryFiles@832] Try to create the GLTF from GDML with eg:  op --j1707 --gdml2gltf  
                                   GLTFBase                                  $TMP/nd
                                   GLTFName                               scene.gltf
    simon:ggeo blyth$ 
    simon:ggeo blyth$ 
    simon:ggeo blyth$ OpticksTest --gltf 3 2>&1 | cat |  grep GLTF
                                   GLTFBase /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300
                                   GLTFName                               g4_00.gltf
    simon:ggeo blyth$ 



::

     798 const char* Opticks::getGLTFPath() const
     799 {
     800     return m_resource->getGLTFPath() ;
     801 }
     802 const char* Opticks::getGLTFBase() const  // config base and name only used whilst testing with gltf >= 100
     803 {
     804     int gltf = getGLTF();
     805     const char* path = getGLTFPath() ;
     806     std::string base = gltf < 100 ? BFile::ParentDir(path) : m_cfg->getGLTFBase() ;
     807     return strdup(base.c_str()) ;
     808 }
     809 const char* Opticks::getGLTFName() const
     810 {
     811     int gltf = getGLTF();
     812     const char* path = getGLTFPath() ;
     813     std::string name = gltf < 100 ? BFile::Name(path) : m_cfg->getGLTFName()  ;
     814     return strdup(name.c_str()) ;
     815 }
     816 



::

     649 void GGeo::loadAnalyticFromCache()
     650 {
     651     LOG(info) << "GGeo::loadAnalyticFromCache START" ;
     652     m_gscene = GScene::Load(m_ok, this); // GGeo needed for m_bndlib 
     653     LOG(info) << "GGeo::loadAnalyticFromCache DONE" ;
     654 }

     068 GScene* GScene::Create(Opticks* ok, GGeo* ggeo)
      69 {
      70     bool loaded = false ;
      71     GScene* scene = new GScene(ok, ggeo, loaded); // GGeo needed for m_bndlib 
      72     return scene ;
      73 }
      74 GScene* GScene::Load(Opticks* ok, GGeo* ggeo)
      75 {
      76     bool loaded = true ;
      77     GScene* scene = new GScene(ok, ggeo, loaded); // GGeo needed for m_bndlib 
      78     return scene ;
      79 }
      80 
      81 bool GScene::HasCache( Opticks* ok ) // static 
      82 {
      83     const char* idpath = ok->getIdPath();
      84     bool analytic = true ;
      85     return GGeoLib::HasCacheConstituent(idpath, analytic, 0 );
      86 }






APPROACH 
----------

* testing limited by available GDML+G4DAE export pairs

* juno processing takes too long (several minutes) for convenient test cycle, so 

  * copy opticksdata/export/DayaBay_VGDX_20140414-1300/ under a new name to act as fresh geometry test
  * OR revive G4DAE export within Opticks ? to go together with the GDML export recently revived in cfg4



Opticks::configureCheckGeometryFiles
---------------------------------------

::

     818 bool Opticks::hasGLTF() const
     819 {
     820     // lookahead to what GScene::GScene will do
     821     return NScene::Exists(getGLTFBase(), getGLTFName()) ;
     822 }
     823 
     824 
     825 void Opticks::configureCheckGeometryFiles()
     826 {
     827     if(isGLTF() && !hasGLTF())
     828     {
     829         LOG(fatal) << "gltf option is selected but there is no gltf file " ;
     830         LOG(fatal) << " GLTFBase " << getGLTFBase() ;
     831         LOG(fatal) << " GLTFName " << getGLTFName() ;
     832         LOG(fatal) << "Try to create the GLTF from GDML with eg:  op --j1707 --gdml2gltf  "  ;
     833 
     834         //setExit(true); 
     835         //assert(0);
     836     }
     837 }


TODO : relocate geocache from /usr/local/opticks/opticksdata into /usr/local/opticks/geocache
-----------------------------------------------------------------------------------------------

This long standing TODO of relocating the geocache separately from the opticksdata checkout directory, 
to avoid the very messy "hg status" in opticksdata and potential accidents, would help with 
flexibility by decoupling source geometry files from derived files.

This will mean switching "opticksdata" into "geocache" in the paths 
of all derived files, so only source files in "opticksdata" and clean "hg status".

* OpticksResource will need to distinguish source and derived


::

    simon:opticksdata blyth$ cd /usr/local/opticks
    simon:opticks blyth$ l
    total 256
    drwxr-xr-x   10 blyth  staff     340 Nov 28 11:43 opticksdata    ## this is the hg cloned dir 
    drwxr-xr-x  380 blyth  staff   12920 Nov 27 21:02 lib
    drwxr-xr-x   33 blyth  staff    1122 Nov 27 11:26 build
    drwxr-xr-x   20 blyth  staff     680 Sep 12 16:05 include
    drwxr-xr-x   20 blyth  staff     680 Sep 12 14:32 bin
    drwxr-xr-x   23 blyth  staff     782 Sep  4 18:10 gl
    drwxr-xr-x   21 blyth  staff     714 Jun 14 17:19 externals
    drwxr-xr-x    5 blyth  staff     170 Jun 14 16:23 installcache
    -rw-r--r--@   1 blyth  staff  127384 Jun 14 13:31 opticks-externals-install.txt
    simon:opticks blyth$ 

    simon:opticks blyth$ 
    simon:opticks blyth$ l opticksdata/
    total 16
    -rw-r--r--   1 blyth  staff   398 Sep 11 21:05 OpticksIDPATH.log
    drwxr-xr-x   6 blyth  staff   204 Sep 11 20:09 gensteps
    drwxr-xr-x  12 blyth  staff   408 Jul 22 10:07 export
    drwxr-xr-x   3 blyth  staff   102 Jun 14 13:13 config
    -rw-r--r--   1 blyth  staff  1150 Jun 14 13:13 opticksdata.bash
    drwxr-xr-x   3 blyth  staff   102 Jun 14 13:13 refractiveindex
    drwxr-xr-x   4 blyth  staff   136 Jun 14 13:13 resource
    simon:opticks blyth$ 




Another derived file, needing to be relocated:

::

    204 opticksdata-ini(){ echo $(opticks-prefix)/opticksdata/config/opticksdata.ini ; }
    205 opticksdata-export-ini()
    206 {
    207    local msg="=== $FUNCNAME :"
    208 
    209    opticksdata-export 
    210 
    211    local ini=$(opticksdata-ini)
    212    local dir=$(dirname $ini)
    213    mkdir -p $dir
    214 
    215    echo $msg writing OPTICKS_DAEPATH_ environment to $ini
    216    env | grep OPTICKSDATA_DAEPATH_ | sort > $ini
    217 
    218    cat $ini
    219 }


OpticksResource paths all based off the daepath
------------------------------------------------


opticksdata paths::

    simon:optickscore blyth$ cat /usr/local/opticks/opticksdata/config/opticksdata.ini
    OPTICKSDATA_DAEPATH_DFAR=/usr/local/opticks/opticksdata/export/Far_VGDX_20140414-1256/g4_00.dae
    OPTICKSDATA_DAEPATH_DLIN=/usr/local/opticks/opticksdata/export/Lingao_VGDX_20140414-1247/g4_00.dae
    OPTICKSDATA_DAEPATH_DPIB=/usr/local/opticks/opticksdata/export/dpib/cfg4.dae
    OPTICKSDATA_DAEPATH_DYB=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
    OPTICKSDATA_DAEPATH_J1707=/usr/local/opticks/opticksdata/export/juno1707/g4_00.dae
    OPTICKSDATA_DAEPATH_JPMT=/usr/local/opticks/opticksdata/export/juno/test3.dae
    OPTICKSDATA_DAEPATH_LXE=/usr/local/opticks/opticksdata/export/LXe/g4_00.dae
    simon:optickscore blyth$ 

geocache layout can ignore the root "/usr/local/opticks/opticksdata/export" just use ParentName::

    /usr/local/opticks/geocache/Far_VGDX_20140414-1256/
    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/

idpath can simplify::

    /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae

    /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/
         ## this form retains the name of src file


* idfold can come from BOpticksResource
* idpath needs to be in OpticksResource as needs the digest 

::

    2017-11-28 14:08:08.203 INFO  [63474] [OpticksResource::dumpPaths@712] dumpPaths
                 daepath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
                gdmlpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml
                gltfpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf
                metapath :  N : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.ini
               g4env_ini :  Y :     /usr/local/opticks/externals/config/geant4.ini
              okdata_ini :  Y : /usr/local/opticks/opticksdata/config/opticksdata.ini
    2017-11-28 14:08:08.204 INFO  [63474] [OpticksResource::dumpDirs@741] dumpDirs
          install_prefix :  Y :                                 /usr/local/opticks
         opticksdata_dir :  Y :                     /usr/local/opticks/opticksdata
            resource_dir :  Y :            /usr/local/opticks/opticksdata/resource
                  idpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
              idpath_tmp :  N :                                                  -
                  idfold :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300
                  idbase :  Y :              /usr/local/opticks/opticksdata/export
           detector_base :  Y :      /usr/local/opticks/opticksdata/export/DayaBay



::


    simon:opticks blyth$ OPTICKS_RESOURCE_LAYOUT=1 BOpticksResourceTest
    2017-11-28 17:54:05.733 INFO  [158492] [BOpticksResource::Summary@367] BOpticksResource::Summary layout 1
    prefix   : /usr/local/opticks
    envprefix: OPTICKS_
    getPTXPath(generate.cu.ptx) = /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
    PTXPath(generate.cu.ptx) = /usr/local/opticks/installcache/PTX/OptiXRap_generated_generate.cu.ptx
    debugging_idpath  /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    debugging_idfold  /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300
    usertmpdir ($TMP) /tmp/blyth/opticks
    ($TMPTEST)        /tmp/blyth/opticks/test
    2017-11-28 17:54:05.734 INFO  [158492] [BOpticksResource::dumpPaths@502] dumpPaths
                         g4env_ini :  Y :     /usr/local/opticks/externals/config/geant4.ini
                        okdata_ini :  Y : /usr/local/opticks/opticksdata/config/opticksdata.ini
                           srcpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
                           daepath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.dae
                          gdmlpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml
                          gltfpath :  Y : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gltf
                          metapath :  N : /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.ini
    2017-11-28 17:54:05.735 INFO  [158492] [BOpticksResource::dumpDirs@532] dumpDirs
                    install_prefix :  Y :                                 /usr/local/opticks
                   opticksdata_dir :  Y :                     /usr/local/opticks/opticksdata
                      geocache_dir :  N :                        /usr/local/opticks/geocache
                      resource_dir :  Y :            /usr/local/opticks/opticksdata/resource
                      gensteps_dir :  Y :            /usr/local/opticks/opticksdata/gensteps
                  installcache_dir :  Y :                    /usr/local/opticks/installcache
              rng_installcache_dir :  Y :                /usr/local/opticks/installcache/RNG
              okc_installcache_dir :  Y :                /usr/local/opticks/installcache/OKC
              ptx_installcache_dir :  Y :                /usr/local/opticks/installcache/PTX
                            idfold :  N : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300
                            idpath :  N : /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1
                        idpath_tmp :  N :                                                  -
    2017-11-28 17:54:05.736 INFO  [158492] [BOpticksResource::dumpNames@480] dumpNames
                            idname :  - :                         DayaBay_VGDX_20140414-1300
                            idfile :  - :                                          g4_00.dae
           OPTICKS_RESOURCE_LAYOUT :  - :                                                  1
     treedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras
    simon:opticks blyth$ 




Running with new layout before generating geocache
----------------------------------------------------

::

    87% tests passed, 36 tests failed out of 283

    Total Test time (real) = 119.24 sec

    The following tests FAILED:
        177 - GGeoTest.GMaterialLibTest (OTHER_FAULT)
        180 - GGeoTest.GScintillatorLibTest (OTHER_FAULT)
        183 - GGeoTest.GBndLibTest (OTHER_FAULT)
        184 - GGeoTest.GBndLibInitTest (OTHER_FAULT)
        195 - GGeoTest.GPartsTest (OTHER_FAULT)
        197 - GGeoTest.GPmtTest (OTHER_FAULT)
        198 - GGeoTest.BoundariesNPYTest (OTHER_FAULT)
        199 - GGeoTest.GAttrSeqTest (OTHER_FAULT)
        203 - GGeoTest.GGeoLibTest (OTHER_FAULT)
        204 - GGeoTest.GGeoTest (OTHER_FAULT)
        205 - GGeoTest.GMakerTest (OTHER_FAULT)
        212 - GGeoTest.GSurfaceLibTest (OTHER_FAULT)
        214 - GGeoTest.NLookupTest (OTHER_FAULT)
        215 - GGeoTest.RecordsNPYTest (OTHER_FAULT)
        216 - GGeoTest.GSceneTest (OTHER_FAULT)
        217 - GGeoTest.GMeshLibTest (OTHER_FAULT)
        ## got the expected errors for all the above

        222 - OpticksGeometryTest.OpticksGeometryTest (OTHER_FAULT)
        223 - OpticksGeometryTest.OpticksHubTest (OTHER_FAULT)
        ## got sensorlist errors, twas expecting 3-dot idpath structure

        241 - OptiXRapTest.OScintillatorLibTest (OTHER_FAULT)
        242 - OptiXRapTest.OOTextureTest (OTHER_FAULT)
        247 - OptiXRapTest.OOboundaryTest (OTHER_FAULT)
        248 - OptiXRapTest.OOboundaryLookupTest (OTHER_FAULT)
        252 - OptiXRapTest.OEventTest (OTHER_FAULT)
        253 - OptiXRapTest.OInterpolationTest (OTHER_FAULT)
        254 - OptiXRapTest.ORayleighTest (OTHER_FAULT)
        258 - OKOPTest.OpSeederTest (OTHER_FAULT)
        267 - cfg4Test.CMaterialLibTest (OTHER_FAULT)
        268 - cfg4Test.CMaterialTest (OTHER_FAULT)
        269 - cfg4Test.CTestDetectorTest (OTHER_FAULT)
        270 - cfg4Test.CGDMLDetectorTest (OTHER_FAULT)
        271 - cfg4Test.CGeometryTest (OTHER_FAULT)
        272 - cfg4Test.CG4Test (OTHER_FAULT)
        277 - cfg4Test.CCollectorTest (OTHER_FAULT)
        278 - cfg4Test.CInterpolationTest (OTHER_FAULT)
        280 - cfg4Test.CGROUPVELTest (OTHER_FAULT)
        283 - okg4Test.OKG4Test (OTHER_FAULT)
    Errors while running CTest
    Tue Nov 28 18:12:01 CST 2017
    opticks-t- : use -V to show output, ctest output written to /usr/local/opticks/build/ctest.log
    simon:opticks blyth$ 


Unexpected errors from 

::

    simon:opticks blyth$ OpticksGeometryTest
    2017-11-28 18:15:22.104 INFO  [180505] [Opticks::dumpArgs@968] Opticks::configure argc 1
      0 : OpticksGeometryTest
    2017-11-28 18:15:22.105 INFO  [180505] [OpticksHub::configure@236] OpticksHub::configure m_gltf 0
    2017-11-28 18:15:22.106 INFO  [180505] [OpticksHub::loadGeometry@366] OpticksHub::loadGeometry START
    2017-11-28 18:15:22.111 INFO  [180505] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    2017-11-28 18:15:22.114 INFO  [180505] [OpticksGeometry::loadGeometry@102] OpticksGeometry::loadGeometry START 
    2017-11-28 18:15:22.114 INFO  [180505] [OpticksGeometry::loadGeometryBase@134] OpticksGeometry::loadGeometryBase START 
    2017-11-28 18:15:22.812 ERROR [180505] [NSensorList::load@88] NSensorList::load idpath is expected to be in 3-parts separted by dot eg  g4_00.gdasdyig3736781.dae  idpath 
    2017-11-28 18:15:22.812 INFO  [180505] [*OpticksResource::getSensorList@1055] OpticksResource::getSensorList NSensorList:  NSensor count 0 distinct identier count 0







::

    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/GammaYIELDRATIO.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/NeutronFASTTIMECONSTANT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/NeutronSLOWTIMECONSTANT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/NeutronYIELDRATIO.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/RAYLEIGH.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/REEMISSIONPROB.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/RESOLUTIONSCALE.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/RINDEX.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/ReemissionFASTTIMECONSTANT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/ReemissionSLOWTIMECONSTANT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/ReemissionYIELDRATIO.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/SCINTILLATIONYIELD.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/SLOWCOMPONENT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/SLOWTIMECONSTANT.npy
    ? xport/DayaBay/GScintillatorLib/LiquidScintillator/YIELDRATIO.npy
    ? xport/DayaBay/GSourceLib/GSourceLib.npy
    ? xport/DayaBay/GSurfaceLib/GPropertyLibMetadata.json
    ? xport/DayaBay/GSurfaceLib/GSurfaceLib.npy
    ? xport/DayaBay/GSurfaceLib/GSurfaceLibOptical.npy
    ? xport/DayaBay/MeshIndex/GItemIndexLocal.json
    ? xport/DayaBay/MeshIndex/GItemIndexSource.json
    simon:opticksgeo blyth$ 
    simon:opticksgeo blyth$ 
    simon:opticksgeo blyth$ 
    simon:opticksgeo blyth$ 
    simon:opticksgeo blyth$ OpticksGeometryTest 




Axel reports GSceneTest fail
--------------------------------

Today I got the latest updates and also did the opticks tests (opticks-t) and got the following error:

::

    99% tests passed, 1 tests failed out of 283

    Total Test time (real) = 176.07 sec

    The following tests FAILED:
        216 - GGeoTest.GSceneTest (OTHER_FAULT)
    Errors while running CTest
    Mon Nov 27 12:58:25 CET 2017


::

    gpu-CELSIUS-R940 opticks # GSceneTest 
    2017-11-27 14:33:48.056 INFO  [6897] [Opticks::dumpArgs@958] Opticks::configure argc 3
      0 : GSceneTest
      1 : --gltf
      2 : 101
    2017-11-27 14:33:48.057 FATAL [6897] [Opticks::configureCheckGeometryFiles@819] gltf option is selected but there is no gltf file 
    2017-11-27 14:33:48.057 FATAL [6897] [Opticks::configureCheckGeometryFiles@820]  GLTFBase $TMP/nd
    2017-11-27 14:33:48.058 FATAL [6897] [Opticks::configureCheckGeometryFiles@821]  GLTFName scene.gltf
    2017-11-27 14:33:48.058 FATAL [6897] [Opticks::configureCheckGeometryFiles@822] Try to create the GLTF from GDML with eg:  op --j1707 --gdml2gltf  
    2017-11-27 14:33:48.058 INFO  [6897] [main@59] GSceneTest base $TMP/nd name scene.gltf config check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0 gltf 101
    2017-11-27 14:33:48.063 INFO  [6897] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    2017-11-27 14:33:48.071 INFO  [6897] [GMaterialLib::postLoadFromCache@70] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-11-27 14:33:48.072 INFO  [6897] [GMaterialLib::replaceGROUPVEL@560] GMaterialLib::replaceGROUPVEL  ni 38
    2017-11-27 14:33:48.083 INFO  [6897] [GGeoLib::loadConstituents@161] GGeoLib::loadConstituents mm.reldir GMergedMesh gp.reldir GParts MAX_MERGED_MESH  10
    2017-11-27 14:33:48.083 INFO  [6897] [GGeoLib::loadConstituents@168] /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-27 14:33:48.184 INFO  [6897] [GGeoLib::loadConstituents@217] GGeoLib::loadConstituents loaded 6 ridx (  0,  1,  2,  3,  4,  5,)
    2017-11-27 14:33:48.248 INFO  [6897] [GMeshLib::loadMeshes@219] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-27 14:33:48.282 INFO  [6897] [GGeo::loadAnalyticFromCache@651] GGeo::loadAnalyticFromCache START
    2017-11-27 14:33:48.354 INFO  [6897] [OpticksResource::getSensorList@1248] OpticksResource::getSensorList NSensorList:  NSensor count 6888 distinct identier count 684
    2017-11-27 14:33:48.354 INFO  [6897] [GGeoLib::loadConstituents@161] GGeoLib::loadConstituents mm.reldir GMergedMeshAnalytic gp.reldir GPartsAnalytic MAX_MERGED_MESH  10
    2017-11-27 14:33:48.354 INFO  [6897] [GGeoLib::loadConstituents@168] /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-27 14:33:48.354 INFO  [6897] [GGeoLib::loadConstituents@217] GGeoLib::loadConstituents loaded 0 ridx ()
    2017-11-27 14:33:48.354 WARN  [6897] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GNodeLibAnalytic/PVNames.txt
    2017-11-27 14:33:48.354 WARN  [6897] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GNodeLibAnalytic/LVNames.txt
    2017-11-27 14:33:48.354 WARN  [6897] [Index::load@420] Index::load FAILED to load index  idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae itemtype GItemIndex Source path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/MeshIndexAnalytic/GItemIndexSource.json Local path /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/MeshIndexAnalytic/GItemIndexLocal.json
    2017-11-27 14:33:48.354 WARN  [6897] [GItemIndex::loadIndex@176] GItemIndex::loadIndex failed for  idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae reldir MeshIndexAnalytic override NULL
    2017-11-27 14:33:48.354 FATAL [6897] [GMeshLib::loadFromCache@61]  meshindex load failure 
    GSceneTest: /home/gpu/opticks/ggeo/GMeshLib.cc:62: void GMeshLib::loadFromCache(): Assertion `has_index && " MISSING MESH INDEX : PERHAPS YOU NEED TO CREATE/RE-CREATE GEOCACHE WITH : op.sh -G "' failed.
    Aborted

I ran "op -G", but still the error occurs.




Succeeding GSceneTest
-----------------------

* note double load of GGeoLib, seems GScene not using the basis geometry approach ?



My successful GSceneTest::

    simon:issues blyth$ GSceneTest 
    2017-11-28 12:14:52.023 INFO  [36458] [Opticks::dumpArgs@968] Opticks::configure argc 3
      0 : GSceneTest
      1 : --gltf
      2 : 101
    2017-11-28 12:14:52.024 FATAL [36458] [Opticks::configureCheckGeometryFiles@829] gltf option is selected but there is no gltf file 
    2017-11-28 12:14:52.024 FATAL [36458] [Opticks::configureCheckGeometryFiles@830]  GLTFBase $TMP/nd
    2017-11-28 12:14:52.024 FATAL [36458] [Opticks::configureCheckGeometryFiles@831]  GLTFName scene.gltf
    2017-11-28 12:14:52.024 FATAL [36458] [Opticks::configureCheckGeometryFiles@832] Try to create the GLTF from GDML with eg:  op --j1707 --gdml2gltf  
    2017-11-28 12:14:52.024 INFO  [36458] [main@62] GSceneTest base $TMP/nd name scene.gltf config check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0 gltf 101
    2017-11-28 12:14:52.028 INFO  [36458] [NSceneConfig::NSceneConfig@50] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0,instance_repeat_min=400,instance_vertex_min=0]
    2017-11-28 12:14:52.031 ERROR [36458] [GSceneTest::GSceneTest@33] loadFromCache
    2017-11-28 12:14:52.034 INFO  [36458] [GMaterialLib::postLoadFromCache@70] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-11-28 12:14:52.034 INFO  [36458] [GMaterialLib::replaceGROUPVEL@560] GMaterialLib::replaceGROUPVEL  ni 38
    2017-11-28 12:14:52.040 INFO  [36458] [GGeoLib::loadConstituents@161] GGeoLib::loadConstituents mm.reldir GMergedMesh gp.reldir GParts MAX_MERGED_MESH  10
    2017-11-28 12:14:52.040 INFO  [36458] [GGeoLib::loadConstituents@168] /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-28 12:14:52.171 INFO  [36458] [GGeoLib::loadConstituents@217] GGeoLib::loadConstituents loaded 6 ridx (  0,  1,  2,  3,  4,  5,)
    2017-11-28 12:14:52.257 INFO  [36458] [GMeshLib::loadMeshes@219] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-28 12:14:52.290 ERROR [36458] [GSceneTest::GSceneTest@35] loadAnalyticFromCache
    2017-11-28 12:14:52.290 INFO  [36458] [GGeo::loadAnalyticFromCache@651] GGeo::loadAnalyticFromCache START
    2017-11-28 12:14:52.456 INFO  [36458] [*OpticksResource::getSensorList@1248] OpticksResource::getSensorList NSensorList:  NSensor count 6888 distinct identier count 684
    2017-11-28 12:14:52.456 INFO  [36458] [GGeoLib::loadConstituents@161] GGeoLib::loadConstituents mm.reldir GMergedMeshAnalytic gp.reldir GPartsAnalytic MAX_MERGED_MESH  10
    2017-11-28 12:14:52.456 INFO  [36458] [GGeoLib::loadConstituents@168] /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-28 12:14:52.603 INFO  [36458] [GGeoLib::loadConstituents@217] GGeoLib::loadConstituents loaded 6 ridx (  0,  1,  2,  3,  4,  5,)
    2017-11-28 12:14:52.679 INFO  [36458] [GMeshLib::loadMeshes@219] idpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    2017-11-28 12:14:53.220 INFO  [36458] [GGeo::loadAnalyticFromCache@653] GGeo::loadAnalyticFromCache DONE
    2017-11-28 12:14:53.220 ERROR [36458] [GSceneTest::GSceneTest@37] dumpStats
    GGeo::dumpStats
     mm  0 : vertices  204464 faces  403712 transforms   12230 itransforms       1 
     mm  1 : vertices       0 faces       0 transforms       1 itransforms    1792 
     mm  2 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  3 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  4 : vertices       8 faces      12 transforms       1 itransforms     864 
     mm  5 : vertices    1474 faces    2928 transforms       5 itransforms     672 
       totVertices    205962  totFaces    406676 
      vtotVertices   1215728 vtotFaces   2402432 (virtual: scaling by transforms)
      vfacVertices     5.903 vfacFaces     5.907 (virtual to total ratio)
    simon:issues blyth$ 


