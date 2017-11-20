tboolean-prism-convexpolyhedron-meta-assert
=============================================


FIXED by adding CMaker::ConvertConvexPolyhedron using G4TessellatedSolid
--------------------------------------------------------------------------

* passed srcverts srcfaces from python thru to NCSG/nnode/nconvexpolyhedron
* used these to populate G4TessellatedSolid



issue CMaker::ConvertPrimitive missing G4 conversion imp for convexpolyhedron other than trapezoid
---------------------------------------------------------------------------------------------------------

* initially noted from CMaker::ConvertPrimitive assert from missing meta

* actually the issue is a wider one of missing implementation for the conversion 
  of a solid defined by planes into G4 solids  


::

    tboolean-;tboolean-prism --okg4 -D

::

    477     else if(node->type == CSG_TRAPEZOID || node->type == CSG_SEGMENT || node->type == CSG_CONVEXPOLYHEDRON)
    478     {
    479         NParameters* meta = node->meta ;
    480         assert(meta);
    481 
    482         std::string src_type = meta->getStringValue("src_type");
    483         if(src_type.compare("trapezoid")==0)
    484         {
    485             float src_z = meta->get<float>("src_z");
    486             float src_x1 = meta->get<float>("src_x1");
    487             float src_y1 = meta->get<float>("src_y1");
    488             float src_x2 = meta->get<float>("src_x2");
    489             float src_y2 = meta->get<float>("src_y2");
    490 
    491             G4Trd* tr = new G4Trd( name, src_x1, src_x2, src_y1, src_y2, src_z );
    492             result = tr ;
    493         }
    494         else
    495         {
    496             assert(0);
    497         }  
    498     }
    499     else
    500     {
    501         LOG(fatal) << "CMaker::ConvertPrimitive MISSING IMP FOR  " << name ;
    502         assert(0);
    503     }


FIXED Missing Node Metadata : there was json naming convention mismatch
-------------------------------------------------------------------------

::


    2017-11-20 14:37:19.337 INFO  [5917208] [*NCSGList::createUniverse@160] NCSGList::createUniverse bnd0 Rock//perfectAbsorbSurface/Vacuum ubnd Rock///Rock scale 1 delta 1
    2017-11-20 14:37:19.338 ERROR [5917208] [*NCSG::LoadMetadata@332] NCSG::LoadMetadata missing metadata  treedir /tmp/blyth/opticks/tboolean-prism--/0 idx 0 metapath /tmp/blyth/opticks/tboolean-prism--/0/0/meta.json

::

    2017-11-20 14:43:26,310] p61036 {/Users/blyth/opticks/analytic/csg.py:443} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tboolean-prism-- 
    [2017-11-20 14:43:26,311] p61036 {/Users/blyth/opticks/analytic/csg.py:706} INFO - write nodemeta to /tmp/blyth/opticks/tboolean-prism--/0/0/nodemeta.json {'verbosity': '0', 'resolution': '40', 'idx': 0, 'poly': 'IM', 'ctrl': '0'} 
    [2017-11-20 14:43:26,317] p61036 {/Users/blyth/opticks/analytic/csg.py:706} INFO - write nodemeta to /tmp/blyth/opticks/tboolean-prism--/1/0/nodemeta.json {'src_depth': 300, 'src_height': 200, 'ctrl': '0', 'verbosity': '0', 'poly': 'IM', 'idx': 0, 'src_angle': 45, 'src_type': 'prism', 'resolution': '40'} 
    analytic=1_csgpath=/tmp/blyth/opticks/tboolean-prism--_mode=PyCsgInBox_outerfirst=1_name=tboolean-prism--


::

    (lldb) p meta
    (NParameters *) $0 = 0x000000010c6c5420
    (lldb) p *meta
    (NParameters) $1 = {
      m_parameters = size=9 {
        [0] = (first = "src_depth", second = "300")
        [1] = (first = "src_height", second = "200")
        [2] = (first = "ctrl", second = "0")
        [3] = (first = "verbosity", second = "0")
        [4] = (first = "poly", second = "IM")
        [5] = (first = "idx", second = "0")
        [6] = (first = "src_angle", second = "45")
        [7] = (first = "src_type", second = "prism")
        [8] = (first = "resolution", second = "40")
      }
      m_lines = size=0 {}
    }





Approach to converting some CSG_CONVEXPOLYHEDRON into G4TessellatedSolid(?)  
-----------------------------------------------------------------------------

* Opticks CSG_CONVEXPOLYHEDRON is just a bunch of planes that are assumed to form a polyhedron  

* G4TessellatedSolid is a collection of tri + quad facets

  * https://geant4.web.cern.ch/geant4/G4UsersDocuments/UsersGuides/ForApplicationDeveloper/html/Detector/geomSolids.html
  * g4-;g4-cls G4TessellatedSolid   # restricts to tri + quad facets 
  * g4-;g4-cls G4ExtrudedSolid      # further specialization of G4TessellatedSolid

* General solution going from a bunch of planes to faces is non-trivial 

  * http://mathworld.wolfram.com/ConvexPolyhedron.html
  * http://mathworld.wolfram.com/VertexEnumeration.html

* python source that makes the planes of prism (and other convexpolyhedeon) starts by
  enumerating vertices based on input spec (angles lengths etc..) and then forms the
  planes from those which is what is currently persisted and used on GPU 

  Need in addition to persist the source verts and faces, for use by the G4 conversion

  * src vertices (nv,3)
  * tri/quad facets (nf,4)  (use -1 for missing 4th index for tri)

opticks/analytic/prism.py::

    221         xmin = -hwidth
    222         xmax =  hwidth
    223         ymin = -depth/2.
    224         ymax =  depth/2.
    225         zmin = -height/2.
    226         zmax =  height/2.
    227 
    228         v[0] = [       0,  ymin, zmax ]   # front apex
    229         v[1] = [    xmin,  ymin, zmin ]   # front left
    230         v[2] = [    xmax,  ymin, zmin ]   # front right
    231 
    232         v[3] = [       0,  ymax, zmax ]   # back apex
    233         v[4] = [    xmin,  ymax, zmin ]   # back left
    234         v[5] = [    xmax,  ymax, zmin ]   # back right
    235 
    236         p[0] = make_plane3( v[5], v[3], v[0] )
    237         p[1] = make_plane3( v[1], v[0], v[3] )
    238         p[2] = make_plane3( v[5], v[2], v[1] )
    239         p[3] = make_plane3( v[1], v[2], v[0] )
    240         p[4] = make_plane3( v[4], v[3], v[5] )



opticks/analytic/csg.py verts currently ignored::

     493     @classmethod
     494     def MakeConvexPolyhedron(cls, planes, verts, bbox, srcmeta, type_="convexpolyhedron"):
     495         """see tboolean-segment- """
     496         obj = CSG(type_)
     497         obj.planes = planes
     498         obj.param2[:3] = bbox[0]
     499         obj.param3[:3] = bbox[1]
     500         obj.meta.update(srcmeta)
     501         return obj
     502 


tboolean funcs using convexpolyhedron
--------------------------------------

* no direct use of "convexpolyhedron"

* there is direct .planes usage but with those coming from make_trapezoid, make_icosahedron

* TODO: move all to higher level intermediary python funcs, for less codepaths/duplication

::

   CSG.MakeConvexPolyhedron
   CSG.MakeTrapezoid
   CSG.MakeSegment
   CSG.MakeIcosahedron
    


::

     514     @classmethod
     515     def MakeIcosahedron(cls, scale=100.):
     516         planes, verts, bbox, srcmeta = make_icosahedron(scale=scale)
     517         return cls.MakeConvexPolyhedron(planes, verts, bbox, srcmeta, "trapezoid")
     518 






::

    (lldb) bt
    * thread #1: tid = 0x595847, 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff8cc60866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff842fd35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff8b04db1a libsystem_c.dylib`abort + 125
        frame #3: 0x00007fff8b0179bf libsystem_c.dylib`__assert_rtn + 321
        frame #4: 0x00000001043e32e6 libcfg4.dylib`CMaker::ConvertPrimitive(node=0x000000010c48d7e0) + 3462 at CMaker.cc:480
        frame #5: 0x00000001043e21c3 libcfg4.dylib`CMaker::makeSolid_r(this=0x000000010dea7c80, node=0x000000010c48d7e0) + 83 at CMaker.cc:314
        frame #6: 0x00000001043e2155 libcfg4.dylib`CMaker::makeSolid(this=0x000000010dea7c80, csg=0x000000010c48ca20) + 53 at CMaker.cc:298
        frame #7: 0x00000001043e63a2 libcfg4.dylib`CTestDetector::makeChildVolume(this=0x000000010dea7c90, csg=0x000000010c48ca20, lvn=0x000000010c4988b0, pvn=0x000000010c4a1960, mother=0x000000010deb02d0) + 354 at CTestDetector.cc:129
        frame #8: 0x00000001043e486d libcfg4.dylib`CTestDetector::makeDetector_NCSG(this=0x000000010dea7c90) + 829 at CTestDetector.cc:181
        frame #9: 0x00000001043e4509 libcfg4.dylib`CTestDetector::makeDetector(this=0x000000010dea7c90) + 57 at CTestDetector.cc:103
        frame #10: 0x00000001043e4395 libcfg4.dylib`CTestDetector::init(this=0x000000010dea7c90) + 709 at CTestDetector.cc:93
        frame #11: 0x00000001043e4078 libcfg4.dylib`CTestDetector::CTestDetector(this=0x000000010dea7c90, hub=0x0000000109641110, query=0x0000000000000000) + 248 at CTestDetector.cc:77
        frame #12: 0x00000001043e44c5 libcfg4.dylib`CTestDetector::CTestDetector(this=0x000000010dea7c90, hub=0x0000000109641110, query=0x0000000000000000) + 37 at CTestDetector.cc:78
        frame #13: 0x0000000104342903 libcfg4.dylib`CGeometry::init(this=0x000000010dea7c20) + 339 at CGeometry.cc:63
        frame #14: 0x00000001043427a0 libcfg4.dylib`CGeometry::CGeometry(this=0x000000010dea7c20, hub=0x0000000109641110) + 112 at CGeometry.cc:53
        frame #15: 0x0000000104342b0d libcfg4.dylib`CGeometry::CGeometry(this=0x000000010dea7c20, hub=0x0000000109641110) + 29 at CGeometry.cc:54
        frame #16: 0x00000001043f8e29 libcfg4.dylib`CG4::CG4(this=0x000000010c66c070, hub=0x0000000109641110) + 297 at CG4.cc:127
        frame #17: 0x00000001043f964d libcfg4.dylib`CG4::CG4(this=0x000000010c66c070, hub=0x0000000109641110) + 29 at CG4.cc:149
        frame #18: 0x00000001044f7cc3 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe510, argc=27, argv=0x00007fff5fbfe5f0) + 547 at OKG4Mgr.cc:35
        frame #19: 0x00000001044f7f53 libokg4.dylib`OKG4Mgr::OKG4Mgr(this=0x00007fff5fbfe510, argc=27, argv=0x00007fff5fbfe5f0) + 35 at OKG4Mgr.cc:41
        frame #20: 0x00000001000132ee OKG4Test`main(argc=27, argv=0x00007fff5fbfe5f0) + 1486 at OKG4Test.cc:56
        frame #21: 0x00007fff880d35fd libdyld.dylib`start + 1
    (lldb) f 8
    frame #8: 0x00000001043e486d libcfg4.dylib`CTestDetector::makeDetector_NCSG(this=0x000000010dea7c90) + 829 at CTestDetector.cc:181
       178          const GMesh* mesh = kso->getMesh();
       179          const NCSG* csg = mesh->getCSG();
       180  
    -> 181          G4VPhysicalVolume* pv = makeChildVolume( csg , lvn , pvn, mother );
       182  
       183          G4LogicalVolume* lv = pv->GetLogicalVolume() ;
       184  
    (lldb) p lvn
    (const char *) $0 = 0x000000010c4988b0 "convexpolyhedron_lv1_"
    (lldb) p pvn
    (const char *) $1 = 0x000000010c4a1960 "convexpolyhedron_pv1_"
    (lldb) p csg
    (const NCSG *) $2 = 0x000000010c48ca20
    (lldb) p *csg
    (const NCSG) $3 = {
      m_meta = 0x000000010c48cb20
      m_treedir = 0x000000010c48c9f0 "/tmp/blyth/opticks/tboolean-prism--/1"
      m_index = 1
      m_surface_epsilon = 0.00000999999974
      m_verbosity = 0
      m_usedglobally = false
      m_root = 0x000000010c48d7e0
      m_points = 0x0000000000000000
      m_uncoincide = 0x0000000000000000
      m_nudger = 0x000000010c48dba0
      m_nodes = 0x000000010c48cc10
      m_transforms = 0x000000010c48d180
      m_gtransforms = 0x000000010c48d370
      m_planes = 0x000000010c48d660
      m_nodemeta = size=0 {}
      m_num_nodes = 1
      m_num_transforms = 1
      m_num_planes = 5
      m_height = 0
      m_boundary = 0x000000010c48b250 "Vacuum///GlassSchottF2"
      m_config = 0x0000000000000000
      m_gpuoffset = {
         = (x = 0, r = 0, s = 0)
         = (y = 0, g = 0, t = 0)
         = (z = 0, b = 0, p = 0)
      }
      m_container = -1
      m_containerscale = 2
      m_tris = 0x000000010c4a1190
      m_surface_points = size=0 {}
    }
    (lldb) 

    (lldb) p *csg->m_meta
    (NParameters) $5 = {
      m_parameters = size=8 {
        [0] = (first = "src_depth", second = "300")
        [1] = (first = "src_height", second = "200")
        [2] = (first = "ctrl", second = "0")
        [3] = (first = "verbosity", second = "0")
        [4] = (first = "poly", second = "IM")
        [5] = (first = "src_angle", second = "45")
        [6] = (first = "src_type", second = "prism")
        [7] = (first = "resolution", second = "40")
      }
      m_lines = size=0 {}
    }



add py debug to the CSG::Serialize
-------------------------------------

::

    simon:analytic blyth$ tboolean-;tboolean-prism-
    args: 
    [2017-11-20 11:01:06,150] p50549 {/Users/blyth/opticks/analytic/csg.py:1003} INFO - CSG.dump name:convexpolyhedron
    co height:0 totnodes:1 

     co
    [2017-11-20 11:01:06,150] p50549 {/Users/blyth/opticks/analytic/csg.py:443} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tboolean-prism-- 
    [2017-11-20 11:01:06,150] p50549 {/Users/blyth/opticks/analytic/csg.py:710} INFO - write treemeta to /tmp/blyth/opticks/tboolean-prism--/0/meta.json {'verbosity': '0', 'resolution': '40', 'poly': 'IM', 'ctrl': '0'}  
    [2017-11-20 11:01:06,151] p50549 {/Users/blyth/opticks/analytic/csg.py:690} INFO - write nodemeta to /tmp/blyth/opticks/tboolean-prism--/0/0/nodemeta.json {'verbosity': '0', 'resolution': '40', 'idx': 0, 'poly': 'IM', 'ctrl': '0'} 
    [2017-11-20 11:01:06,157] p50549 {/Users/blyth/opticks/analytic/csg.py:710} INFO - write treemeta to /tmp/blyth/opticks/tboolean-prism--/1/meta.json {'src_depth': 300, 'src_height': 200, 'ctrl': '0', 'verbosity': '0', 'poly': 'IM', 'src_angle': 45, 'src_type': 'prism', 'resolution': '40'}  
    [2017-11-20 11:01:06,157] p50549 {/Users/blyth/opticks/analytic/csg.py:690} INFO - write nodemeta to /tmp/blyth/opticks/tboolean-prism--/1/0/nodemeta.json {'src_depth': 300, 'src_height': 200, 'ctrl': '0', 'verbosity': '0', 'poly': 'IM', 'idx': 0, 'src_angle': 45, 'src_type': 'prism', 'resolution': '40'} 
    analytic=1_csgpath=/tmp/blyth/opticks/tboolean-prism--_mode=PyCsgInBox_outerfirst=1_name=tboolean-prism--
    simon:analytic blyth$ 


review
----------

Metadata at three levels:

* list of solids
* solid, single tree
* node of the tree


::

     261 std::string NCSG::MetaPath(const char* treedir, int idx)
     262 {
     263     std::string metapath = idx == -1 ? BFile::FormPath(treedir, "meta.json") : BFile::FormPath(treedir, BStr::itoa(idx), "meta.json") ;
     264     return metapath ;
     265 }

     





