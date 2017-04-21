NCSG Translated PMT
======================

ISSUE : ImplicitMesher fails to polygnonize thin volumes 
----------------------------------------------------------------

Thin volumes such as those produced by CSG differences of concentric zspheres
reveal limitation of current polygonization techniques. 

* observations on PMT polygonization in tests/tboolean_pmt.py 
  
* increasing resolution is not a solution, as that runs into 
  memory issues and yields too many tris 

* need another approach ? 


FIXED : ImplicitMesher fails to find surface
--------------------------------------------------

Argh cathode thickness is 0.05, from 127.95 to 128.00

::


    ImplicitMesherF::ImplicitMesherF m_gfunc: GenericFunctionBase::desc verbosity 2 epsilon 0.01 numSeeds 0 numSeedDirs 0
    MakePolygonizer verbosity 2 center.x 0.0000 center.y 0.0000 center.z 64.6400 cubesize 0.4309 convergence 10 bounds  low (  -301 -301 -151 )  high (  301 301 151 ) 
    2017-04-21 12:16:07.142 INFO  [1535407] [NImplicitMesher::addManualSeeds@118] NImplicitMesher::addManualSeeds nseed 6 sxyz(0 0 127.9)  dxyz(0 0 1) 
    ...

    ::next_seed_point verbosity 2 num_seed_points 1
    ::next_seed_point dumping seedpoints verbosity 2 seedpoint_.p[0] 0.0000 seedpoint_.p[1] 0.0000 seedpoint_.p[2] 127.9000 seedpoint_.d[0] 0.0000 seedpoint_.d[1] 0.0000 seedpoint_.d[2] 1.0000
    ::next_seed_point verbosity 2 point.x 0.0000 point.y 0.0000 point.z 127.9000 xdir 0.0000 ydir 0.0000 zdir 1.0000 i 0 j 0 k 147
    GenericFunctionF::Value     1 ( 0.0000  0.0000 127.9000 ) ->  0.0500
    GenericFunctionF::Value     2 ( 0.0000  0.0000 127.9000 ) ->  0.0500
    GenericFunctionF::Value     3 ( 0.0000  0.0000 128.2918 ) ->  0.2918
    GenericFunctionF::Value     4 ( 0.0000  0.0000 128.6835 ) ->  0.6835
    GenericFunctionF::Value     5 ( 0.0000  0.0000 129.0753 ) ->  1.0753
    GenericFunctionF::Value     6 ( 0.0000  0.0000 129.4670 ) ->  1.4670
    GenericFunctionF::Value     7 ( 0.0000  0.0000 129.8588 ) ->  1.8588
    GenericFunctionF::Value     8 ( 0.0000  0.0000 130.2505 ) ->  2.2505
    GenericFunctionF::Value     9 ( 0.0000  0.0000 130.6423 ) ->  2.6423


Step size using resolution 500::

    In [1]: 128.6835 - 128.2918
    Out[1]: 0.39170000000001437

    In [2]: 0.4309/1.1    # cubesize/1.1
    Out[2]: 0.3917272727272727


With resolutions 500,1000,4000::

    MakePolygonizer verbosity 2 center.x 0.0000 center.y 0.0000 center.z 64.6400 cubesize 0.4309 convergence 10 bounds  low (  -301 -301 -151 )  high (  301 301 151 )
    MakePolygonizer verbosity 2 center.x 0.0000 center.y 0.0000 center.z 64.6400 cubesize 0.2155 convergence 10 bounds  low (  -601 -601 -301 )  high (  601 601 301 ) 
    MakePolygonizer verbosity 2 center.x 0.0000 center.y 0.0000 center.z 64.6400 cubesize 0.0539 convergence 10 bounds  low (  -2401 -2401 -1201 )  high (  2401 2401 1201 ) 


Find the surface at resolution 4000, but visualizing see just a tiny cap, continuation failed to follow::

    enericFunctionBase::GetSeedPoints  verbosity 2 nBufIndex 0 nBufSize 6144 nseed 1 nsdir 1
     i 0 seed vec3(0.000000, 0.000000, 127.900002) sdir vec3(0.000000, 0.000000, 1.000000)
    GenericFunctionBase::GetSeedPoints verbosity 2 nBufIndex 6
    GenericFunctionBase::dumpSeedBuffer size 6

         0.0000     0.0000   127.9000     0.0000     0.0000     1.0000
    ::next_seed_point verbosity 2 num_seed_points 1
    ::next_seed_point dumping seedpoints verbosity 2 seedpoint_.p[0] 0.0000 seedpoint_.p[1] 0.0000 seedpoint_.p[2] 127.9000 seedpoint_.d[0] 0.0000 seedpoint_.d[1] 0.0000 seedpoint_.d[2] 1.0000
    ::next_seed_point verbosity 2 point.x 0.0000 point.y 0.0000 point.z 127.9000 xdir 0.0000 ydir 0.0000 zdir 1.0000 i 0 j 0 k 1174
    GenericFunctionF::Value     1 ( 0.0000  0.0000 127.9000 ) ->  0.0500
    GenericFunctionF::Value     2 ( 0.0000  0.0000 127.9000 ) ->  0.0500
    GenericFunctionF::Value     3 ( 0.0000  0.0000 127.9490 ) ->  0.0010
    GenericFunctionF::Value     4 ( 0.0000  0.0000 127.9979 ) -> -0.0021
    find_in_out sx 0.0000 sy 0.0000 sz 127.9000 xdir 0.0000 ydir 0.0000 zdir 1.0000 in ( 0.0000 0.0000 127.9490 )  out ( 0.0000 0.0000 127.9979 )  inValue 0.0010 outValue -0.0021 dist 0.1469 delta 0.0490 step 3
    pg_polygonize START next_seed 0
    GenericFunctionF::Value     5 ( 0.0000  0.0000 127.9652 ) -> -0.0152
    GenericFunctionF::Value     6 ( 0.0000  0.0000 127.9500 ) -> -0.0000
    GenericFunctionF::Value     7 ( 0.0000  0.0000 127.9500 ) -> -0.0000
    GenericFunctionF::Value     8 ( 0.0000  0.0000 127.9500 ) -> -0.0000
    GenericFunctionF::Value     9 ( 0.0000  0.0000 127.9500 ) -> -0.0000


Pushing to resolution 8000 runs into memory limit in ImplicitMesher::


    (lldb) bt
    * thread #1: tid = 0x178c88, 0x00007fff96061866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff96061866 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff8d6fe35c libsystem_pthread.dylib`pthread_kill + 92
        frame #2: 0x00007fff9444eb1a libsystem_c.dylib`abort + 125
        frame #3: 0x00000001054d763c libImplicitMesher.dylib`dv_allocate_element(vec=0x0000000109f10fe0) + 108 at ImplicitPolygonizer.cpp:212
        frame #4: 0x00000001054d7b42 libImplicitMesher.dylib`setcorner(p=0x000000010598b800, i=8, j=-1396, k=1268) + 194 at ImplicitPolygonizer.cpp:784
        frame #5: 0x00000001054d825b libImplicitMesher.dylib`testface(i=7, j=-1396, k=1267, old=0x00007fff5fbfc468, face=1, c1=4, c2=5, c3=6, c4=7, p=0x000000010598b800) + 907 at ImplicitPolygonizer.cpp:758
        frame #6: 0x00000001054d5c61 libImplicitMesher.dylib`pg_polygonize(p=0x000000010598b800) + 913 at ImplicitPolygonizer.cpp:682
        frame #7: 0x00000001054d555c libImplicitMesher.dylib`ImplicitPolygonizer::Polygonize(this=0x0000000109f08e50) + 332 at ImplicitPolygonizer.cpp:102
        frame #8: 0x00000001054df679 libImplicitMesher.dylib`ImplicitMesherBase::polygonize(this=0x0000000109f08c80) + 25 at ImplicitMesherBase.cpp:29
        frame #9: 0x0000000100880b57 libNPY.dylib`NImplicitMesher::operator(this=0x00007fff5fbfc8d0)() + 423 at NImplicitMesher.cpp:183
        frame #10: 0x000000010087e044 libNPY.dylib`NPolygonizer::implicitMesher(this=0x00007fff5fbfcdf0) + 324 at NPolygonizer.cpp:145
        frame #11: 0x000000010087db5a libNPY.dylib`NPolygonizer::polygonize(this=0x00007fff5fbfcdf0) + 714 at NPolygonizer.cpp:71
        frame #12: 0x0000000101e1aa88 libGGeo.dylib`GMaker::makeFromCSG(this=0x0000000108856b40, csg=0x0000000109f01650) + 344 at GMaker.cc:97
        frame #13: 0x0000000101e106f4 libGGeo.dylib`GGeoTest::loadCSG(this=0x0000000108857bf0, csgpath=0x0000000108858a50, solids=0x00007fff5fbfd660) + 1300 at GGeoTest.cc:225
        frame #14: 0x0000000101e0f876 libGGeo.dylib`GGeoTest::create(this=0x0000000108857bf0) + 470 at GGeoTest.cc:116
        frame #15: 0x0000000101e0f5fd libGGeo.dylib`GGeoTest::modifyGeometry(this=0x0000000108857bf0) + 157 at GGeoTest.cc:80
        frame #16: 0x0000000101e34f82 libGGeo.dylib`GGeo::modifyGeometry(this=0x0000000105622ae0, config=0x0000000108859090) + 658 at GGeo.cc:761
        frame #17: 0x0000000101f5e9e4 libOpticksGeometry.dylib`OpticksGeometry::modifyGeometry(this=0x0000000105622990) + 868 at OpticksGeometry.cc:263
        frame #18: 0x0000000101f5df2c libOpticksGeometry.dylib`OpticksGeometry::loadGeometry(this=0x0000000105622990) + 572 at OpticksGeometry.cc:200
        frame #19: 0x0000000101f61fa9 libOpticksGeometry.dylib`OpticksHub::loadGeometry(this=0x000000010570b8c0) + 409 at OpticksHub.cc:243
        frame #20: 0x0000000101f6114d libOpticksGeometry.dylib`OpticksHub::init(this=0x000000010570b8c0) + 77 at OpticksHub.cc:94
        frame #21: 0x0000000101f61080 libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x000000010570b8c0, ok=0x0000000105621c90) + 416 at OpticksHub.cc:81
        frame #22: 0x0000000101f6121d libOpticksGeometry.dylib`OpticksHub::OpticksHub(this=0x000000010570b8c0, ok=0x0000000105621c90) + 29 at OpticksHub.cc:83
        frame #23: 0x00000001038cf266 libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe658, argc=21, argv=0x00007fff5fbfe738, argforced=0x0000000000000000) + 262 at OKMgr.cc:46
        frame #24: 0x00000001038cf69b libOK.dylib`OKMgr::OKMgr(this=0x00007fff5fbfe658, argc=21, argv=0x00007fff5fbfe738, argforced=0x0000000000000000) + 43 at OKMgr.cc:49
        frame #25: 0x000000010000a9ed OKTest`main(argc=21, argv=0x00007fff5fbfe738) + 1373 at OKTest.cc:60
        frame #26: 0x00007fff914d45fd libdyld.dylib`start + 1
        frame #27: 0x00007fff914d45fd libdyld.dylib`start + 1
    (lldb) f 3
    frame #3: 0x00000001054d763c libImplicitMesher.dylib`dv_allocate_element(vec=0x0000000109f10fe0) + 108 at ImplicitPolygonizer.cpp:212
       209          if (vec->segment_ptrs[vec->nCurSegment+1] == NULL) {
       210  
       211              if (vec->nCurSegment+1 >= 64)       // HARDCODED LIMIT FOR NOW...
    -> 212                  abort();
       213              vec->segment_ptrs[vec->nCurSegment+1] = 
       214                  (unsigned char *)malloc( vec->nElemSize * vec->nSegmentSize );
       215          }
    (lldb) 


With resolution 5000::

    (lldb) f 3
    frame #3: 0x00000001054d763c libImplicitMesher.dylib`dv_allocate_element(vec=0x00000001088ea990) + 108 at ImplicitPolygonizer.cpp:212
       209          if (vec->segment_ptrs[vec->nCurSegment+1] == NULL) {
       210  
       211              if (vec->nCurSegment+1 >= 64)       // HARDCODED LIMIT FOR NOW...
    -> 212                  abort();
       213              vec->segment_ptrs[vec->nCurSegment+1] = 
       214                  (unsigned char *)malloc( vec->nElemSize * vec->nSegmentSize );
       215          }
    (lldb) p vec->nCurSegment+1
    (unsigned int) $0 = 64
    (lldb) 





::

     517     float cubesize = aabb_avgcubesize(boundingbox, grid_resolution);

     263 static float aabb_avgcubesize(aabb * box, int resolution)
     264 {
     265   float a,b,c;
     266   a = (box->high[0] - box->low[0]) / (float)resolution;
     267   b = (box->high[1] - box->low[1]) / (float)resolution;
     268   c = (box->high[2] - box->low[2]) / (float)resolution;
     269   return (a+b+c)/3.0f;
     270 }



ImplicitPolygonizer.cpp is stepping right over the solid::

     835   dist = 0.0;
     836   delta = p->size / 1.1f;
     837   
     838   *inValue = p->wrapper->Function()->ValueT(sx, sy, sz);
     839   *outValue = *inValue ;
     840   
     841   in->x = out->x = sx;
     842   in->y = out->y = sy;
     843   in->z = out->z = sz;
     844   
     845   int step = 0 ;
     846   while(MC_SIGN(*outValue) == MC_SIGN(*inValue) && step < 100)
     847   {
     848       step++ ;
     849       
     850       *inValue = *outValue;
     851       
     852       in->x = out->x ;
     853       in->y = out->y ;
     854       in->z = out->z ;
     855       
     856       out->x = sx + (xdir * dist);
     857       out->y = sy + (ydir * dist);
     858       out->z = sz + (zdir * dist);
     859       
     860       *outValue = p->wrapper->Function()->ValueT(out->x, out->y, out->z);
     861 
     862       dist += delta;
     863   }






FIXED: manual seeding headed in wrong direction
--------------------------------------------------

* trivial NImplicitMesher bug in setting seeds

::

    2017-04-20 18:16:35.109 INFO  [1417293] [GPropertyLib::close@384] GPropertyLib::close type GSurfaceLib buf 48,2,39,4
    2017-04-20 18:16:35.109 FATAL [1417293] [*GParts::make@163] GParts::make NCSG  treedir /tmp/blyth/opticks/tboolean-difference-zsphere--/0 node_sh 1,4,4 tran_sh 0,3,4,4 spec Rock//perfectAbsorbSurface/Vacuum type box
    2017-04-20 18:16:35.109 INFO  [1417293] [*GMaker::makeFromCSG@91] GMaker::makeFromCSG index 1
    NPolygonizer::NPolygonizer(meta)
          verbosity :               3
              seeds :     0,0,0,1,0,0
         resolution :              50
               poly :              IM
               ctrl :               0
    2017-04-20 18:16:35.109 INFO  [1417293] [*NPolygonizer::polygonize@51] NPolygonizer::polygonize treedir /tmp/blyth/opticks/tboolean-difference-zsphere--/1 poly IM verbosity 3 index 1
    2017-04-20 18:16:35.109 FATAL [1417293] [NImplicitMesher::init@64] NImplicitMesher::init ImplicitMesherF ctor  verbosity 3
    ImplicitMesherF::ImplicitMesherF m_gfunc: GenericFunctionBase::desc verbosity 3 epsilon 0.01 numSeeds 0 numSeedDirs 0
    MakePolygonizer verbosity 3 center.x 0 center.y 0 center.z 0 cubesize 14.8268 convergence 10 bounds  low (  -35 -35 -7 )  high (  35 35 7 ) 
    2017-04-20 18:16:35.110 INFO  [1417293] [NImplicitMesher::addManualSeeds@84] NImplicitMesher::addManualSeeds
    2017-04-20 18:16:35.110 INFO  [1417293] [NImplicitMesher::addManualSeeds@103] NImplicitMesher::addManualSeeds nseed 6 sxyz(0 0 0)  dxyz(1 0 0) 
    2017-04-20 18:16:35.110 INFO  [1417293] [NImplicitMesher::addCenterSeeds@120] NImplicitMesher::addCenterSeeds
    2017-04-20 18:16:35.110 INFO  [1417293] [nnode::collect_prim_centers@296] nnode::collect_prim_centers verbosity 3 nprim 2
    2017-04-20 18:16:35.110 INFO  [1417293] [nnode::collect_prim_centers@306] nnode::collect_prim_centers i 0 type 7 name zsphere
    2017-04-20 18:16:35.110 INFO  [1417293] [nnode::collect_prim_centers@306] nnode::collect_prim_centers i 1 type 7 name zsphere
    2017-04-20 18:16:35.110 INFO  [1417293] [NImplicitMesher::addCenterSeeds@130] NImplicitMesher::addCenterSeeds ncenters 2 ndirs 2
      0 position {    0.0000    0.0000    0.0000} direction {    0.0000    0.0000    1.0000}
      1 position {    0.0000    0.0000    0.0000} direction {    0.0000    0.0000    1.0000}
    2017-04-20 18:16:35.110 INFO  [1417293] [*NImplicitMesher::operator@150] NImplicitMesher::operator() polygonizing START verbosity 3 bb  mi  (-505.00 -505.00 -102.01)  mx  ( 505.00  505.00  102.01)  
    ImplicitPolygonizer::Polygonize START
    ImplicitPolygonizer::Polygonize reset_polygonizer verbosity: 3
    GenericFunctionBase::GetSeedPoints  nBufIndex 0 nBufSize 6144 nseed 3 nsdir 3
    ::next_seed_point verbosity 3 num_seed_points 3
    ::next_seed_point verbosity 3 point.x 0.0000 point.y 0.0000 point.z 0.0000 xdir 1.0000 ydir 0.0000 zdir 1.0000 i 0 j 0 k 0
    GenericFunctionF::Value     1 ( 0.0000  0.0000  0.0000 ) -> 101.0000
    GenericFunctionF::Value     2 ( 0.0000  0.0000  0.0000 ) -> 101.0000
    GenericFunctionF::Value     3 (13.4789  0.0000 13.4789 ) -> 87.5211
    GenericFunctionF::Value     4 (26.9578  0.0000 26.9578 ) -> 74.0422
    GenericFunctionF::Value     5 (40.4367  0.0000 40.4367 ) -> 60.5633
    GenericFunctionF::Value     6 (53.9156  0.0000 53.9156 ) -> 47.0844
    GenericFunctionF::Value     7 (67.3945  0.0000 67.3945 ) -> 33.6055
    GenericFunctionF::Value     8 (80.8735  0.0000 80.8735 ) -> 20.1265
    GenericFunctionF::Value     9 (94.3524  0.0000 94.3524 ) ->  6.6476
    GenericFunctionF::Value    10 (107.8313  0.0000 107.8313 ) ->  7.8313




FIXED : Missing physvol placement transforms
------------------------------------------------

Doing all 5 solids of pmt together with tboolean-pmt
shows missing transforms dfor BOTTOM and presumably DYNODE too.



::

    081   <!-- The PMT vacuum -->
     82   <logvol name="lvPmtHemiVacuum" material="Vacuum">
     83     <union name="pmt-hemi-vac">
     84       <intersection name="pmt-hemi-bulb-vac">
     85     <sphere name="pmt-hemi-face-vac"
     86         outerRadius="PmtHemiFaceROCvac"/>
     87 
     88     <sphere name="pmt-hemi-top-vac"
     89         outerRadius="PmtHemiBellyROCvac"/>
     90     <posXYZ z="PmtHemiFaceOff-PmtHemiBellyOff"/>
     91 
     92     <sphere name="pmt-hemi-bot-vac"
     93         outerRadius="PmtHemiBellyROCvac"/>
     94     <posXYZ z="PmtHemiFaceOff+PmtHemiBellyOff"/>
     95 
     96       </intersection>
     97       <tubs name="pmt-hemi-base-vac"
     98         sizeZ="PmtHemiGlassBaseLength-PmtHemiGlassThickness"
     99         outerRadius="PmtHemiGlassBaseRadius-PmtHemiGlassThickness"/>
    100       <posXYZ z="-0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness"/>
    101     </union>
    102 
    103     <physvol name="pvPmtHemiCathode" 
    104          logvol="/dd/Geometry/PMT/lvPmtHemiCathode"/>
    105 
    106     <physvol name="pvPmtHemiBottom"
    107          logvol="/dd/Geometry/PMT/lvPmtHemiBottom">
    108       <posXYZ z="PmtHemiFaceOff+PmtHemiBellyOff"/>
    109     </physvol>
    110 
    111     <physvol name="pvPmtHemiDynode"
    112          logvol="/dd/Geometry/PMT/lvPmtHemiDynode">
    113       <posXYZ z="-0.5*PmtHemiGlassBaseLength+PmtHemiGlassThickness"/>
    114     </physvol>
    115   </logvol>



ddbase.py passes posXYZ placement from pv to referenced lv (within children, is it being used by translator?)::

    142     def children(self):
    143         """
    144         Defines the nature of the tree. 
    145 
    146         * for Physvol returns single item list containing the referenced Logvol
    147         * for Logvol returns list of all contained Physvol
    148         * otherwise returns empty list 
    149 
    150         NB bits of geometry of a Logvol are not regarded as children, 
    151         but rather are constitutent to it.
    152         """
    153         if type(self) is Physvol:
    154             posXYZ = self.find_("./posXYZ")
    155             lvn = self.logvolref.split("/")[-1]
    156             lv = self.g.logvol_(lvn)
    157             lv.posXYZ = posXYZ    # propagating the 
    158             if posXYZ is not None:
    159                 log.info("children... %s passing pv posXYZ to lv %s  " % (self.name, repr(lv)))
    160             return [lv]
    161         
    162         elif type(self) is Logvol:
    163             pvs = self.findall_("./physvol")
    164             return pvs
    165         else:
    166             return []
    167         pass


After follow that up with setting of node transforms on the lvnodes, 
the bottom appears to be in correct place, but dynode is poking thru the cathode...
this is probably the lack of transform offsets::

     15 class NCSGConverter(object):
     16     """
     17     Translate single volume detdesc primitives and CSG operations
     18     into an NCSG style node tree
     19     """
     20     @classmethod
     21     def ConvertLV(cls, lv ):
     22         """
     23         :param lv: Elem
     24         :return cn: CSG node instance 
     25         """
     26         lvgeom = lv.geometry()
     27         assert len(lvgeom) == 1, "expecting single CSG operator or primitive Elem within LV"
     28 
     29         cn = cls.convert(lvgeom[0])
     30 
     31         if lv.posXYZ is not None:
     32             assert cn.transform is None
     33             translate  = "%s,%s,%s" % (lv.xyz[0], lv.xyz[1], lv.xyz[2])
     34             cn.translate = translate
     35             log.info("TranslateLV posXYZ:%r -> translate %s  " % (lv.posXYZ, translate) )
     36         pass
     37         return cn
     38 




FIXED : Cathode Inner or Outer
-----------------------------------

* can see from front but disappearing from back 
* observe wierdness in t_min clipping, 

* testing with tboolean-zsphere see the same wierdness, 
  its the missing cap handling 

* intersecting with a zslab works, but then you get a cap 

* used a flag to switch off the cap, but now getting sliver artifact and 
  spurious intersects

* actually switching off the caps prevents slab intersection from working, 
  get nothing with tboolean-sphere-slab ... cannot selectively have the intersect work for doing 
  the intersection chop and not work for giving an open cap...

  * cannot use infinite slab intersection without enabling the caps

  * so cannot use slab intersection and have open caps 

  * hmm, means must implement cap handling similar to cylinder in zsphere


Testing with tboolean-pmt with a kludge to just 
return the inner or outer in ncsgtranslator.py::


    182         cn.param[0] = en.xyz[0]
    183         cn.param[1] = en.xyz[1]
    184         cn.param[2] = en.xyz[2]
    185         cn.param[3] = radius
    186 
    187         if has_inner:
    188             #ret = CSG("difference", left=cn, right=inner )
    189             ret = inner
    190         else:
    191             ret = cn
    192         pass
    193         return ret
    194 






::

    2017-04-18 18:43:57.920 INFO  [962828] [GParts::dump@857] GParts::dump ni 4
         0.0000      0.0000      0.0000   1000.0000 
         0.0000      0.0000     123 <-bnd        0 <-INDEX    bn Rock//perfectAbsorbSurface/Vacuum 
         0.0000      0.0000      0.0000           6 (box) TYPECODE 
         0.0000      0.0000      0.0000           0 (nodeIndex) 

         0.0000      0.0000      0.0000      0.0000 
         0.0000      0.0000     124 <-bnd        1 <-INDEX    bn Vacuum///GlassSchottF2 
         0.0000      0.0000      0.0000           1 (union) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

         0.0000      0.0000      0.0000    127.9500 
        97.2867    127.9500     124 <-bnd        2 <-INDEX    bn Vacuum///GlassSchottF2 
         0.0000      0.0000      0.0000           7 (zsphere) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

         0.0000      0.0000     43.0000     98.9500 
        12.9934     55.7343     124 <-bnd        3 <-INDEX    bn Vacuum///GlassSchottF2 
        0.0000      0.0000      0.0000           7 (zsphere) TYPECODE 
         0.0000      0.0000      0.0000           1 (nodeIndex) 

