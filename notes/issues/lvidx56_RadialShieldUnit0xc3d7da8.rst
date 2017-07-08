lvidx 56 RadialShieldUnit0xc3d7da8
======================================


Viz : following move to CSG_SEGMENT python primitive (phi0,phi1,z,rmax) 
--------------------------------------------------------------------------

::

   op --dlv56 --gltf 3   
       looks reasonable : loada panels, each with 6 holes for PMTs : g4poly is kinda whacky 



Issues with two slab intersects : AVOIDED BY MOVE TO CSG_SEGMENT
-------------------------------------------------------------------

* adding slab intersects pushes tree height above limit of 7
* problem is that tree balancing rearranges to put two slabs together which doesnt work : produces empty raytrace
* a single slab intersect does work however ... because no double unbounded ?



parsurf bb is kinda mute because raytrace not working
-----------------------------------------------------------

::

       332.587               RadialShieldUnit0xc3d7da8 lvidx  56 nsp    288             intersection difference cylinder slab   nds[ 64]  4393 4394 4395 4396 4397 4398 4399 4400 4401 4402 ... 
       332.587               RadialShieldUnit0xc3d7da8 lvidx  56 nsp    288 

       amn (   1878.414     0.000  -498.500) 
       bmn (   1607.600     0.000  -498.500) 
       dmn (    270.814     0.000     0.000)   

       amx (   2262.150  1256.783   498.500) 
       bmx (   2262.150  1589.370   498.500) 
       dmx (      0.000  -332.587     0.000)

       opticks-;opticks-tbool-vi 56       





dumping the raw and blanced trees 
---------------------------------------------

* moving slab intersects to the top work when only one, but not with two 

::

    delta:cu blyth$ opticks-;opticks-tbool- 56 

    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/56/tbool56.bash
    args: 
    [2017-07-08 10:18:08,359] p6906 {/Users/blyth/opticks/analytic/csg.py:822} INFO - raw name:intersection
    in(in(di(di(di(di(di(di(di(cy,cy),cy),cy),cy),cy),cy),cy),sl),sl) height:9 totnodes:1023 

                                                                        in abcdefghij    
                                                                in abcdefghi         sl j
                                                        di abcdefgh         sl i        
                                                di abcdefg         cy h                
                                        di abcdef         cy g                        
                                di abcde         cy f                                
                        di abcd         cy e                                        
                di abc         cy d                                                
        di ab         cy c                                                        
    cy a     cy b                                                                
    [2017-07-08 10:18:08,361] p6906 {/Users/blyth/opticks/analytic/csg.py:822} INFO - optimized name:intersection_prim_balanced
    in(in(in(in(cy,!cy),in(!cy,!cy)),in(in(!cy,!cy),in(!cy,!cy))),in(sl,sl)) height:4 totnodes:31 

                                                                in abcdefghij            
                                in abcdefgh                                     in ij    
                in abcd                             in efgh                 sl i     sl j
        in ab             in cd             in ef             in gh                    
    cy a     !cy b     !cy c     !cy d     !cy e     !cy f     !cy g     !cy h                
    [2017-07-08 10:18:08,361] p6906 {/Users/blyth/opticks/analytic/csg.py:417} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/56 
    analytic=1_csgpath=/tmp/blyth/opticks/tbool/56_name=56_mode=PyCsgInBox





Raytrace tree height limit : max 7
--------------------------------------

* slab intersects add 2 levels to the tree which pushes height above limit of 7

* normally such deep trees get positivized and balanced, but thats disabled for 
  trees with slab segmenting 

::

    evaluative_csg tranOffset 0 numParts 1023 perfect tree height 9 exceeds current limit
    evaluative_csg tranOffset 0 numParts 1023 perfect tree height 9 exceeds current limit
    evaluative_csg tranOffset 0 numParts 1023 perfect tree height 9 exceeds current limit
    evaluative_csg tranOffset 0 numParts 1023 perfect tree height 9 exceeds current limit


::

     542 #define USE_TWIDDLE_POSTORDER 1
     543 
     544 static __device__
     545 void evaluative_csg( const Prim& prim, const uint4& identity )
     546 {
     547     unsigned partOffset = prim.partOffset() ;
     548     unsigned numParts   = prim.numParts() ;
     549     unsigned tranOffset = prim.tranOffset() ;
     550 
     551     unsigned height = TREE_HEIGHT(numParts) ; // 1->0, 3->1, 7->2, 15->3, 31->4 
     552 
     553 #ifdef USE_TWIDDLE_POSTORDER
     554     // bit-twiddle postorder limited to height 7, ie maximum of 0xff (255) nodes
     555     // (using 2-bytes with PACK2 would bump that to 0xffff (65535) nodes)
     556     // In any case 0xff nodes are far more than this is expected to be used with
     557     //
     558     if(height > 7)
     559     {
     560         rtPrintf("evaluative_csg tranOffset %u numParts %u perfect tree height %u exceeds current limit\n", tranOffset, numParts, height ) ;
     561         return ;
     562     }
     563 #else
     564     // pre-baked postorder limited to height 3 tree,  ie maximum of 0xf nodes
     565     // by needing to stuff the postorder sequence 0x137fe6dc25ba498ull into 64 bits 





opticks-tbool 56 
---------------------

Segment of ring with 6 holes cut by cylinders::

    078 # generated by tboolean.py : 20170707-2050 
     79 # opticks-;opticks-tbool 56 
     80 # opticks-;opticks-tbool-vi 56 
     81 
     82 
     83 a = CSG("cylinder", param = [0.000,0.000,0.000,2262.150],param1 = [-498.500,498.500,0.000,0.000])
     84 b = CSG("cylinder", param = [0.000,0.000,0.000,2259.150],param1 = [-503.485,503.485,0.000,0.000])
     85 ab = CSG("difference", left=a, right=b)
     86 
     87 c = CSG("slab", param = [0.000,1.000,0.000,0.000],param1 = [0.000,2263.150,0.000,0.000])
     88 abc = CSG("intersection", left=ab, right=c)
     89 
     90 d = CSG("slab", param = [0.703,-0.712,0.000,0.000],param1 = [0.000,2263.150,0.000,0.000])
     91 abcd = CSG("intersection", left=abc, right=d)
     92 
     93 e = CSG("cylinder", param = [0.000,0.000,0.000,106.600],param1 = [-250.000,250.000,0.000,0.000])
     94 e.transform = [[0.000,-0.127,0.992,0.000],[0.000,0.992,0.127,0.000],[-1.000,-0.000,0.000,0.000],[2242.238,287.939,250.000,1.000]]
     95 abcde = CSG("difference", left=abcd, right=e)
     96 
     97 f = CSG("cylinder", param = [0.000,0.000,0.000,106.600],param1 = [-250.000,250.000,0.000,0.000])
     98 f.transform = [[0.000,-0.380,0.925,0.000],[0.000,0.925,0.380,0.000],[-1.000,-0.000,0.000,0.000],[2091.311,858.461,250.000,1.000]]
     99 abcdef = CSG("difference", left=abcde, right=f)
    100 
    101 g = CSG("cylinder", param = [0.000,0.000,0.000,106.600],param1 = [-250.000,250.000,0.000,0.000])
    102 g.transform = [[0.000,-0.606,0.795,0.000],[0.000,0.795,0.606,0.000],[-1.000,-0.000,0.000,0.000],[1797.865,1370.481,250.000,1.000]]
    103 abcdefg = CSG("difference", left=abcdef, right=g)
    104 
    105 h = CSG("cylinder", param = [0.000,0.000,0.000,106.600],param1 = [-250.000,250.000,0.000,0.000])
    106 h.transform = [[0.000,-0.127,0.992,0.000],[0.000,0.992,0.127,0.000],[-1.000,-0.000,0.000,0.000],[2242.238,287.939,-250.000,1.000]]
    107 abcdefgh = CSG("difference", left=abcdefg, right=h)
    108 
    109 i = CSG("cylinder", param = [0.000,0.000,0.000,106.600],param1 = [-250.000,250.000,0.000,0.000])
    110 i.transform = [[0.000,-0.380,0.925,0.000],[0.000,0.925,0.380,0.000],[-1.000,-0.000,0.000,0.000],[2091.311,858.461,-250.000,1.000]]
    111 abcdefghi = CSG("difference", left=abcdefgh, right=i)
    112 
    113 j = CSG("cylinder", param = [0.000,0.000,0.000,106.600],param1 = [-250.000,250.000,0.000,0.000])
    114 j.transform = [[0.000,-0.606,0.795,0.000],[0.000,0.795,0.606,0.000],[-1.000,-0.000,0.000,0.000],[1797.865,1370.481,-250.000,1.000]]
    115 abcdefghij = CSG("difference", left=abcdefghi, right=j)
    116 
    117 
    118 
    119 obj = abcdefghij
    120 
    121 con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=args.container , poly="IM", resolution="20" )
    122 CSG.Serialize([con, obj], args.csgpath )






Checking tree balancing with slab cuts 
-----------------------------------------

* moving slab cuts to top of tree works with balancing when only one cut but not two 
  (presumably slab-slab double unbounded issue)

::

    opticks-tbool-vi 56


    123 abcdefghij_c = CSG("intersection", left=abcdefghij, right=c )
    124 abcdefghij_cd = CSG("intersection", left=abcdefghij_c, right=d )
    125 
    126 
    127 #raw = abcdefghij
    128 #raw = abcdefghij_c
    129 raw = abcdefghij_cd
    130 
    131 raw.dump("raw")
    132 
    133 maxcsgheight = 4
    134 maxcsgheight2 = 5
    135 obj = Sc.optimize_csg(raw, maxcsgheight, maxcsgheight2 ) 
    136 
    137 obj.dump("optimized")
    138 
    139 
    140 
    141 objs = [obj]



