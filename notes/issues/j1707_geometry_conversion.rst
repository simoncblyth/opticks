J1707 Geometry Conversion
=============================

op --j1707 --pygdml
-----------------------

* NB direct gdml2gltf.py assumes DYB, to setup envvars for j1707 version of JUNO use op machinery 


::

    simon:opticks blyth$ op --j1707 --pygdml
  

GDML Torus
~~~~~~~~~~~~~`

After fixup torus gdml into CSG fix this error::
  

      left = self.first.as_ncsg()
      File "/Users/blyth/opticks/analytic/gdml.py", line 187, in as_ncsg
        right = self.second.as_ncsg()
      File "/Users/blyth/opticks/analytic/gdml.py", line 189, in as_ncsg
        assert right, "right fail as_ncsg for second : %r self: %r " % (self.second, self)
      AssertionError: right fail as_ncsg for second : [56] Torus PMT_20inch_pmt_solid_2_Torus0x1817660 mm rmin 0.0 rmax 47.009  x 0.0 y 0.0 z 0.0   self: [57] Subtraction PMT_20inch_pmt_solid_part20x1817730  
         l:[55] Tube PMT_20inch_pmt_solid_2_Tube0x1817550 mm rmin 0.0 rmax 77.9765319749  x 0.0 y 0.0 z 42.9924690463  
         r:[56] Torus PMT_20inch_pmt_solid_2_Torus0x1817660 mm rmin 0.0 rmax 47.009  x 0.0 y 0.0 z 0.0   
      /Users/blyth/opticks/bin/op.sh RC 1


Treebuiler Warnings
~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:analytic blyth$ op --j1707 --pygdml
    === op-cmdline-binary-match : finds 1st argument with associated binary : --pygdml
    ubin /Users/blyth/opticks/bin/gdml2gltf.py cfm --pygdml cmdline --j1707 --pygdml
    8 -rwxr-xr-x  1 blyth  staff  459 Jun 24 10:09 /Users/blyth/opticks/bin/gdml2gltf.py
    proceeding : python /Users/blyth/opticks/bin/gdml2gltf.py --j1707 --pygdml
    args: /Users/blyth/opticks/bin/gdml2gltf.py --j1707 --pygdml
    [2017-08-02 21:02:08,490] p52930 {/Users/blyth/opticks/analytic/sc.py:457} INFO - start GDML parse
    [2017-08-02 21:02:08,490] p52930 {/Users/blyth/opticks/analytic/gdml.py:930} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml 
    [2017-08-02 21:02:08,801] p52930 {/Users/blyth/opticks/analytic/gdml.py:944} INFO - wrapping gdml element  
    [2017-08-02 21:02:08,964] p52930 {/Users/blyth/opticks/analytic/gdml.py:992} INFO - vv 46 vvs 35 
    [2017-08-02 21:02:08,974] p52930 {/Users/blyth/opticks/analytic/sc.py:460} INFO - start treeify
    [2017-08-02 21:02:29,555] p52930 {/Users/blyth/opticks/analytic/sc.py:463} INFO - start apply_selection
    [2017-08-02 21:02:30,333] p52930 {/Users/blyth/opticks/analytic/treebase.py:504} INFO - apply_selection OpticksQuery range:1:50000 range [1, 50000] index 0 depth 0   Node.selected_count 49999 
    [2017-08-02 21:02:30,333] p52930 {/Users/blyth/opticks/ana/OpticksQuery.py:47} INFO - count 49999 matches expectation 
    [2017-08-02 21:02:30,333] p52930 {/Users/blyth/opticks/analytic/sc.py:466} INFO - start Sc.ctor
    [2017-08-02 21:02:30,333] p52930 {/Users/blyth/opticks/analytic/sc.py:472} INFO - start Sc
    [2017-08-02 21:02:30,333] p52930 {/Users/blyth/opticks/analytic/sc.py:357} INFO - add_tree_gdml START maxdepth:0 maxcsgheight:3 nodesCount:    0
    [2017-08-02 21:02:30,333] p52930 {/Users/blyth/opticks/analytic/treebase.py:34} WARNING - returning DummyTopPV placeholder transform
    [2017-08-02 21:02:37,496] p52930 {/Users/blyth/opticks/analytic/csg.py:548} INFO - MakeEllipsoid axyz array([ 264.,  264.,  194.], dtype=float32) scale array([ 1.3608,  1.3608,  1.    ], dtype=float32) 
    [2017-08-02 21:02:37,497] p52930 {/Users/blyth/opticks/analytic/csg.py:548} INFO - MakeEllipsoid axyz array([ 256.,  256.,  186.], dtype=float32) scale array([ 1.3763,  1.3763,  1.    ], dtype=float32) 
    [2017-08-02 21:02:37,498] p52930 {/Users/blyth/opticks/analytic/csg.py:548} INFO - MakeEllipsoid axyz array([ 254.001,  254.001,  184.001], dtype=float32) scale array([ 1.3804,  1.3804,  1.    ], dtype=float32) 
    [2017-08-02 21:02:37,499] p52930 {/Users/blyth/opticks/analytic/csg.py:548} INFO - MakeEllipsoid axyz array([ 254.,  254.,  184.], dtype=float32) scale array([ 1.3804,  1.3804,  1.    ], dtype=float32) 
    [2017-08-02 21:02:37,501] p52930 {/Users/blyth/opticks/analytic/csg.py:548} INFO - MakeEllipsoid axyz array([ 249.,  249.,  179.], dtype=float32) scale array([ 1.3911,  1.3911,  1.    ], dtype=float32) 
    [2017-08-02 21:02:37,502] p52930 {/Users/blyth/opticks/analytic/treebuilder.py:34} WARNING - balancing trees of this structure not implemented
    [2017-08-02 21:02:37,503] p52930 {/Users/blyth/opticks/analytic/csg.py:548} INFO - MakeEllipsoid axyz array([ 249.,  249.,  179.], dtype=float32) scale array([ 1.3911,  1.3911,  1.    ], dtype=float32) 
    [2017-08-02 21:02:37,504] p52930 {/Users/blyth/opticks/analytic/treebuilder.py:34} WARNING - balancing trees of this structure not implemented




Cone problem ? Nope : polycone cy with swapped zplane : FIXED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    [2017-08-03 11:14:03,396] p54997 {/Users/blyth/opticks/analytic/treebuilder.py:34} WARNING - balancing trees of this structure not implemented
    [2017-08-03 11:15:56,899] p54997 {/Users/blyth/opticks/analytic/gdml.py:749} CRITICAL - Polycone.as_ncsg failed ValueError : ValueError('Polycone bad z-order expect z2>z1 : but z1 -15.8745078664 z2 -75.8755078664 ',) 
    Traceback (most recent call last):
      File "/Users/blyth/opticks/bin/gdml2gltf.py", line 28, in <module>
        sc = gdml2gltf_main( args )
      File "/Users/blyth/opticks/analytic/sc.py", line 479, in gdml2gltf_main
        tg = sc.add_tree_gdml( tree.root, maxdepth=0)
      File "/Users/blyth/opticks/analytic/sc.py", line 364, in add_tree_gdml
        tg = build_r(target)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 348, in build_r
        nd = self.add_node_gdml(node, depth)
      File "/Users/blyth/opticks/analytic/sc.py", line 245, in add_node_gdml
        csg = self.translate_lv( node.lv, self.maxcsgheight )
      File "/Users/blyth/opticks/analytic/sc.py", line 285, in translate_lv
        rawcsg = solid.as_ncsg()
      File "/Users/blyth/opticks/analytic/gdml.py", line 188, in as_ncsg
        assert left, " left fail as_ncsg for first : %r self: %r " % (self.first, self)
    AssertionError:  left fail as_ncsg for first : [68] PMT_3inch_pmt_solid_cyl0x1c9da50  2 z:            [-75.8755078663876, -15.8745078663875] rmax:                           [30.001] rmin:               [0.0]  self: [70] Union PMT_3inch_pmt_solid0x1c9e270  
         l:[68] PMT_3inch_pmt_solid_cyl0x1c9da50  2 z:            [-75.8755078663876, -15.8745078663875] rmax:                           [30.001] rmin:               [0.0] 
         r:[69] Sphere PMT_3inch_pmt_solid_sph0x1c9e130 mm rmin 0.0 rmax 40.001  x 0.0 y 0.0 z 0.0   
    /Users/blyth/opticks/bin/op.sh RC 1
    simon:analytic blyth$ 




Zplanes are mis-ordered, added fix to swap them.

rg::

   421     <polycone aunit="deg" deltaphi="360" lunit="mm" name="PMT_3inch_pmt_solid_cyl0x1c9da50" startphi="0">
   422       <zplane rmax="30.001" rmin="0" z="-15.8745078663875"/>
   423       <zplane rmax="30.001" rmin="0" z="-75.8755078663876"/>
   424     </polycone>
   425     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="PMT_3inch_pmt_solid_sph0x1c9e130" rmax="40.001" rmin="0" startphi="0" starttheta="0"/>
   426     <union name="PMT_3inch_pmt_solid0x1c9e270">
   427       <first ref="PMT_3inch_pmt_solid_cyl0x1c9da50"/>
   428       <second ref="PMT_3inch_pmt_solid_sph0x1c9e130"/>
   429     </union>




The unbalance-able tree probably PMT::

    [2017-08-03 11:32:08,031] p55418 {/Users/blyth/opticks/analytic/treebuilder.py:34} WARNING - balancing trees of this structure not implemented, tree in(un(un(zs,in(cy,!to)),cy),cy) height:4 totnodes:31  
    [2017-08-03 11:32:08,032] p55418 {/Users/blyth/opticks/analytic/csg.py:552} INFO - MakeEllipsoid axyz array([ 249.,  249.,  179.], dtype=float32) scale array([ 1.3911,  1.3911,  1.    ], dtype=float32) 
    [2017-08-03 11:32:08,033] p55418 {/Users/blyth/opticks/analytic/treebuilder.py:34} WARNING - balancing trees of this structure not implemented, tree in(un(un(zs,in(cy,!to)),cy),!cy) height:4 totnodes:31  


     in(un(un(zs,in(cy,!to)),cy),!cy)



Overheight::


    un(in(in(in(cy,!cy),!bo),!cy),in(    in(bo,!bo),!cy)) height:4 totnodes:31  

    un(un(in(in(in(bo,!cy),!cy),!cy),    bo),bo) height:5 totnodes:63  



    [2017-08-03 11:34:14,577] p55418 {/Users/blyth/opticks/analytic/gdml.py:714} WARNING - Polycone swap misordered pair of zplanes for PMT_3inch_pmt_solid_cyl0x1c9da50 
    [2017-08-03 11:34:14,578] p55418 {/Users/blyth/opticks/analytic/csg.py:552} INFO - MakeEllipsoid axyz array([ 40.,  40.,  24.], dtype=float32) scale array([ 1.6667,  1.6667,  1.    ], dtype=float32) 
    [2017-08-03 11:34:14,579] p55418 {/Users/blyth/opticks/analytic/csg.py:552} INFO - MakeEllipsoid axyz array([ 38.,  38.,  22.], dtype=float32) scale array([ 1.7273,  1.7273,  1.    ], dtype=float32) 
    [2017-08-03 11:34:14,580] p55418 {/Users/blyth/opticks/analytic/csg.py:552} INFO - MakeEllipsoid axyz array([ 38.,  38.,  22.], dtype=float32) scale array([ 1.7273,  1.7273,  1.    ], dtype=float32) 
    [2017-08-03 11:34:14,581] p55418 {/Users/blyth/opticks/analytic/gdml.py:714} WARNING - Polycone swap misordered pair of zplanes for PMT_3inch_cntr_solid0x1c9e640 
    [2017-08-03 11:38:26,914] p55418 {/Users/blyth/opticks/analytic/treebuilder.py:34} WARNING - balancing trees of this structure not implemented, tree un(in(in(in(cy,!cy),!bo),!cy),in(in(bo,!bo),!cy)) height:4 totnodes:31  
    [2017-08-03 11:38:26,916] p55418 {/Users/blyth/opticks/analytic/treebuilder.py:34} WARNING - balancing trees of this structure not implemented, tree un(un(in(in(in(bo,!cy),!cy),!cy),bo),bo) height:5 totnodes:63  
    Traceback (most recent call last):
      File "/Users/blyth/opticks/bin/gdml2gltf.py", line 28, in <module>
        sc = gdml2gltf_main( args )
      File "/Users/blyth/opticks/analytic/sc.py", line 479, in gdml2gltf_main
        tg = sc.add_tree_gdml( tree.root, maxdepth=0)
      File "/Users/blyth/opticks/analytic/sc.py", line 364, in add_tree_gdml
        tg = build_r(target)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 351, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 348, in build_r
        nd = self.add_node_gdml(node, depth)
      File "/Users/blyth/opticks/analytic/sc.py", line 245, in add_node_gdml
        csg = self.translate_lv( node.lv, self.maxcsgheight )
      File "/Users/blyth/opticks/analytic/sc.py", line 296, in translate_lv
        csg = cls.optimize_csg(rawcsg, maxcsgheight, maxcsgheight2 )
      File "/Users/blyth/opticks/analytic/sc.py", line 335, in optimize_csg
        assert not overheight_(csg, maxcsgheight2)
    AssertionError
    /Users/blyth/opticks/bin/op.sh RC 1
    simon:analytic blyth$ 



After allowing overheight thru::

    [2017-08-03 12:01:05,252] p55875 {/Users/blyth/opticks/analytic/sc.py:355} INFO - add_tree_gdml count 288000 depth 7 maxdepth 0 
    [2017-08-03 12:01:06,437] p55875 {/Users/blyth/opticks/analytic/sc.py:355} INFO - add_tree_gdml count 289000 depth 7 maxdepth 0 
    [2017-08-03 12:01:08,137] p55875 {/Users/blyth/opticks/analytic/sc.py:355} INFO - add_tree_gdml count 290000 depth 7 maxdepth 0 
    [2017-08-03 12:01:08,596] p55875 {/Users/blyth/opticks/analytic/treebuilder.py:34} WARNING - balancing trees of this structure not implemented, tree un(in(in(in(cy,!cy),!bo),!cy),in(in(bo,!bo),!cy)) height:4 totnodes:31  
    [2017-08-03 12:01:08,599] p55875 {/Users/blyth/opticks/analytic/treebuilder.py:34} WARNING - balancing trees of this structure not implemented, tree un(un(in(in(in(bo,!cy),!cy),!cy),bo),bo) height:5 totnodes:63  
    [2017-08-03 12:01:08,610] p55875 {/Users/blyth/opticks/analytic/sc.py:376} INFO - add_tree_gdml DONE maxdepth:0 maxcsgheight:3 nodesCount:290276 tlvCount:35 addNodeCount:290276 tgNd:                           top Nd ndIdx:  0 soIdx:0 nch:2 par:-1 matrix:[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]   
    [2017-08-03 12:01:08,610] p55875 {/Users/blyth/opticks/analytic/sc.py:492} INFO - start Sc.add_tree_gdml DONE
    [2017-08-03 12:01:08,610] p55875 {/Users/blyth/opticks/analytic/sc.py:409} INFO - saving to /usr/local/opticks/opticksdata/export/juno1707/g4_00.gltf 
    [2017-08-03 12:01:08,711] p55875 {/Users/blyth/opticks/analytic/sc.py:398} INFO - save_extras /usr/local/opticks/opticksdata/export/juno1707/extras  : saved 35 
    [2017-08-03 12:01:08,711] p55875 {/Users/blyth/opticks/analytic/sc.py:402} INFO - write 35 lines to /usr/local/opticks/opticksdata/export/juno1707/extras/csg.txt 
    [2017-08-03 12:01:28,241] p55875 {/Users/blyth/opticks/analytic/sc.py:418} INFO - also saving to /usr/local/opticks/opticksdata/export/juno1707/g4_00.pretty.gltf 
    /Users/blyth/opticks/bin/op.sh RC 0


opticks-tbool machinery based on IDPATH envvar::

    simon:opticksnpy blyth$ op --idpath
    === op-cmdline-binary-match : finds 1st argument with associated binary : --idpath
    ubin /usr/local/opticks/lib/OpticksIDPATH cfm --idpath cmdline --idpath
    IDPATH /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
    simon:opticksnpy blyth$ 
    simon:opticksnpy blyth$ 
    simon:opticksnpy blyth$ 
    simon:opticksnpy blyth$ op --idpath --j1707
    === op-cmdline-binary-match : finds 1st argument with associated binary : --idpath
    ubin /usr/local/opticks/lib/OpticksIDPATH cfm --idpath cmdline --idpath --j1707
    IDPATH /usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae
    simon:opticksnpy blyth$ 




tbool spin over the solids  : NB selection dependant solid lvidx with 50k selection
-----------------------------------------------------------------------------------------

11 : looks like zs only ??? : lots of torus residual output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:opticksnpy blyth$ opticks-tbool- 11
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/juno1707/extras/11/tbool11.bash
    args: 
    [2017-08-03 12:16:31,641] p60174 {/Users/blyth/opticks/analytic/csg.py:941} INFO - raw name:intersection
    in(un(un(zs,in(cy,!to)),cy),cy) height:4 totnodes:31 

                                 in    
                         un          cy
         un                  cy        
     zs          in                    
             cy     !to                
    [2017-08-03 12:16:31,641] p60174 {/Users/blyth/opticks/analytic/csg.py:941} INFO - optimized name:intersection
    in(un(un(zs,in(cy,!to)),cy),cy) height:4 totnodes:31 

                                 in    
                         un          cy
         un                  cy        
     zs          in                    
             cy     !to                
    [2017-08-03 12:16:31,642] p60174 {/Users/blyth/opticks/analytic/csg.py:424} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/11 
    analytic=1_csgpath=/tmp/blyth/opticks/tbool/11_name=11_mode=PyCsgInBox
    simon:opticksnpy blyth$ 



::

    080 # generated by tboolean.py : 20170803-1201 
     81 # opticks-;opticks-tbool 11 
     82 # opticks-;opticks-tbool-vi 11 
     83 
     84 
     85 a = CSG("zsphere", param = [0.000,0.000,0.000,179.000],param1 = [-179.000,179.000,0.000,0.000])
     86 a.transform = [[1.391,0.000,0.000,0.000],[0.000,1.391,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,0.000,1.000]]
     87 b = CSG("cylinder", param = [0.000,0.000,0.000,75.951],param1 = [-23.783,23.783,0.000,0.000])
     88 c = CSG("torus", param = [0.000,0.000,52.010,97.000],param1 = [0.000,0.000,0.000,0.000],complement = True)
     89 c.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-23.773,1.000]]
     90 bc = CSG("intersection", left=b, right=c)
     91 bc.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-195.227,1.000]]
     92 
     93 abc = CSG("union", left=a, right=bc)
     94 
     95 d = CSG("cylinder", param = [0.000,0.000,0.000,45.010],param1 = [-57.510,57.510,0.000,0.000])
     96 d.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-276.500,1.000]]
     97 abcd = CSG("union", left=abc, right=d)
     98 
     99 e = CSG("cylinder", param = [0.000,0.000,0.000,254.000],param1 = [-92.000,92.000,0.000,0.000])
    100 e.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,92.000,1.000]]
    101 abcde = CSG("intersection", left=abcd, right=e)

        ae = CSG("intersection", left=a, right=e)
    102 
    103 
    104 
    105 #raw = abcde
    106 #raw = bc         # in(cy,!to)              neck - torus artifact visible 
    107 #raw = abc        # un(zs,in(cy,!to))       ditto   : pretty PMT bulb and neck 
    108 #raw = abcd        # cy extending down
    109 #raw = abcde     # WOW: all that expensive PMT geometry is chopped away, just intersecting 
    110 raw = ae         # SAME, MUCH LESS EXPENSIVE 


* TODO: dump solid names in codegen
* TODO: auto-prune tree killing prim that have no effect on final shape : ie most of the above 




12 : looks like flat top tree : gives torus residual output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:opticksnpy blyth$ opticks-tbool- 12
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/juno1707/extras/12/tbool12.bash
    args: 
    [2017-08-03 12:20:21,359] p60506 {/Users/blyth/opticks/analytic/csg.py:941} INFO - raw name:intersection
    in(un(un(zs,in(cy,!to)),cy),!cy) height:4 totnodes:31 

                                 in    
                         un         !cy
         un                  cy        
     zs          in                    
             cy     !to                
    [2017-08-03 12:20:21,360] p60506 {/Users/blyth/opticks/analytic/csg.py:941} INFO - optimized name:intersection
    in(un(un(zs,in(cy,!to)),cy),!cy) height:4 totnodes:31 

                                 in    
                         un         !cy
         un                  cy        
     zs          in                    
             cy     !to                
    [2017-08-03 12:20:21,360] p60506 {/Users/blyth/opticks/analytic/csg.py:424} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/12 
    analytic=1_csgpath=/tmp/blyth/opticks/tbool/12_name=12_mode=PyCsgInBox



13 : looks like PMT : gives torus residual output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    simon:opticksnpy blyth$ opticks-tbool- 13
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/juno1707/extras/13/tbool13.bash
    args: 
    [2017-08-03 12:21:49,995] p60835 {/Users/blyth/opticks/analytic/csg.py:941} INFO - raw name:union
    un(un(zs,di(cy,to)),cy) height:3 totnodes:15 

                         un    
         un                  cy
     zs          di            
             cy      to        
    [2017-08-03 12:21:49,996] p60835 {/Users/blyth/opticks/analytic/csg.py:941} INFO - optimized name:union
    un(un(zs,di(cy,to)),cy) height:3 totnodes:15 

                         un    
         un                  cy
     zs          di            
             cy      to        
    [2017-08-03 12:21:49,996] p60835 {/Users/blyth/opticks/analytic/csg.py:424} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/13 
    analytic=1_csgpath=/tmp/blyth/opticks/tbool/13_name=13_mode=PyCsgInBox


14 : again PMT : Neumark constant term zero artifacting apparent at torus side view
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    simon:opticksnpy blyth$ opticks-tbool- 14
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/juno1707/extras/14/tbool14.bash
    args: 
    [2017-08-03 12:25:09,225] p61166 {/Users/blyth/opticks/analytic/csg.py:941} INFO - raw name:union
    un(un(zs,di(cy,to)),cy) height:3 totnodes:15 

                         un    
         un                  cy
     zs          di            
             cy      to        
    [2017-08-03 12:25:09,226] p61166 {/Users/blyth/opticks/analytic/csg.py:941} INFO - optimized name:union
    un(un(zs,di(cy,to)),cy) height:3 totnodes:15 

                         un    
         un                  cy
     zs          di            
             cy      to        
    [2017-08-03 12:25:09,226] p61166 {/Users/blyth/opticks/analytic/csg.py:424} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/14 
    analytic=1_csgpath=/tmp/blyth/opticks/tbool/14_name=14_mode=PyCsgInBox



21 : codegen error : from def reserved word : FIXED
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:opticksnpy blyth$ opticks-tbool- 21
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/juno1707/extras/21/tbool21.bash
      File "<stdin>", line 35
        def = CSG("difference", left=de, right=f)
            ^
    SyntaxError: invalid syntax
    simon:opticksnpy blyth$ 

::

    100 def = CSG("difference", left=de, right=f)
    101 def.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-1665.000,1.000]]
    102 
    103 abcdef = CSG("union", left=abc, right=def)
    104 
    105 




23 : coincident subtraction artifact ring at base
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    simon:opticksnpy blyth$ opticks-tbool- 23
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/juno1707/extras/23/tbool23.bash
    args: 
    [2017-08-03 12:35:06,774] p63917 {/Users/blyth/opticks/analytic/csg.py:941} INFO - raw name:union
    un(in(in(in(cy,!cy),!bo),!cy),in(in(bo,!bo),!cy)) height:4 totnodes:31 

                                 un                    
                         in                      in    
                 in         !cy          in         !cy
         in         !bo              bo     !bo        
     cy     !cy                                        
    [2017-08-03 12:35:06,774] p63917 {/Users/blyth/opticks/analytic/csg.py:941} INFO - optimized name:union
    un(in(in(in(cy,!cy),!bo),!cy),in(in(bo,!bo),!cy)) height:4 totnodes:31 

                                 un                    
                         in                      in    
                 in         !cy          in         !cy
         in         !bo              bo     !bo        
     cy     !cy                                        
    [2017-08-03 12:35:06,775] p63917 {/Users/blyth/opticks/analytic/csg.py:424} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/23 
    analytic=1_csgpath=/tmp/blyth/opticks/tbool/23_name=23_mode=PyCsgInBox




24 : no visibile cy cutouts ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:opticksnpy blyth$ opticks-tbool- 24
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/juno1707/extras/24/tbool24.bash
    args: 
    [2017-08-03 12:37:10,199] p64249 {/Users/blyth/opticks/analytic/csg.py:941} INFO - raw name:union
    un(un(in(in(in(bo,!cy),!cy),!cy),bo),bo) height:5 totnodes:63 

                                         un    
                                 un          bo
                         in          bo        
                 in         !cy                
         in         !cy                        
     bo     !cy                                
    [2017-08-03 12:37:10,200] p64249 {/Users/blyth/opticks/analytic/treebuilder.py:34} WARNING - balancing trees of this structure not implemented, tree un(un(in(in(in(bo,!cy),!cy),!cy),bo),bo) height:5 totnodes:63  
    [2017-08-03 12:37:10,200] p64249 {/Users/blyth/opticks/analytic/csg.py:941} INFO - optimized name:union
    un(un(in(in(in(bo,!cy),!cy),!cy),bo),bo) height:5 totnodes:63 

                                         un    
                                 un          bo
                         in          bo        
                 in         !cy                
         in         !cy                        
     bo     !cy                                
    [2017-08-03 12:37:10,201] p64249 {/Users/blyth/opticks/analytic/csg.py:424} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/24 
    analytic=1_csgpath=/tmp/blyth/opticks/tbool/24_name=24_mode=PyCsgInBox
    simon:opticksnpy blyth$ 



27,28 : only torus artifact rings nothing else ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ginormous R/r 17836/8=2229.5 guide tube -> numerical nightmare

::

    imon:issues blyth$ opticks-;opticks-tbool- 27
    opticks-tbool- : sourcing /usr/local/opticks/opticksdata/export/juno1707/extras/27/tbool27.bash
    args: 
    [2017-08-03 15:02:19,047] p73646 {/Users/blyth/opticks/analytic/csg.py:946} INFO - raw name:torus
    to height:0 totnodes:1 

     to
    [2017-08-03 15:02:19,048] p73646 {/Users/blyth/opticks/analytic/csg.py:946} INFO - optimized name:torus
    to height:0 totnodes:1 

     to
    [2017-08-03 15:02:19,048] p73646 {/Users/blyth/opticks/analytic/csg.py:429} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/27 
    analytic=1_csgpath=/tmp/blyth/opticks/tbool/27_name=27_mode=PyCsgInBox

::

     80 # generated by tboolean.py : 20170803-1201 
     81 # opticks-;opticks-tbool 27 
     82 # opticks-;opticks-tbool-vi 27 
     83 
     84 
     85 a = CSG("torus", param = [0.000,0.000,8.000,17836.000],param1 = [0.000,0.000,0.000,0.000])
     86 
     87 
     88 raw = a

     80 # generated by tboolean.py : 20170803-1201 
     81 # opticks-;opticks-tbool 28 
     82 # opticks-;opticks-tbool-vi 28 
     83 
     84 
     85 a = CSG("torus", param = [0.000,0.000,10.000,17836.000],param1 = [0.000,0.000,0.000,0.000])
     86 
     87 
     88 raw = a
     89 





scene viz : GPU triangulated
--------------------------------

* works : wireframe view caused GPU hang : TODO: disable that 

::

    simon:opticks blyth$ op --j1707 --tracer
    === op-cmdline-binary-match : finds 1st argument with associated binary : --tracer
    ubin /usr/local/opticks/lib/OTracerTest cfm --tracer cmdline --j1707 --tracer
    288 -rwxr-xr-x  1 blyth  staff  145944 Aug  2 19:16 /usr/local/opticks/lib/OTracerTest
    proceeding : /usr/local/opticks/lib/OTracerTest --j1707 --tracer
    dedupe skipping --tracer 
    2017-08-03 12:41:54.114 INFO  [1792911] [OpticksQuery::dump@78] OpticksQuery::init queryType range query_string range:1:50000 query_name NULL query_index 0 query_depth 0 nrange 2 : 1 : 50000
    2017-08-03 12:41:54.115 INFO  [1792911] [Opticks::init@319] Opticks::init DONE OpticksResource::desc digest dc4c5b76e112378f74220a1112129841 age.tot_seconds 1044307 age.tot_minutes 17405.117 age.tot_hours 290.085 age.tot_days     12.087




::

    simon:optickscore blyth$ op --resource --j1707
    === op-cmdline-binary-match : finds 1st argument with associated binary : --resource
    ubin /usr/local/opticks/lib/OpticksResourceTest cfm --resource cmdline --resource --j1707
    232 -rwxr-xr-x  1 blyth  staff  118380 Jul 27 19:56 /usr/local/opticks/lib/OpticksResourceTest
    proceeding : /usr/local/opticks/lib/OpticksResourceTest --resource --j1707
    OpticksResource::Dump
    install_prefix    : /usr/local/opticks
    opticksdata_dir   : /usr/local/opticks/opticksdata
    resource_dir      : /usr/local/opticks/opticksdata/resource
    valid    : valid
    envprefix: OPTICKS_
    geokey   : OPTICKSDATA_DAEPATH_J1707
    daepath  : /usr/local/opticks/opticksdata/export/juno1707/g4_00.dae
    gdmlpath : /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml
    gltfpath : /usr/local/opticks/opticksdata/export/juno1707/g4_00.gltf
    metapath : /usr/local/opticks/opticksdata/export/juno1707/g4_00.ini
    query    : range:1:50000
    ctrl     : volnames
    digest   : dc4c5b76e112378f74220a1112129841
    idpath   : /usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae
    idpath_tmp NULL
    idfold   : /usr/local/opticks/opticksdata/export/juno1707
    idname   : juno1707
    idbase   : /usr/local/opticks/opticksdata/export
    detector : juno
    detector_name : juno
    detector_base : /usr/local/opticks/opticksdata/export/juno
    material_map  : /usr/local/opticks/opticksdata/export/juno/ChromaMaterialMap.json
    getPmtPath(0) : /usr/local/opticks/opticksdata/export/juno/GPmt/0
    meshfix  : iav,oav
    ------ from /usr/local/opticks/opticksdata/export/juno1707/g4_00.ini -------- 
    mmsp(0) :/usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae/GMergedMesh/0
    pmtp(0) :/usr/local/opticks/opticksdata/export/juno/GPmt/0
    /Users/blyth/opticks/bin/op.sh RC 0
    simon:optickscore blyth$ 






scene viz : GPU analytic
--------------------------------

::

    simon:opticks blyth$ op --j1707 --tracer --gltf 3
    === op-cmdline-binary-match : finds 1st argument with associated binary : --tracer
    ubin /usr/local/opticks/lib/OTracerTest cfm --tracer cmdline --j1707 --tracer --gltf 3
    288 -rwxr-xr-x  1 blyth  staff  145944 Aug  2 19:16 /usr/local/opticks/lib/OTracerTest
    proceeding : /usr/local/opticks/lib/OTracerTest --j1707 --tracer --gltf 3
    dedupe skipping --tracer 
    2017-08-03 12:46:01.274 INFO  [1794425] [OpticksQuery::dump@78] OpticksQuery::init queryType range query_string range:1:50000 query_name NULL query_index 0 query_depth 0 nrange 2 : 1 : 50000
    2017-08-03 12:46:01.274 INFO  [1794425] [Opticks::init@319] Opticks::init DONE OpticksResource::desc digest dc4c5b76e112378f74220a1112129841 age.tot_seconds 1044554 age.tot_minutes 17409.232 age.tot_hours 290.154 age.tot_days     12.090
    2017-08-03 12:46:01.275 WARN  [1794425] [BTree::loadTree@48] BTree.loadTree: can't find file /usr/local/opticks/opticksdata/export/juno/ChromaMaterialMap.json
    2017-08-03 12:46:01.275 FATAL [1794425] [NSensorList::read@133] NSensorList::read failed to open /usr/local/opticks/opticksdata/export/juno1707/g4_00.idmap
    2017-08-03 12:46:01.275 INFO  [1794425] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-08-03 12:46:01.538 INFO  [1794425] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-08-03 12:46:01.582 INFO  [1794425] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae/GMergedMesh/2 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae/GMergedMesh/2 index 2 version (null) existsdir 1
    2017-08-03 12:46:01.607 INFO  [1794425] [*GMergedMesh::load@634] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae/GMergedMesh/3 -> cachedir /usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae/GMergedMesh/3 index 3 version (null) existsdir 1
    2017-08-03 12:46:02.186 INFO  [1794425] [GMeshLib::loadMeshes@206] idpath /usr/local/opticks/opticksdata/export/juno1707/g4_00.dc4c5b76e112378f74220a1112129841.dae
    2017-08-03 12:46:02.195 INFO  [1794425] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-08-03 12:46:02.195 INFO  [1794425] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 15
    2017-08-03 12:46:02.195 INFO  [1794425] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [Acrylic]
    2017-08-03 12:46:02.196 INFO  [1794425] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 15,2,39,4
    2017-08-03 12:46:02.197 WARN  [1794425] [*GPmt::load@44] GPmt::load resource does not exist /usr/local/opticks/opticksdata/export/juno/GPmt/0
    2017-08-03 12:46:02.198 INFO  [1794425] [GGeo::loadAnalyticPmt@761] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path -
    2017-08-03 12:46:02.198 INFO  [1794425] [NGLTF::load@35] NGLTF::load path /usr/local/opticks/opticksdata/export/juno1707/g4_00.gltf
    2017-08-03 12:46:11.918 INFO  [1794425] [NGLTF::load@62] NGLTF::load DONE
    2017-08-03 12:46:12.491 INFO  [1794425] [NSceneConfig::NSceneConfig@48] NSceneConfig::NSceneConfig cfg [check_surf_containment=0,check_aabb_containment=0]
            check_surf_containment :                    0
            check_aabb_containment :                    0
    2017-08-03 12:46:12.491 INFO  [1794425] [BFile::preparePath@462] preparePath : created directory /usr/local/opticks/opticksdata/export/juno1707/g4_00
    2017-08-03 12:46:12.491 WARN  [1794425] [NScene::load_asset_extras@275] NScene::load_asset_extras verbosity increase from scene gltf  extras_verbosity 1 m_verbosity 0
    2017-08-03 12:46:12.491 INFO  [1794425] [NScene::init@177] NScene::init START age(s) 2684 days   0.031
    2017-08-03 12:46:12.491 INFO  [1794425] [NScene::load_csg_metadata@310] NScene::load_csg_metadata verbosity 1 num_meshes 35
    nglmext::invert_trs polar_decomposition inverse and straight inverse are mismatched  epsilon 1e-05 diff 0.00195312 diff2 0.00195312 diffFractional 2 diffFractionalMax 0.001
           trs -0.847  -0.489   0.208   0.000 
               -0.500   0.866  -0.000   0.000 
               -0.180  -0.104  -0.978   0.000 
              3352.658 1935.658 18213.107   1.000 

        isirit -0.847  -0.500  -0.180   0.000 
               -0.489   0.866  -0.104   0.000 
                0.208  -0.000  -0.978   0.000 
               -0.000   0.001 18619.998   1.000 

        i_trs  -0.847  -0.500  -0.180   0.000 
               -0.489   0.866  -0.104   0.000 
                0.208  -0.000  -0.978  -0.000 
               -0.000   0.001 18620.000   1.000 

    [ -0.847101: -0.847101:         0:        -0][      -0.5:      -0.5:         0:        -0][ -0.180057: -0.180057:1.49012e-08:-8.27581e-08][**         0:         0:         0:       nan**]
    [ -0.489074: -0.489074:2.98023e-08:-6.09363e-08][  0.866025:  0.866026:5.96046e-08:6.88255e-08][ -0.103956: -0.103956:2.23517e-08:-2.15012e-07][**         0:         0:         0:       nan**]
    [  0.207912:  0.207912:         0:         0][**-7.82311e-08:-7.45058e-08:         0:       nan**][ -0.978148: -0.978148:1.19209e-07:-1.21873e-07][**         0:        -0:         0:       nan**]
    [-0.000488281:-0.000276849:0.000211432: -0.552669][**0.00105862:0.000991064:6.7556e-05: 0.0659185**][     18620:     18620:0.00195312:1.04894e-07][         1:         1:         0:         0]
    Assertion failed: (match), function invert_trs, file /Users/blyth/opticks/opticksnpy/NGLMExt.cpp, line 221.
    /Users/blyth/opticks/bin/op.sh: line 669: 65756 Abort trap: 6           /usr/local/opticks/lib/OTracerTest --j1707 --tracer --gltf 3
    /Users/blyth/opticks/bin/op.sh RC 134



Skip the assert::

    simon:opticks blyth$ op --j1707 --tracer --gltf 3


There are many of them from nan in the comparison::

    nglmext::invert_trs polar_decomposition inverse and straight inverse are mismatched  epsilon 1e-05 diff 0.00195312 diff2 0.00195312 diffFractional 2 diffFractionalMax 0.001
           trs  0.713   0.654   0.252   0.000 
               -0.676   0.737   0.000   0.000 
               -0.186  -0.170   0.968   0.000 
              3612.057 3315.116 -18821.943   1.000 

        isirit  0.713  -0.676  -0.186   0.000 
                0.654   0.737  -0.170   0.000 
                0.252   0.000   0.968   0.000 
                0.001  -0.000 19449.998   1.000 

        i_trs   0.713  -0.676  -0.186  -0.000 
                0.654   0.737  -0.170   0.000 
                0.252  -0.000   0.968  -0.000 
                0.000   0.000 19450.000   1.000 

    [  0.712951:  0.712951:5.96046e-08:8.36027e-08][ -0.676175: -0.676175:         0:        -0][  -0.18571:  -0.18571:1.49012e-08:-8.02389e-08][**         0:        -0:         0:       nan**]
    [  0.654341:  0.654341:         0:         0][  0.736741:  0.736741:5.96046e-08:8.09031e-08][ -0.170443: -0.170443:1.49012e-08:-8.74261e-08][**         0:         0:         0:       nan**]
    [  0.252069:  0.252069:2.98023e-08:1.18231e-07][**         0:        -0:         0:       nan**][  0.967709:  0.967709:1.19209e-07:1.23187e-07][**         0:        -0:         0:       nan**]




Huh 35 extras 
------------------------------------------

35 is correct::

    simon:analytic blyth$ grep solidref /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml | wc -l
          35


::

    simon:issues blyth$ ls /usr/local/opticks/opticksdata/export/juno1707/extras/
    0   10  12  14  16  18  2   21  23  25  27  29  30  32  34  5   7   9
    1   11  13  15  17  19  20  22  24  26  28  3   31  33  4   6   8   csg.txt
    simon:issues blyth$ 
    simon:issues blyth$ wc /usr/local/opticks/opticksdata/export/juno1707/extras/csg.txt 
          34      35     339 /usr/local/opticks/opticksdata/export/juno1707/extras/csg.txt
    simon:issues blyth$ 


Need to adjust selection to get all solids it seems...

* see if can avoid that, ie treat solids at mesh level not at node level
* reason was probably to use lvidx : but constant solid/mesh indices based only on source gdml trumps that 



Huh, they are there where 26 from ? Seems fine 0:34 ::

    simon:extras blyth$ l -1 */*.bash  | wc
          35      35     540



::

    op --j1707 --tracer --gltf 3


    2017-08-03 13:01:58.520 INFO  [1800773] [NScene::postimportnd@558] NScene::postimportnd numNd 290276 num_selected 49999 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-08-03 13:02:02.272 INFO  [1800773] [NScene::count_progeny_digests@932] NScene::count_progeny_digests verbosity 1 node_count 290276 digest_size 35
    2017-08-03 13:02:04.573 INFO  [1800773] [*NCSG::make_nudger@130] sWorld0x14d9850 treeNameIdx 34
    2017-08-03 13:02:04.593 INFO  [1800773] [*NCSG::make_nudger@130] sTopRock0x14da370 treeNameIdx 5
    2017-08-03 13:02:04.615 INFO  [1800773] [*NCSG::make_nudger@130] sExpHall0x14da850 treeNameIdx 4
    2017-08-03 13:02:04.635 INFO  [1800773] [*NCSG::make_nudger@130] Upper_Chimney0x25476d0 treeNameIdx 3
    2017-08-03 13:02:04.646 INFO  [1800773] [*NCSG::make_nudger@130] Upper_LS_tube0x2547790 treeNameIdx 0
    2017-08-03 13:02:04.660 INFO  [1800773] [*NCSG::make_nudger@130] Upper_Steel_tube0x2547890 treeNameIdx 1
    2017-08-03 13:02:04.664 INFO  [1800773] [*NCSG::make_nudger@130] Upper_Tyvek_tube0x2547990 treeNameIdx 2
    2017-08-03 13:02:04.667 INFO  [1800773] [*NCSG::make_nudger@130] sBottomRock0x14dab90 treeNameIdx 33
    2017-08-03 13:02:04.685 INFO  [1800773] [*NCSG::make_nudger@130] sPoolLining0x14db2e0 treeNameIdx 32
    2017-08-03 13:02:04.700 INFO  [1800773] [*NCSG::make_nudger@130] sOuterWaterPool0x14dbc70 treeNameIdx 31
    2017-08-03 13:02:04.716 INFO  [1800773] [*NCSG::make_nudger@130] sReflectorInCD0x14dc560 treeNameIdx 30
    2017-08-03 13:02:04.738 INFO  [1800773] [*NCSG::make_nudger@130] sInnerWater0x14dcb00 treeNameIdx 29
    2017-08-03 13:02:04.761 INFO  [1800773] [*NCSG::make_nudger@130] sAcrylic0x14dd0a0 treeNameIdx 7
    2017-08-03 13:02:04.783 INFO  [1800773] [*NCSG::make_nudger@130] sTarget0x14dd640 treeNameIdx 6
    2017-08-03 13:02:04.806 INFO  [1800773] [*NCSG::make_nudger@130] sStrut0x14ddd50 treeNameIdx 8
    2017-08-03 13:02:04.810 INFO  [1800773] [*NCSG::make_nudger@130] sFasteners0x1506180 treeNameIdx 9
    2017-08-03 13:02:04.814 INFO  [1800773] [*NCSG::make_nudger@130] sMask_virtual0x18163c0 treeNameIdx 15
    2017-08-03 13:02:04.830 INFO  [1800773] [*NCSG::make_nudger@130] sMask0x1816f50 treeNameIdx 10
    2017-08-03 13:02:04.833 INFO  [1800773] [*NCSG::make_nudger@130] PMT_20inch_pmt_solid0x1813600 treeNameIdx 14
    2017-08-03 13:02:04.866 INFO  [1800773] [*NCSG::make_nudger@130] PMT_20inch_body_solid0x1813ec0 treeNameIdx 13
    2017-08-03 13:02:04.903 INFO  [1800773] [*NCSG::make_nudger@130] PMT_20inch_inner1_solid0x1814a90 treeNameIdx 11
    2017-08-03 13:02:04.953 INFO  [1800773] [*NCSG::make_nudger@130] PMT_20inch_inner2_solid0x1863010 treeNameIdx 12
    2017-08-03 13:02:04.987 INFO  [1800773] [*NCSG::make_nudger@130] PMT_3inch_pmt_solid0x1c9e270 treeNameIdx 20
    2017-08-03 13:02:05.012 INFO  [1800773] [*NCSG::make_nudger@130] PMT_3inch_body_solid_ell_ell_helper0x1c9e4a0 treeNameIdx 18
    2017-08-03 13:02:05.026 INFO  [1800773] [*NCSG::make_nudger@130] PMT_3inch_inner1_solid_ell_helper0x1c9e510 treeNameIdx 16
    2017-08-03 13:02:05.039 INFO  [1800773] [*NCSG::make_nudger@130] PMT_3inch_inner2_solid_ell_helper0x1c9e5d0 treeNameIdx 17
    2017-08-03 13:02:05.055 INFO  [1800773] [*NCSG::make_nudger@130] PMT_3inch_cntr_solid0x1c9e640 treeNameIdx 19
    2017-08-03 13:02:05.057 INFO  [1800773] [*NCSG::make_nudger@130] upper_tubeTyvek0x254a890 treeNameIdx 26
    2017-08-03 13:02:05.079 INFO  [1800773] [*NCSG::make_nudger@130] unionLS10x2548db0 treeNameIdx 21
    2017-08-03 13:02:05.083 INFO  [1800773] [*NCSG::make_nudger@130] AcrylicTube0x2548f40 treeNameIdx 22
    2017-08-03 13:02:05.097 INFO  [1800773] [*NCSG::make_nudger@130] unionSteel0x2549960 treeNameIdx 23
    2017-08-03 13:02:05.102 INFO  [1800773] [*NCSG::make_nudger@130] unionLS10x2549c00 treeNameIdx 25
    2017-08-03 13:02:05.118 INFO  [1800773] [*NCSG::make_nudger@130] unionBlocker0x254a570 treeNameIdx 24
    2017-08-03 13:02:05.121 INFO  [1800773] [*NCSG::make_nudger@130] sSurftube0x2548170 treeNameIdx 28
    2017-08-03 13:02:05.127 INFO  [1800773] [*NCSG::make_nudger@130] svacSurftube0x254ba10 treeNameIdx 27
    2017-08-03 13:02:05.132 INFO  [1800773] [NScene::postimportmesh@576] NScene::postimportmesh numNd 290276 dbgnode -1 dbgnode_list 0 verbosity 1
    2017-08-03 13:02:05.132 INFO  [1800773] [BConfig::dump@39] NScene::postimportmesh.cfg eki 13



Solid identity rejig ?
~~~~~~~~~~~~~~~~~~~~~~~~

Ah-ha I see why using lvIdx... tis because there is no list of meshes
in GDML just a list of solid elements most of which are just primitives that 
warrant no mesh ixd as are part of the composites that are the "meshes".

So using lvIdx makes sense : but need to rearrange for it to be absolute 
for a GDML file (ie not depending on selection).


::

   243   <solids>
   244     <tube aunit="deg" deltaphi="360" lunit="mm" name="Upper_LS_tube0x2547790" rmax="400" rmin="0" startphi="0" z="3500"/>
   245     <opticalsurface finish="3" model="1" name="UpperChimneyTyvekOpticalSurface" type="0" value="0.2"/>
   246     <tube aunit="deg" deltaphi="360" lunit="mm" name="Upper_Steel_tube0x2547890" rmax="407" rmin="402" startphi="0" z="3500"/>
   247     <tube aunit="deg" deltaphi="360" lunit="mm" name="Upper_Tyvek_tube0x2547990" rmax="402" rmin="400" startphi="0" z="3500"/>
   248     <tube aunit="deg" deltaphi="360" lunit="mm" name="Upper_Chimney0x25476d0" rmax="412" rmin="0" startphi="0" z="3500"/>
   249     <box lunit="mm" name="sExpHall0x14da850" x="48000" y="48000" z="18600"/>
   250     <box lunit="mm" name="sTopRock0x14da370" x="54000" y="54000" z="21600"/>
   251     <sphere aunit="deg" deltaphi="360" deltatheta="180" lunit="mm" name="sTarget_bottom_ball0x14dd400" rmax="17700" rmin="0" startphi="0" starttheta="0"/>
   252     <tube aunit="deg" deltaphi="360" lunit="mm" name="sTarget_top_tube0x14dd580" rmax="400" rmin="0" startphi="0" z="124.520351230938"/>
   253     <union name="sTarget0x14dd640">
   254       <first ref="sTarget_bottom_ball0x14dd400"/>
   255       <second ref="sTarget_top_tube0x14dd580"/>
   256       <position name="sTarget0x14dd640_pos" unit="mm" x="0" y="0" z="17757.7398243845"/>
   257     </union>



CSG.save treedir arg assumed lvidx::

     645     def save(self, treedir):
     646         if not os.path.exists(treedir):
     647             os.makedirs(treedir)
     648         pass
     649 
     650         nodebuf, tranbuf, planebuf = self.serialize()
     651 
     652         metapath = self.metapath(treedir)
     653         json.dump(self.meta,file(metapath,"w"))
     654 
     655         self.save_nodemeta(treedir)
     656 
     657         lvidx = os.path.basename(treedir)
     658         tboolpath = self.tboolpath(treedir, lvidx)
     659         self.write_tbool(lvidx, tboolpath)
     660 

Sc.save_extras::

    379     def save_extras(self, gdir):
    380         gdir = expand_(gdir)
    381         extras_dir = os.path.join( gdir, "extras" )
    382         log.debug("save_extras %s " % extras_dir )
    383         if not os.path.exists(extras_dir):
    384             os.makedirs(extras_dir)
    385         pass
    386         btxt = []
    387         count = 0
    388         for lvIdx, mesh in self.meshes.items():
    389             soIdx = mesh.soIdx
    390             lvdir = os.path.join( extras_dir, "%d" % lvIdx )
    391             uri = os.path.relpath(lvdir, gdir)
    392             mesh.extras["uri"] = uri
    393             mesh.csg.save(lvdir)
    394             btxt.append(uri)
    395             count += 1
    396         pass
    397 
    398         log.info("save_extras %s  : saved %d " % (extras_dir, count) )
    399 
    400         csgtxt_path = os.path.join(extras_dir, "csg.txt")
    401         log.info("write %d lines to %s " % (len(btxt), csgtxt_path))
    402         file(csgtxt_path,"w").write("\n".join(btxt))






9/35 discrepant bbox
----------------------

::

    op --j1707 --tracer --gltf 3

    2017-08-03 13:02:05.192 INFO  [1800773] [GMeshLib::add@178] GMeshLib::add (GMesh) index   34 name svacSurftube0x254ba10
    2017-08-03 13:02:05.192 INFO  [1800773] [GScene::importMeshes@317] GScene::importMeshes DONE num_meshes 35
    2017-08-03 13:02:05.192 INFO  [1800773] [GScene::compareMeshes_GMeshBB@436] GScene::compareMeshes_GMeshBB num_meshes 35 cut 0.1 bbty CSG_BBOX_PARSURF parsurf_level 2 parsurf_target 200
       85.3516                    sInnerWater0x14dcb00 lvidx  29 nsp    487                             union sphere cylinder   nds[  1]  11 . 
       85.2539                 sReflectorInCD0x14dc560 lvidx  30 nsp    490                             union sphere cylinder   nds[  1]  10 . 
       10.9004                   svacSurftube0x254ba10 lvidx  27 nsp    531                                             torus   nds[  1]  290275 . 
       10.9004                      sSurftube0x2548170 lvidx  28 nsp    296                                             torus   nds[  1]  290274 . 
       7.12817          PMT_20inch_body_solid0x1813ec0 lvidx  13 nsp    532           union difference zsphere cylinder torus   nds[17739]  977 983 989 995 1001 1007 1013 1019 1025 1031 ... 
        6.8313        PMT_20inch_inner2_solid0x1863010 lvidx  12 nsp    681         union intersection zsphere cylinder torus   nds[17739]  979 985 991 997 1003 1009 1015 1021 1027 1033 ... 
       1.85201           PMT_20inch_pmt_solid0x1813600 lvidx  14 nsp    223           union difference zsphere cylinder torus   nds[17739]  976 982 988 994 1000 1006 1012 1018 1024 1030 ... 
         1.815        PMT_20inch_inner1_solid0x1814a90 lvidx  11 nsp    391         union intersection zsphere cylinder torus   nds[17739]  978 984 990 996 1002 1008 1014 1020 1026 1032 ... 
      0.127613PMT_3inch_inner2_solid_ell_helper0x1c9e5d0 lvidx  17 nsp    243                                           zsphere   nds[36572]  107411 107416 107421 107426 107431 107436 107441 107446 107451 107456 ... 
    2017-08-03 13:02:05.501 INFO  [1800773] [GScene::compareMeshes_GMeshBB@527] GScene::compareMeshes_GMeshBB num_meshes 35 cut 0.1 bbty CSG_BBOX_PARSURF num_discrepant 9 frac 0.257143


Material match 
----------------

::

    op --j1707 --tracer --gltf 3

    2017-08-03 13:02:05.502 INFO  [1800773] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GSurfaceLib TRIGGERED A CLOSE  shortname []
    2017-08-03 13:02:05.502 INFO  [1800773] [GPropertyLib::close@384] GPropertyLib::close type GSurfaceLib buf 15,2,39,4
    Assertion failed: (ana.x == tri.x && "imat should match"), function lookupBoundarySpec, file /Users/blyth/opticks/ggeo/GScene.cc, line 820.
    /Users/blyth/opticks/bin/op.sh: line 669: 67964 Abort trap: 6           /usr/local/opticks/lib/OTracerTest --j1707 --tracer --gltf 3
    /Users/blyth/opticks/bin/op.sh RC 134



::

     813 std::string GScene::lookupBoundarySpec( const GSolid* node, const nd* n) const
     814 {
     815     unsigned tri_boundary = node->getBoundary();    // get the just transferred tri_boundary 
     816 
     817     guint4 tri = m_tri_bndlib->getBnd(tri_boundary);
     818     guint4 ana = m_tri_bndlib->parse( n->boundary.c_str());  // NO SURFACES
     819 
     820  
     821     //assert( ana.x == tri.x && "imat should match");  
     822     //assert( ana.w == tri.w && "omat should match");
     823 
     824     std::string ana_spec = m_tri_bndlib->shortname(ana);
     825     std::string tri_spec = m_tri_bndlib->shortname(tri);
     826     std::string spec = tri_spec ;
     827     
     828     
     829     
     830     if( !(ana.x == tri.x && ana.w == tri.w) )
     831     {                 
     832          LOG(warning) << "GScene::lookupBoundarySpec ana/tri imat/omat MISMATCH "
     833                       << " tri " << tri.description()
     834                       << " ana " << ana.description()
     835                       << " tri_spec " << tri_spec
     836                       << " ana_spec " << ana_spec
     837                       << " spec " << spec
     838                       ;
     839     }




Analytic j1707 with 50k : op --j1707 --tracer --gltf 3
---------------------------------------------------------


Works, but lots of dumping of large torus residuals when viewing from afar::

     ireal 2 i 0 root 26564.4 residual 40554.1  dis12 ( 66281.2 -405509 ) h 12.9854  pqr (169450 1.38834e+10 -2.3459e+12 )  j g/j (-16528.1 101419 )  
     ireal 2 i 0 root 27679.7 residual 3.19585e+08  dis12 ( 546942 -1.49804e+06 ) h 1.15459  pqr (480221 2.62488e+11 -3.5162e+11 )  j g/j (-136735 374510 )  
     ireal 2 i 0 root 27679.7 residual 3.19585e+08  dis12 ( 546942 -1.49804e+06 ) h 1.15459  pqr (480221 2.62488e+11 -3.5162e+11 )  j g/j (-136735 374510 )  
     ireal 2 i 0 root 26371.2 residual 8.44554e+07  dis12 ( 1.11988e+06 -1.27232e+06 ) h 3.33796  pqr (76813.7 3.57686e+11 -3.98911e+12 )  j g/j (-279967 318083 )  
     ireal 2 i 0 root 26401.5 residual 5.59586e+06  dis12 ( 979883 -1.17579e+06 ) h 10.8617  pqr (97881.2 2.90423e+11 -3.4267e+13 )  j g/j (-244941 293977 )  
     ireal 2 i 0 root 26547.2 residual 886052  dis12 ( 489099 -662001 ) h 10.1987  pqr (86361.4 8.28061e+10 -8.61433e+12 )  j g/j (-122249 165526 )  
     ireal 2 i 0 root 26329.8 residual 1.12956e+09  dis12 ( 1.3067e+06 -1.50284e+06 ) h 1.15268  pqr (104983 4.93695e+11 -6.61956e+11 )  j g/j (-326675 375710 )  



* maybe using OptiX selector : to swap the torus for smth simpler (cylinder ring, diff of cones) from afar ?

  * not near cathode ? so probably no effect on results, but big performance effect 






