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
  
::

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




Cone problem ?
~~~~~~~~~~~~~~~~~

::

    [2017-08-02 21:02:37,503] p52930 {/Users/blyth/opticks/analytic/csg.py:548} INFO - MakeEllipsoid axyz array([ 249.,  249.,  179.], dtype=float32) scale array([ 1.3911,  1.3911,  1.    ], dtype=float32) 
    [2017-08-02 21:02:37,504] p52930 {/Users/blyth/opticks/analytic/treebuilder.py:34} WARNING - balancing trees of this structure not implemented
    Traceback (most recent call last):
      File "/Users/blyth/opticks/bin/gdml2gltf.py", line 28, in <module>
        sc = gdml2gltf_main( args )
      File "/Users/blyth/opticks/analytic/sc.py", line 474, in gdml2gltf_main
        tg = sc.add_tree_gdml( tree.root, maxdepth=0)
      File "/Users/blyth/opticks/analytic/sc.py", line 359, in add_tree_gdml
        tg = build_r(target)
      File "/Users/blyth/opticks/analytic/sc.py", line 346, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 346, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 346, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 346, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 346, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 346, in build_r
        ch = build_r(child, depth+1)
      File "/Users/blyth/opticks/analytic/sc.py", line 343, in build_r
        nd = self.add_node_gdml(node, depth)
      File "/Users/blyth/opticks/analytic/sc.py", line 246, in add_node_gdml
        csg = self.translate_lv( node.lv, self.maxcsgheight )
      File "/Users/blyth/opticks/analytic/sc.py", line 286, in translate_lv
        rawcsg = solid.as_ncsg()
      File "/Users/blyth/opticks/analytic/gdml.py", line 186, in as_ncsg
        left = self.first.as_ncsg()
      File "/Users/blyth/opticks/analytic/gdml.py", line 744, in as_ncsg
        prims = self.prims()
      File "/Users/blyth/opticks/analytic/gdml.py", line 714, in prims
        assert z2 > z1, (z2,z1)
    AssertionError: (-75.8755078663876, -15.8745078663875)
    /Users/blyth/opticks/bin/op.sh RC 1


