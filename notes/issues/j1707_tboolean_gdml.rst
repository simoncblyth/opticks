tboolean-gdml geocache selection ?
======================================


Volumes from one geometry are being looked for in wrong geocache ?

::

    simon:tests blyth$ t tboolean-ntc-
    tboolean-ntc- () 
    { 
        tboolean-gdml- $TMP/$FUNCNAME --gsel /dd/Geometry/PoolDetails/lvNearTopCover0x
    }


::

    2473 tboolean-gdml-()
    2474 {
    2475     local csgpath=$1
    2476     shift
    2477     python $(tboolean-gdml-translator) \
    2478           --csgpath $csgpath \
    2479           --container $(tboolean-container)  \
    2480           --testobject $(tboolean-testobject) \
    2481           $*
    2482 }
    2483 tboolean-gdml-translator(){ echo $(opticks-home)/analytic/translate_gdml.py ; }
    2484 tboolean-gdml-translator-vi(){ vi $(tboolean-gdml-translator); }



::

    simon:tests blyth$ tboolean-ntc-
    args: /Users/blyth/opticks/analytic/translate_gdml.py --csgpath /tmp/blyth/opticks/tboolean-ntc- --container Rock//perfectAbsorbSurface/Vacuum --testobject Vacuum///GlassSchottF2 --gsel /dd/Geometry/PoolDetails/lvNearTopCover0x
    [2017-08-16 18:33:10,440] p11422 {/Users/blyth/opticks/analytic/translate_gdml.py:79} INFO -  gsel:/dd/Geometry/PoolDetails/lvNearTopCover0x gidx:0 gmaxnode:0 gmaxdepth:0 
    [2017-08-16 18:33:10,440] p11422 {/Users/blyth/opticks/analytic/gdml.py:954} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/juno1707/g4_00.gdml 
    [2017-08-16 18:33:10,756] p11422 {/Users/blyth/opticks/analytic/gdml.py:968} INFO - wrapping gdml element  
    [2017-08-16 18:33:10,919] p11422 {/Users/blyth/opticks/analytic/gdml.py:1016} INFO - vv 46 vvs 35 
    ^CTraceback (most recent call last):
      File "/Users/blyth/opticks/analytic/translate_gdml.py", line 83, in <module>
        tree = Tree(gdml.world)
      File "/Users/blyth/opticks/analytic/treebase.py", line 527, in __init__
        self.root = self.create_r(self.base, ancestors)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 556, in create_r
        self.create_r(child, volpath)
      File "/Users/blyth/opticks/analytic/treebase.py", line 552, in create_r
        node = Node.create(volpath, lvtype=self.lvtype, pvtype=self.pvtype, postype=self.postype )
      File "/Users/blyth/opticks/analytic/treebase.py", line 147, in create
        node.posXYZ = node.pv.find_(postype) if node.pv is not None else None
      File "/Users/blyth/opticks/analytic/gdml.py", line 102, in find_
        e = self.elem.find(expr) 
      File "lxml.etree.pyx", line 1448, in lxml.etree._Element.find (src/lxml/lxml.etree.c:51339)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/lxml/_elementpath.py", line 281, in find
        it = iterfind(elem, path, namespaces)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/lxml/_elementpath.py", line 271, in iterfind
        selector = _build_path_iterator(path, namespaces)
      File "/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/lxml/_elementpath.py", line 234, in _build_path_iterator
        return _cache[(path, namespaces and tuple(sorted(namespaces.items())) or None)]
    KeyboardInterrupt




After swithing IDPATH pick up from correct geocache::

    simon:analytic blyth$ tboolean-;tboolean-ntc-
    args: /Users/blyth/opticks/analytic/translate_gdml.py --csgpath /tmp/blyth/opticks/tboolean-ntc- --container Rock//perfectAbsorbSurface/Vacuum --testobject Vacuum///GlassSchottF2 --gsel /dd/Geometry/PoolDetails/lvNearTopCover0x
    [2017-08-16 18:46:08,372] p11910 {/Users/blyth/opticks/analytic/translate_gdml.py:65} INFO -  gsel:/dd/Geometry/PoolDetails/lvNearTopCover0x gidx:0 gmaxnode:0 gmaxdepth:0 
    [2017-08-16 18:46:08,372] p11910 {/Users/blyth/opticks/analytic/gdml.py:953} INFO - gdmlpath defaulting to OPTICKS_GDMLPATH /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml which is derived by opticks.ana.base from the IDPATH input envvar 
    [2017-08-16 18:46:08,372] p11910 {/Users/blyth/opticks/analytic/gdml.py:955} INFO - parsing gdmlpath /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.gdml 
    [2017-08-16 18:46:08,409] p11910 {/Users/blyth/opticks/analytic/gdml.py:969} INFO - wrapping gdml element  
    [2017-08-16 18:46:08,441] p11910 {/Users/blyth/opticks/analytic/gdml.py:1017} INFO - vv 249 vvs 249 
    [2017-08-16 18:46:09,373] p11910 {/Users/blyth/opticks/analytic/treebase.py:393} INFO - found 1 nodes with lvn(LV name prefix) starting:/dd/Geometry/PoolDetails/lvNearTopCover0x 
    [2017-08-16 18:46:09,373] p11910 {/Users/blyth/opticks/analytic/treebase.py:252} INFO - rprogeny numProgeny:1 (maxnode:0 maxdepth:0 skip:{'count': 0, 'depth': 0, 'total': 0} ) 
    [2017-08-16 18:46:09,373] p11910 {/Users/blyth/opticks/analytic/translate_gdml.py:74} INFO -  subtree 1 nodes 
    [2017-08-16 18:46:09,374] p11910 {/Users/blyth/opticks/analytic/translate_gdml.py:81} INFO - [ 0] converting solid 'near_top_cover_box0xc23f970' 
    [2017-08-16 18:46:09,375] p11910 {/Users/blyth/opticks/analytic/translate_gdml.py:87} WARNING - skipping transform
    [2017-08-16 18:46:09,375] p11910 {/Users/blyth/opticks/analytic/translate_gdml.py:94} INFO - cn.meta {'lvname': '/dd/Geometry/PoolDetails/lvNearTopCover0xc137060', 'soname': 'near_top_cover_box0xc23f970', 'verbosity': '0', 'poly': 'IM', 'height': 4, 'depth': 3, 'pvname': '/dd/Geometry/Sites/lvNearHallTop#pvNearTopCover0xc23f9b8', 'nchild': 0, 'pdigest': 'c166e2c6fcc6492d15e51306e64711cf', 'resolution': '20', 'digest': 'a78671d760574b26e069270e7fb7e992', 'treeindex': 3} 


    near_top_cover_box0xc23f970
    di(di(di(di(bo,bo),bo),bo),bo) height:4 totnodes:31 
                                 di    
                         di          bo
                 di          bo        
         di          bo                
     bo      bo                        [2017-08-16 18:46:09,376] p11910 {/Users/blyth/opticks/analytic/csg.py:430} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tboolean-ntc- 
    analytic=1_csgpath=/tmp/blyth/opticks/tboolean-ntc-_name=tboolean-ntc-_mode=PyCsgInBox
    simon:analytic blyth$ 
    simon:analytic blyth$ 
    simon:analytic blyth$ 


