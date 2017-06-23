Analytic Geometry Shakedown
===============================


Issue : flickery ray trace in rib shape
-------------------------------------------

* ~/opticks_refs/tgltf_looking_up_poke_thru_ribs.png


Issue : upward yellow cone lid photons think start in Ac
----------------------------------------------------------

* Many ~5% photons (all upward cone) think they are in acrylic.
* BUT, the yellow photons are spread around, not all pointing at poke thru ribs

::

    In [4]: print a.mat[:10]
    .                             1:gltf 
    .                             100000         1.00 
    0000           343231        0.460       45953      [6 ] Gd Ac LS Ac MO Ac
    0001          aa33231        0.047        4728      [7 ] Gd Ac LS Ac Ac ES ES
    0002               11        0.044        4396      [2 ] Gd Gd
    0003          3432311        0.026        2563      [7 ] Gd Gd Ac LS Ac MO Ac
    0004          5d43231        0.024        2406      [7 ] Gd Ac LS Ac MO Vm Bk
    0005               33        0.024        2385      [2 ] Ac Ac
    0006       4323233133        0.023        2345      [10] Ac Ac Gd Ac Ac LS Ac LS Ac MO
    0007          aa34231        0.018        1819      [7 ] Gd Ac LS MO Ac ES ES
    0008       3432323133        0.015        1461      [10] Ac Ac Gd Ac LS Ac LS Ac MO Ac
    0009         5de43231        0.011        1129      [8 ] Gd Ac LS Ac MO Py Vm Bk
    .                             100000         1.00 



Approach ? Decide to implement recursive geo selection to onion in on the problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Simplify... test with just the GdLS


* succeed to reproduce with, 2 volumes (presumably polycone concidence issue)
  (need an outer)

::

    export OPTICKS_QUERY="range:3158:3160"   # 3158+3159
    #export OPTICKS_QUERY="index:3159,depth:2"


::

    [2017-06-23 19:50:29,366] p36145 {/Users/blyth/opticks/ana/OpticksQuery.py:75} INFO - index found at depth 14 
    [2017-06-23 19:50:29,367] p36145 {/Users/blyth/opticks/analytic/treebase.py:216} INFO - selected index  3159 depth 14 name /dd/Geometry/AD/lvIAV#pvGDS0xbf6ab00 mat GdDopedLS
    [2017-06-23 19:50:29,387] p36145 {/Users/blyth/opticks/analytic/treebase.py:501} INFO - apply_selection OpticksQuery index:3159,depth:2 range [] index 3159 depth 2   Node.selected_count 1 


* ~/opticks_refs/tachyon_reflection_from_top_3159.png 

::

    3157      3156 [ 11:   0/ 520]    3 ( 0)              __dd__Geometry__AD__lvOAV0xbf1c760  oav0xc2ed7c8
    3158      3157 [ 12:   0/   3]   35 ( 0)               __dd__Geometry__AD__lvLSO0xc403e40  lso0xc028a38
    3159      3158 [ 13:   0/  35]    2 ( 0)                __dd__Geometry__AD__lvIAV0xc404ee8  iav0xc346f90
    3160      3159 [ 14:   0/   2]    0 ( 0)                 __dd__Geometry__AD__lvGDS0xbf6cbb8  gds0xc28d3f0
    3161      3160 [ 14:   1/   2]    0 ( 0)                 __dd__Geometry__AdDetails__lvOcrGdsInIav0xbf6dd58  OcrGdsInIav0xc405b10
    3162      3161 [ 13:   1/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvIavTopHub0xc129d88  IavTopHub0xc405968
    3163      3162 [ 13:   2/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflBotClp0xc407eb0  CtrGdsOflBotClp0xbf5dec0
    3164      3163 [ 13:   3/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflTfbInLso0xbfa0728  CtrGdsOflTfbInLso0xbfa2d30
    3165      3164 [ 13:   4/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflInLso0xc28cc88  CtrGdsOflInLso0xbfa1178
    3166      3165 [ 13:   5/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOcrGdsPrt0xc352630  OcrGdsPrt0xc352518
    3167      3166 [ 13:   6/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvCtrGdsOflBotClp0xc407eb0  CtrGdsOflBotClp0xbf5dec0
    3168      3167 [ 13:   7/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOcrGdsTfbInLso0xc3529c0  OcrGdsTfbInLso0xbfa2370
    3169      3168 [ 13:   8/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOcrGdsInLso0xc353990  OcrGdsInLso0xbfa2190
    3170      3169 [ 13:   9/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOavBotRib0xc353d30  OavBotRib0xbfaafe0
    3171      3170 [ 13:  10/  35]    0 ( 0)                __dd__Geometry__AdDetails__lvOavBotRib0xc353d30  OavBotRib0xbfaafe0



::

    In [12]: c.mesh.csg.dump(detailed=True)
    [2017-06-23 20:29:35,658] p37472 {/Users/blyth/opticks/analytic/csg.py:783} INFO - CSG.dump name:gds0xc28d3f0
    un(cy,un(co,cy) height:1 totnodes:3 ) height:2 totnodes:7 
     union;gds0xc28d3f0                                : abc = CSG("union", left=a, right=bc) 
        cylinder;gds_cyl0xc570d78_outer                : a = CSG("cylinder", param = [0.000,0.000,0.000,1550.000],param1 = [-1535.000,1535.000,0.000,0.000]) 
        union;gds_polycone0xc404f40_uniontree          : bc = CSG("union", left=b, right=c) 
           cone;gds_polycone0xc404f40_zp_1             : b = CSG("cone", param = [1520.000,3070.000,75.000,3145.729],param1 = [0.000,0.000,0.000,0.000]) 
           cylinder;gds_polycone0xc404f40_zp_2         : c = CSG("cylinder", param = [0.000,0.000,0.000,75.000],param1 = [3145.729,3159.440,0.000,0.000]) 

        un abc            
    cy a         un bc    
            co b     cy c



::

    In [15]: c.mesh.csg.dump_tboolean("gds")


    tboolean-gds(){ TESTCONFIG=$($FUNCNAME-) tboolean-- $* ; }
    tboolean-gds-(){ $FUNCNAME- | python $* ; } 
    tboolean-gds--(){ cat << EOP

    outdir = "$TMP/$FUNCNAME"
    obj_ = "$(tboolean-testobject)"
    con_ = "$(tboolean-container)"

    import logging
    log = logging.getLogger(__name__)
    from opticks.ana.base import opticks_main
    from opticks.analytic.csg import CSG  
    args = opticks_main()

    CSG.boundary = obj_
    CSG.kwa = dict(verbosity="1")

    a = CSG("cylinder", param = [0.000,0.000,0.000,1550.000],param1 = [-1535.000,1535.000,0.000,0.000])

    b = CSG("cone", param = [1520.000,3070.000,75.000,3145.729],param1 = [0.000,0.000,0.000,0.000])  # r1,z1,r2,z2  
    c = CSG("cylinder", param = [0.000,0.000,0.000,75.000],param1 = [3145.729,3159.440,0.000,0.000])
    bc = CSG("union", left=b, right=c)
    bc.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[0.000,0.000,-1535.000,1.000]]

    abc = CSG("union", left=a, right=bc)

    obj = abc

    con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=con_ , poly="HY", level="5" )
    CSG.Serialize([con, obj], outdir )


    EOP
    }




::

      519     <tube aunit="deg" deltaphi="360" lunit="mm" name="gds_cyl0xc570d78" rmax="1550" rmin="0" startphi="0" z="3070"/>
      520     <polycone aunit="deg" deltaphi="360" lunit="mm" name="gds_polycone0xc404f40" startphi="0">
      521       <zplane rmax="1520" rmin="0" z="3070"/>
      522       <zplane rmax="75" rmin="0" z="3145.72924106399"/>
      523       <zplane rmax="75" rmin="0" z="3159.43963177189"/>
      524     </polycone>
      525     <union name="gds0xc28d3f0">
      526       <first ref="gds_cyl0xc570d78"/>
      527       <second ref="gds_polycone0xc404f40"/>
      528       <position name="gds0xc28d3f0_pos" unit="mm" x="0" y="0" z="-1535"/>
      529     </union>



