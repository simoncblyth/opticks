NScanTest zero crossings
============================

not outside ?
------------------

* NScanTest /tmp/blyth/opticks/tgltf/extras/196  : small z range, so 10% cage and integer mm rounding fails to get outside

Fixed the "not outside" problem of scan line SDFs starting or ending in negative territory inside geometry 
by preventing NScan::init_cage coming up with too small a cage when thin along an axis.



Tree level z-nudging of unions of cylinders and cones fixes most odd nzero crossings
-----------------------------------------------------------------------------------------

::

    opticks-tscan-all

    2017-06-27 12:18:56.883 INFO  [1370914] [main@91]  autoscan non-zero counts trees 249 mmstep 0.1

     nzero    0 count   43 frac 0.173    ## scanline misses geometry, TODO improve aim

     nzero    1 count    5 frac 0.020    ## thin geometry : probably not an issue, as no prob for raytrace  

     nzero    2 count  179 frac 0.719    ## expected crossings 
     nzero    4 count   20 frac 0.080

     nzero    5 count    1 frac 0.004    ## wierd union of cylinders with cone cut

     nzero 11195 count    1 frac 0.004   ## TO INVESTGATE : involves convexpolyhedron 



::

    simon:opticks_refs blyth$ opticks-tscan-all
    opticks-tscan : scanning /tmp/blyth/opticks/tgltf/extras//
    2017-06-27 12:18:50.778 INFO  [1370914] [NCSG::Deserialize@958] NCSG::Deserialize VERBOSITY 0 basedir /tmp/blyth/opticks/tgltf/extras// txtpath /tmp/blyth/opticks/tgltf/extras//csg.txt nbnd 249
    ...

    2017-06-27 12:18:51.646 INFO  [1370914] [Primitives::dump@496] before znudge treedir /tmp/blyth/opticks/tgltf/extras//145 typmsk union cylinder  nprim 6 znudge_count 0 verbosity 0
    dump_qty : model frame r1/r2 (local) 
            [ 7:cy]       r1    930.000       r2    930.000
            [ 8:cy]                           r1   1015.000       r2   1015.000
            [ 9:cy]                                               r1   1010.000       r2   1010.000
            [10:cy]                                                                   r1    930.000       r2    930.000
            [ 5:cy]                                                                                       r1    380.000       r2    380.000
            [ 6:cy]                                                                                                           r1    400.300       r2    400.300
    dump_qty : bbox.min/max.z (globally transformed) 
            [ 7:cy] bb.min.z    -92.500 bb.max.z     92.500
            [ 8:cy]                     bb.min.z     92.500 bb.max.z    107.500
            [ 9:cy]                                         bb.min.z    107.500 bb.max.z    127.500
            [10:cy]                                                             bb.min.z    127.500 bb.max.z    177.500
            [ 5:cy]                                                                                 bb.min.z    177.500 bb.max.z    187.500
            [ 6:cy]                                                                                                     bb.min.z    187.500 bb.max.z    207.500
    dump_qty : bbox (globally transformed) 
            [ 7:cy] mi  (-930.00 -930.00  -92.50)  mx  ( 930.00  930.00   92.50) 
            [ 8:cy] mi  (-1015.00 -1015.00   92.50)  mx  (1015.00 1015.00  107.50) 
            [ 9:cy] mi  (-1010.00 -1010.00  107.50)  mx  (1010.00 1010.00  127.50) 
            [10:cy] mi  (-930.00 -930.00  127.50)  mx  ( 930.00  930.00  177.50) 
            [ 5:cy] mi  (-423.74 -274.40  177.50)  mx  ( 336.26  485.60  187.50) 
            [ 6:cy] mi  (-444.04 -294.70  187.50)  mx  ( 356.56  505.90  207.50) 
    dump_joins
     ja:         [ 7:cy] jb:         [ 8:cy] za:     92.500 zb:     92.500 join           COINCIDENT ra:    930.000 rb:   1015.000
     ja:         [ 8:cy] jb:         [ 9:cy] za:    107.500 zb:    107.500 join           COINCIDENT ra:   1015.000 rb:   1010.000
     ja:         [ 9:cy] jb:         [10:cy] za:    127.500 zb:    127.500 join           COINCIDENT ra:   1010.000 rb:    930.000
     ja:         [10:cy] jb:         [ 5:cy] za:    177.500 zb:    177.500 join           COINCIDENT ra:    930.000 rb:    380.000
     ja:         [ 5:cy] jb:         [ 6:cy] za:    187.500 zb:    187.500 join           COINCIDENT ra:    380.000 rb:    400.300


    2017-06-27 12:18:51.646 INFO  [1370914] [Primitives::dump@496] after znudge treedir /tmp/blyth/opticks/tgltf/extras//145 typmsk union cylinder  nprim 6 znudge_count 5 verbosity 0
    dump_qty : model frame r1/r2 (local) 
            [ 7:cy]       r1    930.000       r2    930.000
            [ 8:cy]                           r1   1015.000       r2   1015.000
            [ 9:cy]                                               r1   1010.000       r2   1010.000
            [10:cy]                                                                   r1    930.000       r2    930.000
            [ 5:cy]                                                                                       r1    380.000       r2    380.000
            [ 6:cy]                                                                                                           r1    400.300       r2    400.300
    dump_qty : bbox.min/max.z (globally transformed) 
            [ 7:cy] bb.min.z    -92.500 bb.max.z     93.500
            [ 8:cy]                     bb.min.z     92.500 bb.max.z    107.500
            [ 9:cy]                                         bb.min.z    106.500 bb.max.z    127.500
            [10:cy]                                                             bb.min.z    126.500 bb.max.z    177.500
            [ 5:cy]                                                                                 bb.min.z    176.500 bb.max.z    188.500
            [ 6:cy]                                                                                                     bb.min.z    187.500 bb.max.z    207.500
    dump_qty : bbox (globally transformed) 
            [ 7:cy] mi  (-930.00 -930.00  -92.50)  mx  ( 930.00  930.00   93.50) 
            [ 8:cy] mi  (-1015.00 -1015.00   92.50)  mx  (1015.00 1015.00  107.50) 
            [ 9:cy] mi  (-1010.00 -1010.00  106.50)  mx  (1010.00 1010.00  127.50) 
            [10:cy] mi  (-930.00 -930.00  126.50)  mx  ( 930.00  930.00  177.50) 
            [ 5:cy] mi  (-423.74 -274.40  176.50)  mx  ( 336.26  485.60  188.50) 
            [ 6:cy] mi  (-444.04 -294.70  187.50)  mx  ( 356.56  505.90  207.50) 
    dump_joins
     ja:         [ 7:cy] jb:         [ 8:cy] za:     93.500 zb:     92.500 join              OVERLAP ra:    930.000 rb:   1015.000
     ja:         [ 8:cy] jb:         [ 9:cy] za:    107.500 zb:    106.500 join              OVERLAP ra:   1015.000 rb:   1010.000
     ja:         [ 9:cy] jb:         [10:cy] za:    127.500 zb:    126.500 join              OVERLAP ra:   1010.000 rb:    930.000
     ja:         [10:cy] jb:         [ 5:cy] za:    177.500 zb:    176.500 join              OVERLAP ra:    930.000 rb:    380.000
     ja:         [ 5:cy] jb:         [ 6:cy] za:    188.500 zb:    187.500 join              OVERLAP ra:    380.000 rb:    400.300






lvidx 29 : wierd nzero 5
-----------------------------

* ~/opticks_refs/opticks_tscan_29_nzero_5_OcrGdsPrt.png
* ~/opticks_refs/opticks_tscan_29_ok_without_cone_subtraction.png

Without the cone subtraction the znudge works to uncoincide it.


::

   opticks-tscan 29 
   opticks-tbool 29      
   opticks-tbool-vi 29   # edit to just show cone, shows its extremly flat  


::

     62 tbool29--(){ cat << EOP
     63 
     64 import logging
     65 log = logging.getLogger(__name__)
     66 from opticks.ana.base import opticks_main
     67 from opticks.analytic.csg import CSG  
     68 args = opticks_main(csgpath="$TMP/tbool/29")
     69 
     70 CSG.boundary = args.testobject
     71 CSG.kwa = dict(verbosity="0", poly="IM", resolution="20")
     72 
     73 
     74 
     75 
     76 a = CSG("cylinder", param = [0.000,0.000,0.000,100.000],param1 = [0.000,160.000,0.000,0.000])
     77 b = CSG("cylinder", param = [0.000,0.000,0.000,150.000],param1 = [160.000,185.000,0.000,0.000])
     78 ab = CSG("union", left=a, right=b)
     79 
     80 c = CSG("cone", param = [1520.393,0.000,100.000,74.440],param1 = [0.000,0.000,0.000,0.000])
                               //   r1     z1    r2      z2

     81 c.transform = [[1.000,0.000,0.000,0.000],[0.000,1.000,0.000,0.000],[0.000,0.000,1.000,0.000],[-516.623,1247.237,37.220,1.000]]
     82 abc = CSG("difference", left=ab, right=c)
     83 
     84 
     85 
     86 
     87 
     88 obj = ab
     89 #obj = c
     90 
     91 con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=args.container , poly="HY", level="5" )
     92 CSG.Serialize([con, obj], args.csgpath )
     93 
     94 EOP
     95 }



Visualizing problem geometry
-------------------------------

tgltf-tt (sc.py) now standardly dumps tboolN.bash scripts into extras, so to 
view some geometry, use *opticks-tbool N* when N is the *lvid* index::

::

    simon:opticks blyth$ t opticks-tbool
    opticks-tbool () 
    { 
        local msg="$FUNCNAME :";
        local lvid=${1:-0};
        local path=$TMP/tgltf/extras/${lvid}/tbool${lvid}.bash;
        echo $msg sourcing $path lvid $lvid;
        [ ! -f $path ] && echo $msg no such path && return;
        . $path;
        tbool${lvid}
    }


Visualize nzero 3,5,7
------------------------

::
     opticks-tbool 143   # etc..


     nzero    3 count    7 frac 0.0281125
     i  105 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/143      soname                          GdsOfl0xbf73918 tag    [ 0:un] msg   cy-cy big flat one, with small other 
     i  180 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/68       soname                       SstTopHub0xc2643d8 tag    [ 0:un] msg   cy-cy (flange like)
     i  194 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/54       soname                 headon-pmt-assy0xbf55198 tag    [ 0:un] msg   cy-cy (torch shape, speckles visible in raytrace)
     i  206 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/42       soname                             oav0xc2ed7c8 tag    [ 0:un] msg   cy-co-cy ? with lip 
     i  211 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/37       soname                             lso0xc028a38 tag    [ 0:un] msg   cy-co-cy
     i  222 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/26       soname                 CtrGdsOflBotClp0xbf5dec0 tag    [ 0:un] msg   cy-cy   squat   
     i  226 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/22       soname                             gds0xc28d3f0 tag    [ 0:un] msg   cy-co-cy

     nzero    5 count    1 frac 0.00401606
     i  219 nzero    5 NScanTest /tmp/blyth/opticks/tgltf/extras/29       soname                       OcrGdsPrt0xc352518 tag    [ 0:di] msg  wierd one, ~3 cy with visibly sliced cut 

     nzero    7 count    1 frac 0.00401606
     i  103 nzero    7 NScanTest /tmp/blyth/opticks/tgltf/extras/145      soname                 OflTnkContainer0xc17cf50 tag    [ 0:un] msg   stack of plates cy




With some uncoincidencing
--------------------------

prim/prim uncoincidencing only manages to fix a few... 
need to be able to uncoincide with one of em a union ?

* hmm will mostly be pure uniontree, so can order 
  all the primitives in z and look for bbox coincidence one 
  by one


::

    2017-06-26 19:08:49.914 INFO  [1278367] [main@91]  autoscan non-zero counts trees 249 mmstep 0.1
     nzero    0 count   43 frac 0.1727
     nzero    1 count    5 frac 0.0201
     nzero    2 count  171 frac 0.6867
     nzero    3 count    7 frac 0.0281
     nzero    4 count   22 frac 0.0884
     nzero 11195 count    1 frac 0.0040


Central x,y -z to +z scanline
-----------------------------------

* nzero 2 and 4 are expected crossings 

* nzero 0, mostly differences, the single scanline failed to find geometry... need multiple scan lines
* nzero 1, very thin geometry : probably not an issue, as ray trace intersects is not bothered by scan step size issues
* nzero 3, unions : extra internal surfaces is a major issue that needs fixing


::

    delta:ana blyth$ NScanTest /tmp/blyth/opticks/tgltf/extras
    2017-06-26 10:36:55.130 INFO  [1112314] [NCSG::Deserialize@928] NCSG::Deserialize VERBOSITY 0 basedir /tmp/blyth/opticks/tgltf/extras txtpath /tmp/blyth/opticks/tgltf/extras/csg.txt nbnd 249
    2017-06-26 10:36:55.229 INFO  [1112314] [NCSG::DeserializeTrees@897] NCSG::DeserializeTrees /tmp/blyth/opticks/tgltf/extras found trees : 249
    2017-06-26 10:36:55.229 INFO  [1112314] [main@55]  NScanTest autoscan trees  basedir /tmp/blyth/opticks/tgltf/extras ntree 249 verbosity 0
    ...
    2017-06-26 10:36:58.068 INFO  [1112314] [main@91]  autoscan non-zero counts trees 249 mmstep 0.1
     nzero    0 count   43 frac 0.172691
     nzero    1 count    5 frac 0.0200803
     nzero    2 count  167 frac 0.670683
     nzero    3 count    7 frac 0.0281125
     nzero    4 count   24 frac 0.0963855
     nzero    5 count    1 frac 0.00401606
     nzero    7 count    1 frac 0.00401606
     nzero 11195 count    1 frac 0.00401606

     nzero    0 count   43 frac 0.172691
     i   17 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/231      soname       lvOutOutWaterPipeNear_Tub0xce5b598 tag    [ 0:di] msg 
     i   18 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/230      soname        lvOutInWaterPipeNear_Tub0xce5b3f0 tag    [ 0:di] msg 
     i   29 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/219      soname                 out_Sid_ver_rib0xc212138 tag    [ 0:di] msg 
     i   31 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/217      soname                 out_bot_ver_rib0xcd573e8 tag    [ 0:di] msg 
     i   38 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/210      soname       lvInnOutWaterPipeNear_Tub0xc95a8a0 tag    [ 0:di] msg 
     i   39 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/209      soname        lvInnInWaterPipeNear_Tub0xc273850 tag    [ 0:di] msg 
     i   48 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/200      soname                 table_panel_box0xc00f558 tag    [ 0:in] msg 
     i   54 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/194      soname                   pmt-base-ring0xc401a00 tag    [ 0:di] msg 
     i   55 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/193      soname                    pmt-top-ring0xc2f0608 tag    [ 0:di] msg 
     i   60 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/188      soname                   MOFTTopFlange0xc047418 tag    [ 0:di] msg 
     i   63 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/185      soname                        MOFTTube0xc046b40 tag    [ 0:di] msg 
     i   68 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/180      soname                    MCBTopFlange0xc213a48 tag    [ 0:di] msg 
     i   70 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/178      soname                         MCBTube0xc20e0c0 tag    [ 0:di] msg 
     i   75 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/173      soname                    GDBTopFlange0xc20d820 tag    [ 0:di] msg 
     i   77 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/171      soname                         GDBTube0xc213f68 tag    [ 0:di] msg 
     i   86 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/162      soname            LSCalibTubAbvLidTub50xc17c6f8 tag    [ 0:di] msg 
     i   87 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/161      soname            LSCalibTubAbvLidTub40xc17c470 tag    [ 0:di] msg 
     i   88 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/160      soname            LSCalibTubAbvLidTub30xc17c220 tag    [ 0:di] msg 
     i   89 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/159      soname            LSCalibTubAbvLidTub20xc17bfc8 tag    [ 0:di] msg 
     i   90 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/158      soname            LSCalibTubAbvLidTub10xc17bd80 tag    [ 0:di] msg 
     i   98 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/150      soname          GdLSCalibTubAbvLidTub50xc341080 tag    [ 0:di] msg 
     i   99 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/149      soname          GdLSCalibTubAbvLidTub40xc340e28 tag    [ 0:di] msg 
     i  100 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/148      soname          GdLSCalibTubAbvLidTub30xc340bd0 tag    [ 0:di] msg 
     i  101 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/147      soname          GdLSCalibTubAbvLidTub20xc340980 tag    [ 0:di] msg 
     i  102 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/146      soname          GdLSCalibTubAbvLidTub10xc3406d8 tag    [ 0:di] msg 
     i  106 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/142      soname                       GdsOflTnk0xc3d5160 tag    [ 0:un] msg 
     i  107 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/141      soname                          LsoOfl0xc348ac0 tag    [ 0:un] msg 
     i  108 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/140      soname                       LsoOflTnk0xc17d928 tag    [ 0:un] msg 
     i  112 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/136      soname                 AcrylicCylinder0xc3d3830 tag    [ 0:di] msg 
     i  114 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/134      soname           NeutronShieldCylinder0xc3d3378 tag    [ 0:di] msg 
     i  115 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/133      soname             GammaShieldCylinder0xc3d30f0 tag    [ 0:di] msg 
     i  151 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/97       soname                     BearingRing0xbf778c8 tag    [ 0:di] msg 
     i  172 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/76       soname                    CtrLsoOflTfb0xc1797a8 tag    [ 0:di] msg 
     i  179 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/69       soname                SstTopCirRibBase0xc264f78 tag    [ 0:in] msg 
     i  183 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/65       soname                SstBotCirRibBase0xc26e2d0 tag    [ 0:di] msg 
     i  184 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/64       soname                       SsTBotHub0xc26d1d0 tag    [ 0:di] msg 
     i  186 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/62       soname                      BotRefHols0xc3cd380 tag    [ 0:in] msg 
     i  189 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/59       soname                   TopRefCutHols0xbf9bd50 tag    [ 0:in] msg 
     i  192 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/56       soname                RadialShieldUnit0xc3d7da8 tag    [ 0:in] msg 
     i  193 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/55       soname                headon-pmt-mount0xc2a7670 tag    [ 0:un] msg 
     i  200 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/48       soname                     AdPmtCollar0xc2c5260 tag    [ 0:di] msg 
     i  218 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/30       soname                  OcrGdsTfbInLso0xbfa2370 tag    [ 0:in] msg 
     i  221 nzero    0 NScanTest /tmp/blyth/opticks/tgltf/extras/27       soname               CtrGdsOflTfbInLso0xbfa2d30 tag    [ 0:di] msg 

     nzero    1 count    5 frac 0.0200803
     i  187 nzero    1 NScanTest /tmp/blyth/opticks/tgltf/extras/61       soname                BotRefGapCutHols0xc34bb28 tag    [ 0:in] msg 
     i  188 nzero    1 NScanTest /tmp/blyth/opticks/tgltf/extras/60       soname                   BotESRCutHols0xbfa7368 tag    [ 0:in] msg 
     i  190 nzero    1 NScanTest /tmp/blyth/opticks/tgltf/extras/58       soname                TopRefGapCutHols0xbf9cef8 tag    [ 0:in] msg 
     i  191 nzero    1 NScanTest /tmp/blyth/opticks/tgltf/extras/57       soname                   TopESRCutHols0xbf9de10 tag    [ 0:in] msg 
     i  205 nzero    1 NScanTest /tmp/blyth/opticks/tgltf/extras/43       soname                pmt-hemi-cathode0xc2f1ce8 tag    [ 0:un] msg 

     nzero    2 count  167 frac 0.670683

     nzero    3 count    7 frac 0.0281125
     i  105 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/143      soname                          GdsOfl0xbf73918 tag    [ 0:un] msg   cy-cy big flat one, with small other 
     i  180 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/68       soname                       SstTopHub0xc2643d8 tag    [ 0:un] msg   cy-cy (flange like)
     i  194 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/54       soname                 headon-pmt-assy0xbf55198 tag    [ 0:un] msg   cy-cy (torch shape, speckles visible in raytrace)
     i  206 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/42       soname                             oav0xc2ed7c8 tag    [ 0:un] msg   cy-co-cy ? with lip 
     i  211 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/37       soname                             lso0xc028a38 tag    [ 0:un] msg   cy-co-cy
     i  222 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/26       soname                 CtrGdsOflBotClp0xbf5dec0 tag    [ 0:un] msg   cy-cy   squat   
     i  226 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras/22       soname                             gds0xc28d3f0 tag    [ 0:un] msg   cy-co-cy

     nzero    4 count   24 frac 0.0963855

     nzero    5 count    1 frac 0.00401606
     i  219 nzero    5 NScanTest /tmp/blyth/opticks/tgltf/extras/29       soname                       OcrGdsPrt0xc352518 tag    [ 0:di] msg 

     nzero    7 count    1 frac 0.00401606
     i  103 nzero    7 NScanTest /tmp/blyth/opticks/tgltf/extras/145      soname                 OflTnkContainer0xc17cf50 tag    [ 0:un] msg 

     nzero 11195 count    1 frac 0.00401606
     i  182 nzero 11195 NScanTest /tmp/blyth/opticks/tgltf/extras/66       soname                 SstTopRadiusRib0xc271720 tag    [ 0:di] msg 
    delta:ana blyth$ 



extras/66 fails to load : problem with planes
-------------------------------------------------

* body writing of CSG code by CSG.write_tbool omits the planes...


::

    simon:issues blyth$ l /tmp/blyth/opticks/tgltf/extras/66/
    total 48
    -rw-r--r--  1 blyth  wheel   166 Jun 26 11:41 meta.json
    -rw-r--r--  1 blyth  wheel   528 Jun 26 11:41 nodes.npy
    -rw-r--r--  1 blyth  wheel   176 Jun 26 11:41 planes.npy
    -rw-r--r--  1 blyth  wheel  2701 Jun 26 11:41 tbool66.bash
    -rw-r--r--  1 blyth  wheel   400 Jun 26 11:41 transforms.npy
    -rw-r--r--  1 blyth  wheel  2673 Jun 26 11:29 tboolean.bash
    simon:issues blyth$ 

    simon:analytic blyth$ l /tmp/blyth/opticks/tbool/66/0/
    total 32
    -rw-r--r--  1 blyth  wheel    69 Jun 26 12:05 meta.json
    -rw-r--r--  1 blyth  wheel   144 Jun 26 12:05 nodes.npy
    -rw-r--r--  1 blyth  wheel  2167 Jun 26 12:05 tbool0.bash
    -rw-r--r--  1 blyth  wheel   144 Jun 26 12:05 transforms.npy

    simon:analytic blyth$ l /tmp/blyth/opticks/tbool/66/1/
    total 32
    -rw-r--r--  1 blyth  wheel    32 Jun 26 12:05 meta.json
    -rw-r--r--  1 blyth  wheel   528 Jun 26 12:05 nodes.npy
    -rw-r--r--  1 blyth  wheel  2683 Jun 26 12:05 tbool1.bash
    -rw-r--r--  1 blyth  wheel   400 Jun 26 12:05 transforms.npy
    simon:analytic blyth$ 




::

    simon:issues blyth$ opticks-tbool 66
    opticks-tbool : sourcing /tmp/blyth/opticks/tgltf/extras/66/tbool66.bash lvid 66
    args: 
    [2017-06-26 12:05:27,213] p97104 {/Users/blyth/opticks/analytic/csg.py:392} INFO - CSG.Serialize : writing 2 trees to directory /tmp/blyth/opticks/tbool/66 
    288 -rwxr-xr-x  1 blyth  staff  143804 Jun 25 18:41 /usr/local/opticks/lib/OKTest
    proceeding : /usr/local/opticks/lib/OKTest --animtimemax 20 --timemax 20 --geocenter --eye 1,0,0 --dbganalytic --test --testconfig analytic=1_csgpath=/tmp/blyth/opticks/tbool/66_name=66_mode=PyCsgInBox --torch --torchconfig type=sphere_photons=10000_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1000.000,1.000_source=0,0,0_target=0,0,1_time=0.1_radius=100_distance=400_zenithazimuth=0,1,0,1_material=GdDopedLS_wavelength=500 --torchdbg --tag 1 --cat tbool --save
    2017-06-26 12:05:27.477 INFO  [1137714] [OpticksDbg::postconfigure@49] OpticksDbg::postconfigure OpticksDbg  debug_photon  size: 0 elem: () other_photon  size: 0 elem: ()
    2017-06-26 12:05:27.647 INFO  [1137714] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-06-26 12:05:27.752 INFO  [1137714] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-06-26 12:05:27.835 INFO  [1137714] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-06-26 12:05:27.835 INFO  [1137714] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-06-26 12:05:27.835 INFO  [1137714] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-06-26 12:05:27.836 INFO  [1137714] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-06-26 12:05:27.841 INFO  [1137714] [GGeo::loadAnalyticPmt@772] GGeo::loadAnalyticPmt AnalyticPMTIndex 0 AnalyticPMTSlice ALL Path /usr/local/opticks/opticksdata/export/DayaBay/GPmt/0
    2017-06-26 12:05:27.849 WARN  [1137714] [GGeoTest::init@54] GGeoTest::init booting from m_ggeo 
    2017-06-26 12:05:27.849 WARN  [1137714] [GMaker::init@171] GMaker::init booting from cache
    2017-06-26 12:05:27.849 INFO  [1137714] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/0 index 0 version (null) existsdir 1
    2017-06-26 12:05:27.965 INFO  [1137714] [*GMergedMesh::load@632] GMergedMesh::load dir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 -> cachedir /usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMergedMesh/1 index 1 version (null) existsdir 1
    2017-06-26 12:05:27.969 INFO  [1137714] [GMaterialLib::postLoadFromCache@67] GMaterialLib::postLoadFromCache  nore 0 noab 0 nosc 0 xxre 0 xxab 0 xxsc 0 fxre 0 fxab 0 fxsc 0 groupvel 1
    2017-06-26 12:05:27.969 INFO  [1137714] [GMaterialLib::replaceGROUPVEL@552] GMaterialLib::replaceGROUPVEL  ni 38
    2017-06-26 12:05:27.969 INFO  [1137714] [GPropertyLib::getIndex@338] GPropertyLib::getIndex type GMaterialLib TRIGGERED A CLOSE  shortname [GdDopedLS]
    2017-06-26 12:05:27.970 INFO  [1137714] [GPropertyLib::close@384] GPropertyLib::close type GMaterialLib buf 38,2,39,4
    2017-06-26 12:05:27.973 INFO  [1137714] [GGeoTest::loadCSG@212] GGeoTest::loadCSG  csgpath /tmp/blyth/opticks/tbool/66 verbosity 0
    2017-06-26 12:05:27.973 INFO  [1137714] [NCSG::Deserialize@928] NCSG::Deserialize VERBOSITY 0 basedir /tmp/blyth/opticks/tbool/66 txtpath /tmp/blyth/opticks/tbool/66/csg.txt nbnd 2
    Assertion failed: (idx < m_num_planes), function import_planes, file /Users/blyth/opticks/opticksnpy/NCSG.cpp, line 708.
    /Users/blyth/opticks/bin/op.sh: line 619: 97334 Abort trap: 6           /usr/local/opticks/lib/OKTest --animtimemax 20 --timemax 20 --geocenter --eye 1,0,0 --dbganalytic --test --testconfig analytic=1_csgpath=/tmp/blyth/opticks/tbool/66_name=66_mode=PyCsgInBox --torch --torchconfig type=sphere_photons=10000_frame=-1_transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1000.000,1.000_source=0,0,0_target=0,0,1_time=0.1_radius=100_distance=400_zenithazimuth=0,1,0,1_material=GdDopedLS_wavelength=500 --torchdbg --tag 1 --cat tbool --save
    /Users/blyth/opticks/bin/op.sh RC 134
    simon:issues blyth$ 









tree level uncoincidence ?
-----------------------------


::

    simon:sysrap blyth$ opticks-tscan /
    opticks-tscan : scanning /tmp/blyth/opticks/tgltf/extras//
    017-06-26 20:22:50.457 INFO  [1304562] [main@55]  NScanTest autoscan trees  basedir /tmp/blyth/opticks/tgltf/extras// ntree 249 verbosity 0
    ...
    2017-06-26 20:22:53.438 INFO  [1304562] [main@91]  autoscan non-zero counts trees 249 mmstep 0.1
     nzero    0 count   43 frac 0.172691
     nzero    1 count    5 frac 0.0200803
     nzero    2 count  167 frac 0.670683
     nzero    3 count    7 frac 0.0281125
     nzero    4 count   24 frac 0.0963855
     nzero    5 count    1 frac 0.00401606
     nzero    7 count    1 frac 0.00401606
     nzero 11195 count    1 frac 0.00401606

     nzero    0 count   43 frac 0.172691
    ...

     nzero    1 count    5 frac 0.0200803
     i  187 nzero    1 NScanTest /tmp/blyth/opticks/tgltf/extras//61      soname                BotRefGapCutHols0xc34bb28 tag    [ 0:in] typ intersection box3 disc  msg 
     i  188 nzero    1 NScanTest /tmp/blyth/opticks/tgltf/extras//60      soname                   BotESRCutHols0xbfa7368 tag    [ 0:in] typ intersection box3 disc  msg 
     i  190 nzero    1 NScanTest /tmp/blyth/opticks/tgltf/extras//58      soname                TopRefGapCutHols0xbf9cef8 tag    [ 0:in] typ   intersection disc  msg 
     i  191 nzero    1 NScanTest /tmp/blyth/opticks/tgltf/extras//57      soname                   TopESRCutHols0xbf9de10 tag    [ 0:in] typ   intersection disc  msg 
     i  205 nzero    1 NScanTest /tmp/blyth/opticks/tgltf/extras//43      soname                pmt-hemi-cathode0xc2f1ce8 tag    [ 0:un] typ union difference zsphere  msg 

     nzero    2 count  167 frac 0.670683

     nzero    3 count    7 frac 0.0281125
     i  105 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras//143     soname                          GdsOfl0xbf73918 tag    [ 0:un] typ      union cylinder  msg 
     i  180 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras//68      soname                       SstTopHub0xc2643d8 tag    [ 0:un] typ      union cylinder  msg 
     i  194 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras//54      soname                 headon-pmt-assy0xbf55198 tag    [ 0:un] typ      union cylinder  msg 
     i  206 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras//42      soname                             oav0xc2ed7c8 tag    [ 0:un] typ union cylinder cone  msg 
     i  211 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras//37      soname                             lso0xc028a38 tag    [ 0:un] typ union cylinder cone  msg 
     i  222 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras//26      soname                 CtrGdsOflBotClp0xbf5dec0 tag    [ 0:un] typ      union cylinder  msg 
     i  226 nzero    3 NScanTest /tmp/blyth/opticks/tgltf/extras//22      soname                             gds0xc28d3f0 tag    [ 0:un] typ union cylinder cone  msg 


    2017-06-26 20:22:50.361 INFO  [1304562] [NCSG::Deserialize@932] NCSG::Deserialize VERBOSITY 0 basedir /tmp/blyth/opticks/tgltf/extras// txtpath /tmp/blyth/opticks/tgltf/extras//csg.txt nbnd 249
    2017-06-26 20:22:50.401 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//145 typmsk union cylinder  uniontree_cy YES uniontree_cy_co NO
    2017-06-26 20:22:50.401 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//144 typmsk union cylinder  uniontree_cy YES uniontree_cy_co NO
    2017-06-26 20:22:50.402 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//143 typmsk union cylinder  uniontree_cy YES uniontree_cy_co NO
    2017-06-26 20:22:50.408 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//130 typmsk union cylinder  uniontree_cy YES uniontree_cy_co NO
    2017-06-26 20:22:50.427 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//77 typmsk union cylinder  uniontree_cy YES uniontree_cy_co NO
    2017-06-26 20:22:50.428 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//75 typmsk union cylinder  uniontree_cy YES uniontree_cy_co NO
    2017-06-26 20:22:50.431 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//68 typmsk union cylinder  uniontree_cy YES uniontree_cy_co NO
    2017-06-26 20:22:50.438 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//54 typmsk union cylinder  uniontree_cy YES uniontree_cy_co NO
    2017-06-26 20:22:50.442 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//42 typmsk union cylinder cone  uniontree_cy NO uniontree_cy_co YES
    2017-06-26 20:22:50.444 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//37 typmsk union cylinder cone  uniontree_cy NO uniontree_cy_co YES
    2017-06-26 20:22:50.448 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//26 typmsk union cylinder  uniontree_cy YES uniontree_cy_co NO
    2017-06-26 20:22:50.448 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//25 typmsk union cylinder  uniontree_cy YES uniontree_cy_co NO
    2017-06-26 20:22:50.449 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//24 typmsk union cylinder cone  uniontree_cy NO uniontree_cy_co YES
    2017-06-26 20:22:50.450 INFO  [1304562] [NNodeUncoincide::uncoincide_tree@312]  treedir /tmp/blyth/opticks/tgltf/extras//22 typmsk union cylinder cone  uniontree_cy NO uniontree_cy_co YES
    2017-06-26 20:22:50.457 INFO  [1304562] [NCSG::DeserializeTrees@901] NCSG::DeserializeTrees /tmp/blyth/opticks/tgltf/extras// found trees : 249
    2


     nzero    4 count   24 frac 0.0963855

     nzero    5 count    1 frac 0.00401606
     i  219 nzero    5 NScanTest /tmp/blyth/opticks/tgltf/extras//29      soname                       OcrGdsPrt0xc352518 tag    [ 0:di] typ union difference cylinder cone  msg 

     nzero    7 count    1 frac 0.00401606
     i  103 nzero    7 NScanTest /tmp/blyth/opticks/tgltf/extras//145     soname                 OflTnkContainer0xc17cf50 tag    [ 0:un] typ      union cylinder  msg 

     nzero 11195 count    1 frac 0.00401606
     i  182 nzero 11195 NScanTest /tmp/blyth/opticks/tgltf/extras//66      soname                 SstTopRadiusRib0xc271720 tag    [ 0:di] typ difference box3 convexpolyhedron  msg 
    simon:sysrap blyth$ 




