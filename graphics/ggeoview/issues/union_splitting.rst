Union Splitting
====================

many upwards going photons think their m1 is Ac when actually Gd
---------------------------------------------------------------------------

* investigating using a torch emitter from middle of IAV

::

   3150 : nf    0 nv    0 id   3150 pid   3149 : __dd__Geometry__Pool__lvNearPoolLiner--pvNearPoolOWS0xbf55b10       __dd__Geometry__Pool__lvNearPoolOWS0xbf93840 
   3151 : nf    0 nv    0 id   3151 pid   3150 : __dd__Geometry__Pool__lvNearPoolOWS--pvNearPoolCurtain0xc5c5f20   __dd__Geometry__Pool__lvNearPoolCurtain0xc2ceef0 
   3152 : nf    0 nv    0 id   3152 pid   3151 : __dd__Geometry__Pool__lvNearPoolCurtain--pvNearPoolIWS0xc15a498       __dd__Geometry__Pool__lvNearPoolIWS0xc28bc60 
   3153 : nf   96 nv  157 id   3153 pid   3152 : __dd__Geometry__Pool__lvNearPoolIWS--pvNearADE10xc2cf528                 __dd__Geometry__AD__lvADE0xc2a78c0 
   3154 : nf   96 nv  157 id   3154 pid   3153 : __dd__Geometry__AD__lvADE--pvSST0xc128d90                 __dd__Geometry__AD__lvSST0xc234cd0 
   3155 : nf   96 nv  157 id   3155 pid   3154 : __dd__Geometry__AD__lvSST--pvOIL0xc241510                 __dd__Geometry__AD__lvOIL0xbf5e0b8 
   3156 : nf  288 nv  481 id   3156 pid   3155 : __dd__Geometry__AD__lvOIL--pvOAV0xbf8f638                 __dd__Geometry__AD__lvOAV0xbf1c760 
   3157 : nf  332 nv  678 id   3157 pid   3156 : __dd__Geometry__AD__lvOAV--pvLSO0xbf8e120                 __dd__Geometry__AD__lvLSO0xc403e40 

   3158 : nf  288 nv  483 id   3158 pid   3157 :    __dd__Geometry__AD__lvLSO--pvIAV0xc2d0348                 __dd__Geometry__AD__lvIAV0xc404ee8 
   3159 : nf  288 nv  617 id   3159 pid   3158 :       __dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00                 __dd__Geometry__AD__lvGDS0xbf6cbb8 
   3160 : nf   92 nv  211 id   3160 pid   3158 :       __dd__Geometry__AD__lvIAV--pvOcrGdsInIAV0xbf6b0e0         __dd__Geometry__AdDetails__lvOcrGdsInIav0xbf6dd58 

   3161 : nf  384 nv  632 id   3161 pid   3157 :    __dd__Geometry__AD__lvLSO--pvIavTopHub0xc34e6e8    __dd__Geometry__AdDetails__lvIavTopHub0xc129d88 
   3162 : nf  384 nv  636 id   3162 pid   3157 :    __dd__Geometry__AD__lvLSO--pvCtrGdsOflBotClp0xc2ce2a8 __dd__Geometry__AdDetails__lvCtrGdsOflBotClp0xc407eb0 
   3163 : nf  192 nv  336 id   3163 pid   3157 : __dd__Geometry__AD__lvLSO--pvCtrGdsOflTfbInLso0xc2ca538 __dd__Geometry__AdDetails__lvCtrGdsOflTfbInLso0xbfa0728 
   3164 : nf   96 nv  157 id   3164 pid   3157 : __dd__Geometry__AD__lvLSO--pvCtrGdsOflInLso0xbf74250 __dd__Geometry__AdDetails__lvCtrGdsOflInLso0xc28cc88 
   3165 : nf  576 nv 1189 id   3165 pid   3157 : __dd__Geometry__AD__lvLSO--pvOcrGdsPrt0xbf6d0d0    __dd__Geometry__AdDetails__lvOcrGdsPrt0xc352630 
   3166 : nf  384 nv  636 id   3166 pid   3157 : __dd__Geometry__AD__lvLSO--pvOcrGdsBotClp0xbfa1610 __dd__Geometry__AdDetails__lvCtrGdsOflBotClp0xc407eb0 
   3167 : nf  192 nv  488 id   3167 pid   3157 : __dd__Geometry__AD__lvLSO--pvOcrGdsTfbInLso0xbfa1818 __dd__Geometry__AdDetails__lvOcrGdsTfbInLso0xc3529c0 
   3168 : nf   92 nv  210 id   3168 pid   3157 : __dd__Geometry__AD__lvLSO--pvOcrGdsInLso0xbf6d280  __dd__Geometry__AdDetails__lvOcrGdsInLso0xc353990 
   3169 : nf   12 nv   24 id   3169 pid   3157 : __dd__Geometry__AD__lvLSO--pvOavBotRibs--OavBotRibs--OavBotRibRot0xbf5af90    __dd__Geometry__AdDetails__lvOavBotRib0xc353d30 
   3170 : nf   12 nv   24 id   3170 pid   3157 : __dd__Geometry__AD__lvLSO--pvOavBotRibs--OavBotRibs..1--OavBotRibRot0xc3531c0    __dd__Geometry__AdDetails__lvOavBotRib0xc353d30 
   3171 : nf   12 nv   24 id   3171 pid   3157 : __dd__Geometry__AD__lvLSO--pvOavBotRibs--OavBotRibs..2--OavBotRibRot0xc353e30    __dd__Geometry__AdDetails__lvOavBotRib0xc353d30 
   3172 : nf   12 nv   24 id   3172 pid   3157 : __dd__Geometry__AD__lvLSO--pvOavBotRibs--OavBotRibs..3--OavBotRibRot0xc541230    __dd__Geometry__AdDetails__lvOavBotRib0xc353d30 


Problem remains with only 2 volumes, 3158 and 3159::

    see ~/env/bin/ggv.sh
    export GGEOVIEW_QUERY="range:3158:3160" 
       # just 2 volumes (python style range) __dd__Geometry__AD__lvLSO--pvIAV0xc2d0348, __dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00  

    ggv --idyb --torchconfig="radius=0;zenith_azimuth=0,1,0,1"


Isolate issue to single volume : 3158
--------------------------------------

Single volume 3158 messing up all by itself ::

    ggv --jdyb --torchconfig "radius=0;zenith_azimuth=0,1,0,1"   
         

OpenGL Eyeballing
~~~~~~~~~~~~~~~~~~~ 
  
* flickery underside of top lid
* __dd__Geometry__AD__lvLSO--pvIAV0xc2d0348  => /dd/Geometry/AD/lvLSO#pvIAV

* union of tubs and polycone seems to fail in this case, with the "internal" 
  tubs/polycone transition acting as an effective boundary to OptiX rayTrace 
  intersection tests (there is no corresponding GBoundary : so m1/m2/su will be wonky)

  side view in orthographic mode makes this very apparent, with a clear disc
  of photon intersections at the top of the cylinder with another disc on the polycone
  surface   

* looking up from inside (with flipped normals) can see a featureless but flickery surface
  in wireframe its apparent that the "spokes" are doubled up 


NumPy Look at faces/vertices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Jump into geocache for 1 volume geometry::

    delta:ggeoview blyth$ cd $(ggv --jdyb --idp)

Check mergedmesh 0::

    In [1]: n = np.load("GMergedMesh/0/nodeinfo.npy")

    In [3]: n[n[:,0]>0]
    Out[3]: array([[ 288,  483, 3158, 3157]], dtype=uint32)

    In [4]: f = np.load("GMergedMesh/0/indices.npy")

    In [14]: f.min()
    Out[14]: 0

    In [15]: f.max()
    Out[15]: 482

    In [8]: v = np.load("GMergedMesh/0/vertices.npy")

    In [9]: v.shape
    Out[9]: (483, 3)

    In [19]: cuf = count_unique(f[:,0])   # hub vertices should be apparent by appearing in more faces 

    In [20]: cuf[cuf[:,1]>4]
    Out[20]: 
    array([[ 96,   6],
           [127,   6],
           [421,   6],
           [453,   6]])    # expected more, but the many repeated vertices explains why only 6 


    In [24]: v[[96,127,421,453]]
    Out[24]: 
    array([[ -18079.453, -799699.438,   -5565.   ],                 
           [ -18079.453, -799699.438,   -8650.   ],
           [ -18079.461, -799699.562,   -5475.51 ],
           [ -18079.461, -799699.562,   -5564.95 ]], dtype=float32)

    In [26]: v[[96,127,421,453]][:,2] + 8650
    Out[26]: array([ 3085.  ,     0.  ,  3174.49,  3085.05], dtype=float32)    ## OOPS 2 layers of Z only 0.05 different from each other

    In [29]: cnv = count_unique(v[:,2])     # unique z values

    In [30]: cnv
    Out[30]: 
    array([[-8650.  ,    79.  ],    # base
           [-5565.  ,    78.  ],    # squealer-
           [-5564.95,    79.  ],    # squealer+
           [-5549.95,   168.  ],    
           [-5475.51,    79.  ]])


    In [31]: cnv[:,0]
    Out[31]: array([-8650.  , -5565.  , -5564.95, -5549.95, -5475.51])

    In [32]: cnv[:,0] + 8650
    Out[32]: array([    0.  ,  3085.  ,  3085.05,  3100.05,  3174.49])    

    ##
    ##                        observed from         expected from
    ##                        vertices              detdesc parameter calc below
    ##        
    ##     IavBrlHeight         3085. 
    ##     IavLidFlgThickness     15.
    ##     IavHeight            3174.49  (+0.05)    3174.44     
    ##     
    ##
    ##     presumably Geant4 triangulation did the 0.05 nudge for visualization reasons ?
    ##
    ##     Pragmatic approach: need code to identify and heal afflicted meshes...
    ##     (G4 triangulation code is not smth I am motivated to get into)
    ## 
    ##   :google:`mesh remove internal faces`
    ##
    ##  hmm some circle fitting would be useful here ... 
    ##       http://stackoverflow.com/questions/26574945/how-to-find-the-center-of-circle-using-the-least-square-fit-in-python
    ##         http://autotrace.sourceforge.net/WSCG98.pdf
    ##
    ##   will need scipy py27-scipy 
    ##   maybe not   http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html
    ## 

::

    In [37]: p0 = v[v[:,2] == -8650.]

    In [41]: p1 = v[v[:,2] == -5565. ]

    In [42]: p2 = v[v[:,2] == (-5565.+.05) ]

    In [43]: p3 = v[v[:,2] == -5549.95]

    In [44]: p4 = v[v[:,2] == -5475.51]


    In [57]: p0   # half of the 79 are duplicated ?
    Out[57]: 
    array([[ -17232.102, -801009.25 ,   -8650.   ],
           [ -16921.973, -800745.312,   -8650.   ],
           [ -16921.973, -800745.312,   -8650.   ],
           [ -16690.721, -800410.062,   -8650.   ],
           [ -16690.721, -800410.062,   -8650.   ],
           [ -16554.107, -800026.438,   -8650.   ],
           [ -16554.107, -800026.438,   -8650.   ],
            ...

    In [59]: p1   # again 1st half are duplicated other than 1st 
    Out[59]: 
    array([[ -17232.102, -801009.25 ,   -5565.   ],
           [ -16921.973, -800745.312,   -5565.   ],
           [ -16921.973, -800745.312,   -5565.   ],
           [ -16690.721, -800410.062,   -5565.   ],
           [ -16690.721, -800410.062,   -5565.   ],
           [ -16554.107, -800026.438,   -5565.   ],






    In [39]: plt.plot( p0[:,0], p0[:,1] )
    Out[39]: [<matplotlib.lines.Line2D at 0x11143acd0>]

    In [40]: plt.show()


Some but not all the spokes line up::

    In [47]: plt.plot(p1[:,0], p1[:,1], p2[:,0], p2[:,1] )
    Out[47]: 
    [<matplotlib.lines.Line2D at 0x10fa8a390>,
     <matplotlib.lines.Line2D at 0x10fa8a610>]

    In [48]: plt.show()

Flange and top::

    In [49]: plt.plot(p3[:,0], p3[:,1], p4[:,0], p4[:,1] )
    Out[49]: 
    [<matplotlib.lines.Line2D at 0x113b5a550>,
     <matplotlib.lines.Line2D at 0x113b5a7d0>]

All together::

    In [55]: plt.plot(p0[:,0], p0[:,1], p1[:,0], p1[:,1], p2[:,0], p2[:,1], p3[:,0], p3[:,1], p4[:,0], p4[:,1] )


dybgaudi/Detector/XmlDetDesc/DDDB/AD/IAV.xml::

     01 <?xml version="1.0" encoding="UTF-8"?>
      2 <!-- Warning: this is a generated file.  Any modifications may be lost. -->
      3 <!DOCTYPE DDDB SYSTEM "../DTD/geometry.dtd" [
      4   <!ENTITY ADParameters SYSTEM "parameters.xml">
      5   <!ENTITY AdDetailParameters SYSTEM "../AdDetails/parameters.xml">
      6   <!ENTITY OverflowParameters SYSTEM "../OverflowTanks/parameters.xml">
      7   <!ENTITY CalibrationBoxParameters SYSTEM "../CalibrationBox/parameters.xml">
      8   <!ENTITY HandWrittenPhysVols SYSTEM "../AdDetails/IAVPhysVols.xml">
      9 ${DD_AD_IAV_EE}
     10  ]>
     11 <DDDB>
     12 &ADParameters;
     13 &AdDetailParameters;
     14 &OverflowParameters;
     15 &CalibrationBoxParameters;
     16 ${DD_AD_IAV_TOP}
     17 <logvol name="lvIAV" material="Acrylic">
     18   <union name="iav">
     19     <tubs name="iav_cyl"
     20           sizeZ="IavBrlHeight"
     21           outerRadius="IavBrlOutRadius"
     22           />
     23     <polycone name="iav_polycone">
     24       <zplane z="IavBrlHeight"
     25               outerRadius="IavLidRadius"
     26               />
     27       <zplane z="IavBrlHeight+IavLidFlgThickness"
     28               outerRadius="IavLidRadius"
     29               />
     30       <zplane z="IavBrlHeight+IavLidFlgThickness"
     31               outerRadius="IavLidConBotRadius"
     32               />
     33       <zplane z="IavHeight"
     34               outerRadius="IavLidConTopRadius"
     35               />
     36     </polycone>
     //
     //
     //     ARGHH : IS THIS THE CAUSE ???????? 
     //                   POLYCONE WITH TWO ZPLANES AT SAME Z 
     // 
     //
     37     <posXYZ z="-(IavBrlHeight)/2"/>
     38   </union>
     39   <physvol name="pvGDS" logvol="/dd/Geometry/AD/lvGDS">
     40     <posXYZ z="IavBotThickness-IavBrlHeight/2+GdsBrlHeight/2" />
     41   </physvol>
     42   &HandWrittenPhysVols;
     43   ${DD_AD_IAV_PV}
     44 </logvol>
     45 </DDDB>





dybgaudi/Detector/XmlDetDesc/DDDB/AD/parameters.xml::

    149 <!-- Iav barrel thickness -->
    150 <parameter name="IavBrlThickness" value="10*mm"/>
    ...
    153 <!-- Iav bottom thickness -->
    154 <parameter name="IavBotThickness" value="15*mm"/>
    ...
    158 <parameter name="IavBrlHeight" value="3085*mm"/>
    159 <!-- Iav barrel outer radius -->
    160 <parameter name="IavBrlOutRadius" value="1560*mm"/>
    161 <!-- Iav barrel outer radius -->
    162 <parameter name="ADiavRadius" value="IavBrlOutRadius"/>
    163 <!-- Iav lid radius -->
    164 <parameter name="IavLidRadius" value="1565*mm"/>
    165 <!-- Iav lid thickness -->
    166 <parameter name="IavLidThickness" value="15*mm"/>
    167 <!-- Iav lid flange thickness -->
    168 <parameter name="IavLidFlgThickness" value="15*mm"/>
    169 <!-- Iav lid cone inside radius -->
    170 <parameter name="IavLidConInrRadius" value="1520*mm"/>
    171 <!-- Iav lid conical angle -->
    172 <parameter name="IavLidConAngle" value="3.*degree"/>
    173 <!-- Iav lid cone bottom radius -->
    174 <parameter name="IavLidConBotRadius" value="IavLidConInrRadius+IavLidFlgThickness*tan(IavLidConAngle/2.)"/>
    ///
    ///       1520 + 15*tan(3deg/2.)
    ///
    175 <!-- Iav lid cone top radius -->
    176 <parameter name="IavLidConTopRadius" value="100*mm"/>
    177 <!-- Iav lid cone height -->
    178 <parameter name="IavLidConHeight" value="(IavLidConBotRadius-IavLidConTopRadius)*tan(IavLidConAngle)"/>
    ///
    ///          (1520 + 15*tan(1.5deg) - 100)*tan(3deg)
    ///
    /// In [16]: (1520. + 15.*math.tan( math.pi*1.5/180. ) - 100.)*math.tan(math.pi*3./180. )
    /// Out[16]: 74.43963177188732

    ...
    189 <!-- Iav height to the top of the cone -->
    190 <parameter name="IavHeight" value="IavBrlHeight+IavLidFlgThickness+IavLidConHeight"/>
    ///
    /// In [17]: 3085. + 15. + (1520. + 15.*math.tan( math.pi*1.5/180. ) - 100.)*math.tan(math.pi*3./180. )
    /// Out[17]: 3174.4396317718874
    ///     
    ///
    191 <!-- Iav lid height from barrel top the cone top -->
    192 <parameter name="IavLidHeight" value="IavHeight-IavBrlHeight"/>
    ///
    ///
    ///


    ...
    217 <!-- Gds cone top radius -->
    218 <parameter name="GdsConTopRadius" value="75*mm"/>
    219 <!-- Gds cone bottom radius (same as IAV lid cone inner radius -->
    220 <parameter name="GdsConBotRadius" value="IavLidConInrRadius"/>
    221 <!-- Gds barrel radius -->
    222 <parameter name="GdsBrlRadius" value="IavBrlOutRadius-IavBrlThickness"/>
    223 <!-- Gds barrel height -->
    224 <parameter name="GdsBrlHeight" value="IavBrlHeight-IavBotThickness"/>
    225 <!-- Gds cone height -->
    226 <parameter name="GdsConHeight" value="(GdsConBotRadius-GdsConTopRadius)*tan(IavLidConAngle)"/>
    227 <!-- Gds total height (till the bot of IAV hub) -->
    228 <parameter name="GdsHeight" value="GdsBrlHeight+IavLidFlgThickness+IavLidConHeight"/>



dybgaudi/Detector/XmlDetDesc/DDDB/AD/parameters.xml::

    058 <parameter name="OavThickness" value="18*mm"/>
     59 <!-- Oav barrel height -->
     60 <parameter name="OavBrlHeight" value="3982*mm"/>
     61 <!-- Oav barrel outer radius -->
     62 <parameter name="OavBrlOutRadius" value="2000*mm"/>
     63 <!-- Oav barrel flange thickness -->
     64 <parameter name="OavBrlFlgThickness" value="45*mm"/>
     65 <!-- Oav barrel flange radius -->
     66 <parameter name="OavBrlFlgRadius" value="2040*mm"/>
     67 <!-- Oav lid flange thickness -->
     68 <parameter name="OavLidFlgThickness" value="39*mm"/>
     69 <!-- Oav lid flange width -->
     70 <parameter name="OavLidFlgWidth" value="110*mm"/>
     71 <!-- Oav lid conical angle -->
     72 <parameter name="OavLidConAngle" value="3.*degree"/>
     73 <!-- Oav conical lid bottom radius -->
     74 <parameter name="OavLidConBotRadius" value="OavBrlFlgRadius-OavLidFlgWidth"/>
     75 <!-- Oav conical lid top radius -->
     76 <parameter name="OavLidConTopRadius" value="125*mm"/>
     77 <!-- Oav cone height from the turning point -->
     78 <parameter name="OavLidConHeight" value="(OavLidConBotRadius-OavLidConTopRadius)*tan(OavLidConAngle)"/>
     79 <!-- Oav height to the top of the cone -->
     80 <parameter name="OavHeight" value="OavBrlHeight+OavThickness/cos(OavLidConAngle)+OavLidConHeight"/>
     81 <!-- Oav lid height from barrel top to the cone top -->
     82 <parameter name="OavLidHeight" value="OavHeight-OavBrlHeight"/>
     83 <!-- Oav bottom rib height -->
     84 <parameter name="OavBotRibHeight" value="197*mm"/>
    ...
    109 <!-- Lso barrel radius -->
    110 <parameter name="LsoBrlRadius" value="OavBrlOutRadius - OavThickness"/>
    111 <!-- Lso barrel height -->
    112 <parameter name="LsoBrlHeight" value="OavBrlHeight-OavThickness"/>
    113 <!-- Lso cone bottom radius -->
    114 <parameter name="LsoConBotRadius" value="OavLidConBotRadius"/>
    115 <!-- Lso cone top radius (same as the OAV lid top) -->
    116 <parameter name="LsoConTopRadius" value="OavLidConTopRadius"/>
    117 <!--
    118     The tip of LSO (with thickness of OAV lid flange) so LSO is filled to the very top of its container: OAV
    119 -->
    120 <parameter name="LsoConTopTipRadius" value="50*mm"/>
    121 <!-- Lso cone height -->
    122 <parameter name="LsoConHeight" value="(LsoConBotRadius-LsoConTopRadius)*tan(OavLidConAngle)"/>
    123 <!-- Lso total height (till the bot of hub, or the very top of OAV) -->
    124 <parameter name="LsoHeight" value="LsoBrlHeight+OavThickness/cos(OavLidConAngle)+OavLidConHeight"/>
    125 <!-- The 1th corner z pos of LSO -->
    ...


Next volume : 3159, same structure acting OK
-----------------------------------------------
 
::

    ggv --kdyb --torchconfig "radius=0;zenith_azimuth=0,1,0,1"     # volume 3159

Single volume 3159 : uniform all Gd 1st intersection

* __dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00  == /dd/Geometry/AD/lvIAV#pvGDS

* in this case the union seems to work with no photons "seeing" the virtual 
  tubs/polycone boundary : again use orthographic side view and rotate 
  around, clearly only one boundary being intersected

* looking up from inside (with flipped normals) can see up to the top little cylindrical snout



Check at detdesc level 
--------------------------

Below detdesc xml generated by 

http://dayabay.ihep.ac.cn/tracs/dybsvn/browser/dybgaudi/trunk/Detector/XmlDetDesc/python/XmlDetDescGen/AD/gen.py







dybgaudi/Detector/XmlDetDesc/DDDB/AD/LSO.xml::

     01 <?xml version="1.0" encoding="UTF-8"?>
      2 <!-- Warning: this is a generated file.  Any modifications may be lost. -->
      3 <!DOCTYPE DDDB SYSTEM "../DTD/geometry.dtd" [
      4   <!ENTITY ADParameters SYSTEM "parameters.xml">
      5   <!ENTITY AdDetailParameters SYSTEM "../AdDetails/parameters.xml">
      6   <!ENTITY OverflowParameters SYSTEM "../OverflowTanks/parameters.xml">
      7   <!ENTITY CalibrationBoxParameters SYSTEM "../CalibrationBox/parameters.xml">
      8   <!ENTITY HandWrittenPhysVols SYSTEM "../AdDetails/LSOPhysVols.xml">
      9 ${DD_AD_LSO_EE}
     10  ]>
     11 <DDDB>
     12 &ADParameters;
     13 &AdDetailParameters;
     14 &OverflowParameters;
     15 &CalibrationBoxParameters;
     16 ${DD_AD_LSO_TOP}
     17 <logvol name="lvLSO" material="LiquidScintillator">
     18   <union name="lso">
     19     <tubs name="lso_cyl"
     20           sizeZ="LsoBrlHeight"
     21           outerRadius="LsoBrlRadius"
     22           />
     23     <polycone name="lso_polycone">
     24       <zplane z="LsoBrlHeight"
     25               outerRadius="LsoConBotRadius"
     26               />
     27       <zplane z="LsoBrlHeight+LsoConHeight"
     28               outerRadius="LsoConTopRadius"
     29               />
     30       <zplane z="LsoBrlHeight+LsoConHeight"
     31               outerRadius="LsoConTopTipRadius"
     32               />
     33       <zplane z="LsoHeight"
     34               outerRadius="LsoConTopTipRadius"
     35               />
     36     </polycone>
     37     <posXYZ z="-(LsoBrlHeight)/2"/>
     38   </union>
     39   <physvol name="pvIAV" logvol="/dd/Geometry/AD/lvIAV">
     40     <posXYZ z="OavBotRibHeight+IavBotVitHeight+IavBotRibHeight-LsoBrlHeight/2+IavBrlHeight/2" />
     41   </physvol>
     42   &HandWrittenPhysVols;
     43   ${DD_AD_LSO_PV}
     44 </logvol>
     45 </DDDB>




dybgaudi/Detector/XmlDetDesc/DDDB/AD/GDS.xml::

     01 <?xml version="1.0" encoding="UTF-8"?>
      2 <!-- Warning: this is a generated file.  Any modifications may be lost. -->
      3 <!DOCTYPE DDDB SYSTEM "../DTD/geometry.dtd" [
      4   <!ENTITY ADParameters SYSTEM "parameters.xml">
      5   <!ENTITY AdDetailParameters SYSTEM "../AdDetails/parameters.xml">
      6   <!ENTITY OverflowParameters SYSTEM "../OverflowTanks/parameters.xml">
      7   <!ENTITY CalibrationBoxParameters SYSTEM "../CalibrationBox/parameters.xml">
      8   <!ENTITY HandWrittenPhysVols SYSTEM "../AdDetails/GDSPhysVols.xml">
      9 ${DD_AD_GDS_EE}
     10  ]>
     11 <DDDB>
     12 &ADParameters;
     13 &AdDetailParameters;
     14 &OverflowParameters;
     15 &CalibrationBoxParameters;
     16 ${DD_AD_GDS_TOP}
     17 <logvol name="lvGDS" material="GdDopedLS">
     18   <union name="gds">
     19     <tubs name="gds_cyl"
     20           sizeZ="GdsBrlHeight"
     21           outerRadius="GdsBrlRadius"
     22           />
     23     <polycone name="gds_polycone">
     24       <zplane z="GdsBrlHeight"
     25               outerRadius="GdsConBotRadius"
     26               />
     27       <zplane z="GdsBrlHeight+GdsConHeight"
     28               outerRadius="GdsConTopRadius"
     29               />
     30       <zplane z="GdsHeight"
     31               outerRadius="GdsConTopRadius"
     32               />
     33     </polycone>
     34     <posXYZ z="-(GdsBrlHeight)/2"/>
     35   </union>
     36   &HandWrittenPhysVols;
     37   ${DD_AD_GDS_PV}
     38 </logvol>
     39 </DDDB>




     * polycons : 
     * https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch04.html






~                                                                                                                                      
~                                                                                                                                      


