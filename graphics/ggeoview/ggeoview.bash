ggv-(){   ggeoview- ; }
ggv-cd(){ ggeoview-cd ; }
ggv-i(){  ggeoview-install ; }
ggv--(){  ggeoview-depinstall ; }
ggv-lldb(){ 
   echo use ggv --dbg  in order to setup environment
   #ggeoview-lldb $* ; 
}


ggeoview-src(){      echo graphics/ggeoview/ggeoview.bash ; }
ggeoview-source(){   echo ${BASH_SOURCE:-$(env-home)/$(ggeoview-src)} ; }
ggeoview-vi(){       vi $(ggeoview-source) ; }
ggeoview-usage(){ cat << EOU

GGeoView
==========

Start from glfwtest- and add in OptiX functionality from optixrap-

* NB raytrace- is another user of optixwrap- 


surface property debug, PMT id 
-------------------------------


ISSUE : many upwards going photons think their m1 is Ac when actually Gd
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

::

    ggv --jdyb --torchconfig "radius=0;zenith_azimuth=0,1,0,1"     # volume 3158
         
         single volume 3158 messing up all by itself 
  
         * flickery underside of top lid
         * __dd__Geometry__AD__lvLSO--pvIAV0xc2d0348  => /dd/Geometry/AD/lvLSO#pvIAV


    ggv --kdyb --torchconfig "radius=0;zenith_azimuth=0,1,0,1"     # volume 3159

         single volume 3159 : uniform all Gd 1st intersection

         * __dd__Geometry__AD__lvIAV--pvGDS0xbf6ab00  == /dd/Geometry/AD/lvIAV#pvGDS



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
     37     <posXYZ z="-(IavBrlHeight)/2"/>
     38   </union>
     39   <physvol name="pvGDS" logvol="/dd/Geometry/AD/lvGDS">
     40     <posXYZ z="IavBotThickness-IavBrlHeight/2+GdsBrlHeight/2" />
     41   </physvol>
     42   &HandWrittenPhysVols;
     43   ${DD_AD_IAV_PV}
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
    175 <!-- Iav lid cone top radius -->
    176 <parameter name="IavLidConTopRadius" value="100*mm"/>
    177 <!-- Iav lid cone height -->
    178 <parameter name="IavLidConHeight" value="(IavLidConBotRadius-IavLidConTopRadius)*tan(IavLidConAngle)"/>
    179 <!-- Iav bottom rib height -->
    180 <parameter name="IavBotRibHeight" value="200*mm"/>
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






     * polycons : 
     * https://geant4.web.cern.ch/geant4/UserDocumentation/UsersGuides/ForApplicationDeveloper/html/ch04.html



~                                                                                                                                      
~                                                                                                                                      





     




ISSUE : seqmat first material mismatch to genstep material
-------------------------------------------------------------

Do not observe any seq trucation to 0xF mismatch, although that is possible.

npy- genstep_sequence_material_mismatch.py::

    In [156]: off = np.arange(len(s_first))[ s_first != p_gsgmat ]

    In [157]: off
    Out[157]: array([  3006,   8521,   8524, ..., 612838, 612839, 612840])

    In [158]: off.shape
    Out[158]: (104400,)

    In [159]: s_first.shape
    Out[159]: (612841,)


Many due to MI, but large chunk of gs:Gs sq:Ac 

TODO: make specialized indices categorizing these discrepancies to allow visualization  



ISSUE : genstep material index in wrong lingo 
----------------------------------------------

Genstep material index read into cs.MaterialIndex and used in wavelength lookups
as a standard line number cu/cerenkovstep.h::

    225         float4 props = wavelength_lookup(wavelength, cs.MaterialIndex);


G4StepNPY::applyLookup does a to b mapping between lingo which is invoked 
immediately after loading the genstep from file in App::loadGenstep::

     549     G4StepNPY genstep(npy);
     550     genstep.setLookup(m_loader->getMaterialLookup());
     551     if(!juno)
     552     {
     553         genstep.applyLookup(0, 2);   // translate materialIndex (1st quad, 3rd number) from chroma to GGeo 
     554     }   


But dumping it, clear that not a line number::

    gs = stc_(1)

    In [33]: gsmat = gs.view(np.int32)[:,0,2]

    In [34]: gsmat
    Out[34]: array([12, 12, 12, ...,  8,  8,  8], dtype=int32)

    In [35]: np.unique(gsmat)
    Out[35]: array([ 1,  8, 10, 12, 13, 14, 19], dtype=int32)
           

Ahha the in file numbers need the lookup applied. 

Is the lookup accomodating the material code customization anyhow ?






RESOLVED photon level material code debug
------------------------------------------

Seeing some crazy materials, seems some parts of npy- not 
updated for the optical buffer approach::

   npy-dump 0 

* trace the material codes into OptiX from GBoundaryLib::createWavelengthAndOpticalBuffers
* simplify access to avoid such divergences

m_materials is GItemIndex which wraps Index::

     845             else if(psrc->isMaterial())
     846             {
     847                 m_materials->add(shortname.c_str(), psrc->getIndex() );  // registering source indices (aiScene mat index) into GItemIndex
     848                 unsigned int index_local = m_materials->getIndexLocal(shortname.c_str());
     849 
     850                 optical_data[opticalOffset + p*4 + optical_index]  = index_local ;
     851                 optical_data[opticalOffset + p*4 + optical_type]   =  0 ;
     852                 optical_data[opticalOffset + p*4 + optical_finish] =  0 ;
     853                 optical_data[opticalOffset + p*4 + optical_value]  =  0 ;



enhancement : COMPUTE mode
----------------------------

Next:

* pull out ggeoview- App into separate file and make reusable 
  from tests/computeTest.cc with minimal duplication with main.cc
  
  * ie need to partition up compute from interop 

  * try to arrange use of same code no matter whether
    OpenGL or OptiX backed buffers are in use, need 
    some kind of facade to make these look the same 
    from the point of view of Thrust interop, CBufSpec 
    will help here

  * splitting the monolithic OEngine  


enhancement : interop with stream compaction, deferred host allocation 
-----------------------------------------------------------------------

Next:

* revive host based detailed debug dumping with PhotonsNPY RecordNPY 

* look into photon picking ? maybe provide a GUI to invoke the pullbacks (perhaps partial)
  and run the detailed debug

* test operation without any host allocations

* implement stream compaction "partial" pullbacks based on internal criteria or external mask  
  following testcode of optixthrust-



RESOLVED issue: jpmt timeouts binary search to pin down 
---------------------------------------------------------

* problem was a genstep with parameters causing an infinite loop
  in cerenkov generation wavelength sampling 

  * solution was protections to prevent sin2Theta going negative



::

    ggv --juno 
       # no pmt, evt propagation vis not working 

    ggv --jpmt --modulo 1000 
       # causes a timeout+freeze requiring a reboot

    ggv --jpmt --modulo 1000 --nopropagate
       # can visualize jpmt OptiX geometry: and it looks OK

    ggv --jpmt --modulo 1000 --trivial
       # swap generate program with a trivial standin  : works 

    ggv --jpmt --modulo 1000 --bouncemax 0
       # just generate, no propagation : timeout+freeze, reboot
       
    ggv --jpmt --modulo 1000 --trivial
       # progressively adding lines from generate into trivial
       # suggests first issue inside generate_cerenkov_photon/wavelength_lookup

    ggv --jpmt --modulo 1000 --bouncemax 0
       # just generate, no propagation : works after kludging wavelength_lookup 
       # to always give a constant valid float4

    ggv --jpmt --modulo 100
       #  still with kludged wavelength_lookup : works, with photon animation operational
       #
       # interestingly performance of OptiX and OpenGL geometry visualizations
       # are about the same with the full JUNO geometry 2-3 fps, 
       # with DYB OpenGL vis is much faster than OptiX appoaching: 60 fps  

    ggv --jpmt --modulo 50
       #  still with kludged wavelength_lookup : timeout...  maybe stepping off reservation somewhere else reemission texture ?

    ggv --jpmt --modulo 100 --override 1 
       # putting back the wavelength_look get timeout even when override to a single photon

    ggv --jpmt --modulo 100 --override 1 --trivial
       # with trivial prog doing wavelength dumping

    ggv --make --jpmt --modulo 100 --override 1
       # with bounds checking on wavelength lookup succeed with single photon, but not without the override

    ggv --make --jpmt --modulo 100 --override 5181
       # with bounds checking on wavelength lookup succeed with override 5181, failing at override 5182
       #    photon_id = 5181  is doing something naughty

    ggv --make --jpmt --modulo 100 --override 5181 --debugidx 5180
       # check on the photon before, which works
       #
       #
       # [2015-Aug-31 16:23:50.594320]: OEngine::generate OVERRIDE photon count for debugging to 5181
       #  generate debug photon_id 5180 genstep_id 18 ghead.i.x -18001 
       #  cs.Id -18001 ParentId 1 MaterialIndex 48 NumPhotons 39 
       #

    ggv --make --jpmt --modulo 100 --override 5182 --debugidx 5181 --bouncemax 0 
       # now the one that fails, with propagation inhibited  : still failing 

    ggv --make --jpmt --modulo 100 --override 5182 --debugidx 5181 
       # try with kludge skipping of Aluminium : works, so can dump nemesis 
       #
       # ... hmm refractive index of 1.000 for a metal 
       #
       # [2015-Aug-31 16:45:04.481506]: OEngine::generate count 0 size(10406,1)
       # [2015-Aug-31 16:45:04.481600]: OEngine::generate OVERRIDE photon count for debugging to 5182
       # generate debug photon_id 5181 genstep_id 19 ghead.i.x -19001 
       # cs.Id -19001 ParentId 1 MaterialIndex 24 NumPhotons 282 
       # x0 -15718.109375 -2846.020996 -9665.920898  t0 62.278240 
       # DeltaPosition -1.087246 -0.197473 -0.667886  step_length 1.291190  
       # code 13  charge -1.000000 weight 1.000000 MeanVelocity 299.792267 
       # BetaInverse 1.000001  Pmin 0.000002 Pmax 0.000015 maxCos 0.751880 
       # maxSin2 0.434676  MeanNumberOfPhotons1 232.343796 MeanNumberOfPhotons2 232.343796 MeanNumberOfPhotonsMax 232.343796 
       # p0 -0.842050 -0.152938 -0.517264  
       # cscheck sample wavelength lo/mi/hi   59.999996 111.724136 810.000122 
       # cscheck sample rindex lo/mi/hi   1.000000 1.000000 1.000000 
       # cscheck sample abslen lo/mi/hi   1000000.000000 1000000.000000 1000000.000000 
       # cscheck sample scalen lo/mi/hi   1000000.000000 1000000.000000 1000000.000000 
       # cscheck sample reempr lo/mi/hi   0.000000 0.000000 0.000000 
       #

    ggv --make --jpmt --modulo 100 
       #
       # fix by modifying cerernkovstep.h wavelength sampling loop 
       # to avoid sin2Theta from going -ve 
       #
       #      sin2Theta = fmaxf( 0.0001f, (1.f - cosTheta)*(1.f + cosTheta));  
       #
       # TODO: check for artifacts in wavelength distribution
 
   ggv --make --jpmt
       #
       # at modulo 10,    propagate time is 1.009s  
       # at modulo  5,    propagate time is 1.740s
       # at full genstep, propagate time is 7.053s 



Do things go bad on a genstep boundary ?

::

    In [1]: c = np.load("/usr/local/env/juno/cerenkov/1.npy")

    In [7]: c.view(np.int32)[:,0,3][::100].sum()    # number of photons, modulo scaled down and summed matches log 
    Out[7]: 10406

::
    In [8]: cc = c.view(np.int32)[:,0,3][::100].cumsum()    # genstep index 18 has cumsum 5181
    Out[8]: 
    array([  322,   607,   883,  1164,  1476,  1513,  1831,  2160,  2462,
            2776,  3078,  3375,  3699,  4002,  4310,  4603,  4881,  5142,
            5181,  5463,  5776,  6052,  6346,  6628,  6636,  6646,  6942,
            7235,  7521,  7817,  8123,  8399,  8695,  9012,  9295,  9584,
            9777, 10068, 10406])


    In [34]: c[::100][:,0].view(np.int32)
    Out[34]: 
    array([[    -1,      1,     48,    322],    Id/ParentId/MaterialIndex/NumPhotons
           [ -1001,      1,     48,    285],
           [ -2001,      1,     48,    276],
           [ -3001,      1,     48,    281],
           ...


    In [50]: for i,_ in enumerate(c[::100][:,0].view(np.int32)):print i,_,cc[i]
    0  [ -1         1     48    322] 322
    1  [-1001       1     48    285] 607
    2  [-2001       1     48    276] 883
    3  [-3001       1     48    281] 1164
    4  [-4001       1     48    312] 1476     #  48:Water, 24:Aluminium, 42:Tyvek 
    5  [-5001       1     48     37] 1513
    6  [-6001       1     48    318] 1831
    7  [-7001       1     48    329] 2160
    8  [-8001       1     48    302] 2462
    9  [-9001       1     48    314] 2776
    10 [-10001      1     48    302] 3078
    11 [-11001      1     48    297] 3375
    12 [-12001      1     48    324] 3699
    13 [-13001      1     48    303] 4002
    14 [-14001      1     48    308] 4310
    15 [-15001      1     48    293] 4603
    16 [-16001      1     48    278] 4881
    17 [-17001      1     48    261] 5142
    18 [-18001      1     48     39] 5181    ### genstep index 18 ends with photon_id 5180
    19 [-19001      1     24    282] 5463
    20 [-20001      1     24    313] 5776
    21 [-21001      1     24    276] 6052
    22 [-22001      1     24    294] 6346
    23 [-23001      1     24    282] 6628
    24 [-24001      1     24      8] 6636
    25 [-25001   4720     24     10] 6646
    26 [-26001   1553     48    296] 6942
    27 [-27001   4964     48    293] 7235
    28 [-28001   5540     42    286] 7521
    29 [-29001   1552     48    296] 7817
    30 [-30001   6048     48    306] 8123
    31 [-31001   6464     48    276] 8399
    32 [-32001   1156     48    296] 8695
    33 [-33001   1050     48    317] 9012
    34 [-34001   6977     48    283] 9295
    35 [-35001    692     48    289] 9584
    36 [-36001    456     48    193] 9777
    37 [-37001    222     48    291] 10068
    38 [-38001    106     48    338] 10406



issue: jpmt wavelengthBuffer/boundarylib ? maybe bad material indices ?
-------------------------------------------------------------------------

* is the cs.MaterialIndex expected to be the wavelength texture line number ?

  * if so then the jpmt/juno numbers do need a "translation" applied ?
  * GBoundaryLibMetadata.json has 18 boundaries 0..17

::

    In [5]: cd /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae
    /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae

    In [6]: a = np.load("wavelength.npy")

    In [40]: a.reshape(-1,6,39,4).shape
    Out[40]: (18, 6, 39, 4)

    In [47]: a.reshape(-1,6,39,4)[6]
    Out[47]: 
    array([[[       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.33 ,      273.208,  1000000.   ,        0.   ],
            [       1.345,      273.208,  1000000.   ,        0.   ],
            [       1.36 ,      273.208,  1000000.   ,        0.   ],
            [       1.375,      273.208,  1000000.   ,        0.   ],
            [       1.39 ,      691.558,  1000000.   ,        0.   ],
            [       1.384,     1507.119,  1000000.   ,        0.   ],

    In [54]: 18*6
    Out[54]: 108

::

    delta:npy blyth$ /usr/local/env/numerics/npy/bin/NPYTest
    [2015-08-28 21:14:53.244421] [0x000007fff7650e31] [debug]   NPY<T>::load /usr/local/env/juno/cerenkov/1.npy
    G4StepNPY
     ni 3840 nj 6 nk 4 nj*nk 24 
     (    0,    0)               -1                1               48              322  sid/parentId/materialIndex/numPhotons 
     (    0,    1)            0.000            0.000            0.000            0.000  position/time 
     (    0,    2)           -0.861           -0.156           -0.530            1.023  deltaPosition/stepLength 
     (    0,    3)               13           -1.000            1.000          299.792  code 
     (    0,    4)            1.000            0.000            0.000            0.688 
     (    0,    5)            0.527          293.245          293.245            0.000 
     ( 3839,    0)           -38391                4               48               47  sid/parentId/materialIndex/numPhotons 
     ( 3839,    1)          -16.246           -2.947          -10.006            0.064  position/time 
     ( 3839,    2)           -0.191           -0.194            0.236            0.378  deltaPosition/stepLength 
     ( 3839,    3)               11           -1.000            1.000          230.542  code 
     ( 3839,    4)            1.300            0.000            0.000            0.895 
     ( 3839,    5)            0.200          165.673          110.064            0.000 
     24 
     42 
     48 
     24 : 750 
     42 : 52 
     48 : 3038 




Initial values of material indices are not unreasonable, maybe problem on subsequent steps::

    simon:ggeoview blyth$ ggeoview-detector-jpmt
    simon:ggeo blyth$ ggeo-blt 24 42 48
    /usr/local/env/optix/ggeo/bin/GBoundaryLibTest 24 42 48
    GCache::readEnvironment setting IDPATH internally to /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae 
    [2015-08-31 12:54:10.546692] [0x000007fff77ea531] [warning] GBoundaryLib::setWavelengthBuffer didnt see 54, numBoundary: 18
    GBoundaryLib::loadBoundary digest mismatch 7 : d1a3424507d661c74ab51c4b5c82dff0 202bc56442e88df7f4be6f3af62acf70 
    GBoundaryLib::loadBoundary digest mismatch 13 : 8dc0d036da7ed8b5d4606cfe506a82f7 82a76e8ae56ac49dc00174734af2d8b8 
    GBoundaryLib::loadBoundary digest mismatch 14 : ac621cac48edd9555db9b8f9f5f56015 1bb254d022a246eb98cef4846123154e 
    GBoundaryLib::loadBoundary digest mismatch 15 : c3baf1e9325fac7e81b218e23804557d 39b93748d45456bc1aa6cb0e326f0fd3 
    boundary : index  0 dede45b90304e0f9dd9c7c5edce7c8b1 Galactic/Galactic/-/- 
    boundary : index  1 124d278374f95ec3742e1268e6e8f478 Rock/Galactic/-/- 
    boundary : index  2 4befaffca91e8cb0fd5662ae2d81bd65 Air/Rock/-/- 
    boundary : index  3 231c44f02f80c88638cb09dff25df5f6 Air/Air/-/- 
    boundary : index  4 576a076a3f1f332dad075d3c2d8181d7 Aluminium/Air/-/- 
    boundary : index  5 eb855bbd039a6401bfacc6202ea5034c Steel/Rock/-/- 
    boundary : index  6 d18726a8d2660e6be4b8ae326bd38ee6 Water/Steel/-/- 
    boundary : index  7 d1a3424507d661c74ab51c4b5c82dff0 Tyvek/Water/-/CDTyvekSurface 
    boundary : index  8 608795d154c5752988d6882d87de18e6 Water/Tyvek/-/- 
    boundary : index  9 1118e140d2fe2dc9f07c350302e5ee1e Acrylic/Water/-/- 
    boundary : index 10 9ad9179c5dc8584ab0a68f460dbfddde LS/Acrylic/-/- 
    boundary : index 11 da505cbe2bdfaa95b091f31761d81a93 Pyrex/Water/-/- 
    boundary : index 12 11467e52d1bc229355bf173f871790d2 Pyrex/Pyrex/-/- 
    boundary : index 13 8dc0d036da7ed8b5d4606cfe506a82f7 Vacuum/Pyrex/-/PMT_20inch_photocathode_logsurf2 
    boundary : index 14 ac621cac48edd9555db9b8f9f5f56015 Vacuum/Pyrex/PMT_20inch_mirror_logsurf1/- 
    boundary : index 15 c3baf1e9325fac7e81b218e23804557d Vacuum/Pyrex/-/PMT_3inch_photocathode_logsurf2 
    boundary : index 16 00fbb4643f7986d8c5f1499d5b3b3e22 Steel/Water/-/- 
    boundary : index 17 47a41d6b6a602cc04be06523254ec39c Copper/Water/-/- 

    GBoundaryLib.dumpWavelengthBuffer 24 
    GBoundaryLib::dumpWavelengthBuffer wline 24 numSub 18 domainLength 39 numQuad 6 

      24 |   4/  0 Aluminium0x22ca560 
               1.000           1.000           1.000           1.000           1.000           1.000           1.000           1.000
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000

    GBoundaryLib.dumpWavelengthBuffer 42 
    GBoundaryLib::dumpWavelengthBuffer wline 42 numSub 18 domainLength 39 numQuad 6 

      42 |   7/  0 Tyvek0x229f920 
               1.000           1.000           1.000           1.000           1.000           1.000           1.000           1.000
           10000.000       10000.000       10000.000       10000.000       10000.000       10000.000       10000.000       10000.000
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000

    GBoundaryLib.dumpWavelengthBuffer 48 
    GBoundaryLib::dumpWavelengthBuffer wline 48 numSub 18 domainLength 39 numQuad 6 

      48 |   8/  0 Water0x22c0a30 
               1.330           1.360           1.372           1.357           1.352           1.346           1.341           1.335
             273.208         273.208        3164.640       12811.072       28732.207       13644.791        2404.398         371.974
         1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000     1000000.000
               0.000           0.000           0.000           0.000           0.000           0.000           0.000           0.000
    simon:ggeo blyth$ 




issue: ~/jpmt_mm0_too_many_vertices.txt
------------------------------------------

1.79M vertices for jpmt mm0 (global) seems excessive, either missing a repeater or some bug.::

    ggv -G --jpmt

    120 [2015-Aug-25 18:52:37.665158]: GMergedMesh::create index 0 from default root base lWorld0x22ccd90
    121 [2015-Aug-25 18:52:37.730168]: GMergedMesh::create index 0 numVertices 1796042 numFaces 986316 numSolids 289733 numSolidsSelected 1032


From m_mesh_usage in GGeo and GMergedMesh sStrut and sFasteners are the culprits::

    [2015-Aug-25 19:50:40.251333]: AssimpGGeo::convertMeshes  i   19 v  312 f  192 n sStrut0x304f210
    [2015-Aug-25 19:50:40.251575]: AssimpGGeo::convertMeshes  i   20 v 3416 f 1856 n sFasteners0x3074ea0

    [2015-Aug-25 19:54:01.663594]: GMergedMesh::create index 0 from default root base lWorld0x22ccd90
    [2015-Aug-25 19:54:07.339150]: GMergedMesh::create index 0 numVertices 1796042 numFaces 986316 numSolids 289733 numSolidsSelected 1032
    GLoader::load reportMeshUsage (global)
         5 :     62 : sWall0x309ce60 
         6 :      1 : sAirTT0x309cbb0 
         7 :      1 : sExpHall0x22cdb00 
         8 :      1 : sTopRock0x22cd500 
         9 :      1 : sTarget0x22cfbd0 
        10 :      1 : sAcrylic0x22cf9a0 
        19 :    480 : sStrut0x304f210 
        20 :    480 : sFasteners0x3074ea0 
        21 :      1 : sInnerWater0x22cf770 
        22 :      1 : sReflectorInCD0x22cf540 
        23 :      1 : sOuterWaterPool0x22cef90 
        24 :      1 : sSteelTub0x22ce610 
        25 :      1 : sBottomRock0x22cde40 
            ---------

    In [7]: 480+480+62+10
    Out[7]: 1032          ## matches numSolidsSelected

    In [5]: 3416*480+312*480
    Out[5]: 1789440


::

    simon:juno blyth$ grep sFasteners t3.dae
        <geometry id="sFasteners0x3074ea0" name="sFasteners0x3074ea0">
            <source id="sFasteners0x3074ea0-Pos">
              <float_array count="2742" id="sFasteners0x3074ea0-Pos-array">
                <accessor count="914" source="#sFasteners0x3074ea0-Pos-array" stride="3">
            <source id="sFasteners0x3074ea0-Norm">
              <float_array count="5184" id="sFasteners0x3074ea0-Norm-array">
                <accessor count="1728" source="#sFasteners0x3074ea0-Norm-array" stride="3">
            <source id="sFasteners0x3074ea0-Tex">
              <float_array count="2" id="sFasteners0x3074ea0-Tex-array">
                <accessor count="1" source="#sFasteners0x3074ea0-Tex-array" stride="2">
            <vertices id="sFasteners0x3074ea0-Vtx">
              <input semantic="POSITION" source="#sFasteners0x3074ea0-Pos"/>
              <input offset="0" semantic="VERTEX" source="#sFasteners0x3074ea0-Vtx"/>
              <input offset="1" semantic="NORMAL" source="#sFasteners0x3074ea0-Norm"/>
              <input offset="2" semantic="TEXCOORD" source="#sFasteners0x3074ea0-Tex"/>
              <meta id="sFasteners0x3074ea0">
          <instance_geometry url="#sFasteners0x3074ea0">
    simon:juno blyth$ 



Contiguous block of Fasteners all leaves at depth 6::

    simon:env blyth$ grep Fasteners /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GTreePresent.txt | wc -l
         480
    simon:env blyth$ grep Fasteners /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GTreePresent.txt | head -1
       282429 [  6:54799/55279]    0          lFasteners0x3075090   
    simon:env blyth$ grep Fasteners /usr/local/env/geant4/geometry/export/juno/test3.fcc8b4dc9474af8826b29bf172452160.dae/GTreePresent.txt | tail -1
       282908 [  6:55278/55279]    0          lFasteners0x3075090   

    In [9]: 282429+480-1
    Out[9]: 282908


TODO: 


* on dyb GTreeCheck::findRepeatCandidates 

  * not restricting repeats to non-leaf looses some geometry
  * but putting it back gives PMTs in both instance0 and instance1  
  * GTreeCheck.dumpTree ridx not making sense when allow leaf repeats 

* dump the text node tree for juno, to see where sFasteners is 
* add --repeatidx 0,1,2,3 controlled loading in GGeo::loadMergedMeshes etc..
  so can skip the problematic extremely large 0




squeeze approaches for jpmt
----------------------------

* remove vertex color, do at solid/boundary level
* compress vertex normals 
* reuse vertex structures for OptiX ?



computeTest with different core counts controlled via CUDA_VISIBLE_DEVICES
----------------------------------------------------------------------------

Juno Scintillation 2, genstep scaledown 25
--------------------------------------------

::

    genstepAsLoaded : 4e16b039dc40737a4c0c51d7b213a118
    genstepAfterLookup : 4e16b039dc40737a4c0c51d7b213a118
               Type :   scintillation
                Tag :               1
           Detector :            juno
        NumGensteps :            1774
             RngMax :         3000000
         NumPhotons :         1493444
         NumRecords :        14934440
          BounceMax :               9
          RecordMax :              10
        RepeatIndex :              10
         photonData : 33b5c1f991b46e09036e38c110e36102
         recordData : 55a15aacf09d4e8dcf269d6e882b481e
       sequenceData : 035310267fc2a678f2c8cad2031d7101




::

    2.516              GT 750M          ggv.sh --cmp --juno -s 
 
    0.487              GTX 750 Ti 


    0.153      -         Tesla K40m  ( 11520 )

    0.157      0,1,2,3 

                      
              Tesla K40m   (5760)

    0.201      0,1                      
    0.200      2,3                      

    0.179      1,2                     
    0.179      0,2                      
    0.178      1,3                      
     
    0.202      0,1,2
    0.201      0,1,3

    0.134      1,2,3
 


::

    In [1]: 2.516/0.134
    Out[1]: 18.776119402985074




Juno Cerenkov 1, scaledown ?10
---------------------------------

::

    0.126,0.126   0         Tesla K40m  2880 CUDA cores  
    0.127         1
    0.127         2
    0.126         3
  
    0.088,0.087   0,1             5760 
    0.076         0,2
    0.099         2,3
    0.080         1,3

    0.076         0,1,2           8640
    0.058         1,2,3
    0.057         1,2,3

    0.062         0,1,2,3         11520
    0.062,0.062,0.062,0.063   NO ENVVAR
    

    1.130          GT750M    ggv.sh --juno --cmp      384 CUDA cores
    1.143 
    1.146 
    1.137 
    1.139 


    0.195,0.197    GTX 750 Ti    640 CUDA Cores                             


    a = np.array( [[384, 1.130],[640,0.195],[2880,0.126],[5760,0.080],[8640,0.070],[11520,0.062]] )

    plt.plot( a[:,0], a[0,-1]/a[:,1], "*-")



GGeoview Compute 
------------------

Compute only mode::

   ggeoview-compute -b0
   ggeoview-compute -b1
   ggeoview-compute -b2
   ggeoview-compute -b4   # up to 4 bounces working 

   ggeoview-compute -b5   # crash for beyond 4  


Usage tips
-----------

Thoughts on touch mode : OptiX single-ray-cast OR OpenGL depth buffer/unproject 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

OptiX based touch mode, is not so useful operationally (although handy as a debug tool) as:

#. it requires to do an OptiX render before it can operate
#. will usually be using OpenGL rendering to see the geometry often with 
   clipping planes etc.. that only OpenGL knows about.  

Thus need an OpenGL depth buffer unproject approach too.


Low GPU memory running
~~~~~~~~~~~~~~~~~~~~~~~~~~

When GPU memory is low OptiX startup causes a crash, 
to run anyhow disable OptiX with::

    ggeoview-run --optixmode -1

To free up GPU memory restart the machine, or try sleep/unsleep and
exit applications including Safari, Mail that all use GPU memory. 
Observe that sleeping for ~1min rather than my typical few seconds 
frees almost all GPU memory.

Check available GPU memory with **cu** if less than ~512MB OptiX will
crash at startup::

    delta:optixrap blyth$ t cu
    cu is aliased to cuda_info.sh


Clipping Planes and recording frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    udp.py --cutnormal 1,0,0 --eye -2,0,0 --cutpoint 0,0,0
    udp.py --cutnormal 1,1,0 --cutpoint -0.1,-0.1,0


Although model frame coordinates are useful for 
intuitive data entry the fact that the meaning is relative
to the selected geometry makes them less 
useful as a way of recording a plane, 
so record planes in world frame coordinates.

This would allow to find the precise plane that 
halves a piece of geometry by selecting that
and providing a way to accept world planes, could 
use --cutplane x,y,x,w to skip the model_to_world 
conversion.

The same thinking applies to recording viewpoint bookmarks.


New way of interpolated photon position animation ?
----------------------------------------------------

See oglrap- for untested idea using geometry shaders alone.


Old way of doing interpolated photon position animation
-----------------------------------------------------------

* splayed out by maxsteps VBO

* recorded each photon step into its slot 

* pre-render CUDA time-presenter to find before and after 
  positions and interpolate between them writing into specific top slot 
  of the splay.


Problems:

* limited numbers of particles can be animated (perhaps 1000 or so)
  as approach multiplies storage by the max number of steps are kept

* most of the storage is empty, for photons without that many steps 

Advantages:

* splaying out allows CUDA to operate fully concurrently 
  with no atomics complexities, as every photon step has its place 
  in the structure 

* OpenGL can address and draw the VBO using fixed offsets/strides
  pointing at the interpolated slot, geometry shaders can be used to 
  amplify a point and momentum direction into a line


Package Dependencies Tree of GGeoView
--------------------------------------

* higher level repeated dependencies elided for clarity 

::

    NPY*   (~11 classes)
       Boost
       GLM         

    Cfg*  (~1 class)
       Boost 

    numpyserver*  (~7 classes)
       Boost.Asio
       ZMQ
       AsioZMQ
       Cfg*
       NPY*

    cudawrap* (~5 classes)
       CUDA



 
    GGeo*  (~22 classes)
       NPY*

    AssimpWrap* (~7 classes)
       Assimp
       GGeo* 

    OGLRap*  (~29 classes)
       GLEW
       GLFW
       ImGui
       AssimpWrap*
       Cfg*
       NPY*

    OptiXRap* (~7 classes)
       OptiX
       OGLRap*
       AssimpWrap*
       GGeo*    
   


Data Flow thru the app
-------------------------

* Gensteps NPY loaded from file (or network)

* main.NumpyEvt::setGenstepData 

  * determines num_photons
  * allocates NPY arrays for photons, records, sequence, recsel, phosel
    and characterizes content with MultiViewNPY 

* main.Scene::uploadEvt

  * gets genstep, photon and record renderers to upload their respective buffers 
    and translate MultiViewNPY into OpenGL vertex attributes

* main.Scene::uploadSelection

  * recsel upload
  * hmm currently doing this while recsel still all zeroes 

* main.OptiXEngine::initGenerate(NumpyEvt* evt)

  * populates OptiX context, using OpenGL buffer ids lodged in the NPY  
    to create OptiX buffers for each eg::

        m_genstep_buffer = m_context->createBufferFromGLBO(RT_BUFFER_INPUT, genstep_buffer_id);

* main.OptiXEngine::generate, cu/generate.cu

  * fills OptiX buffers: photon_buffer, record_buffer, sequence_buffer

* main.Rdr::download(NPY*)

  * pullback to host NPY the VBO/OptiX buffers using Rdr::mapbuffer 
    Rdr::unmapbuffer to get void* pointers from OpenGL

    * photon, record and sequence buffers are downloaded

* main.ThrustArray::ThrustArray created for: sequence, recsel and phosel 

  * OptiXUtil::getDevicePtr devptr used to allow Thrust to access these OpenGL buffers 
    
* main.ThrustIdx indexes the sequence outputing into phosel and recsel

  * recsel is created from phosel using ThrustArray::repeat_to

* main.Scene::render Rdr::render for genstep, photon, record 

  * glBindVertexArray(m_vao) and glDrawArrays 
  * each renderer has a single m_vao which contains buffer_id and vertex attributes


Issue: recsel changes not seen by OpenGL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The zeroed recsel buffer was uploaded early, it was modified
with Thrust using the below long pipeline but the 
changes to the device buffer where not seen by OpenGL

* NumpyEvt create NPY
* Scene::uploadEvt, Scene::uploadSelection - Rdr::upload (setting buffer_id in the NPY)
* OptiXEngine::init (convert to OptiX buffers)
* OptiXUtil provides raw devptr for use by ThrustArray
* Rdr::render draw shaders do not see the changes to the recsel buffer 

Workaround by simplifying pipeline for non-conforming buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

recsel, phosel do not conform to the pattern of other buffers 

* not needed by OptiX
* only needed in host NPY for debugging 
* phosel is populated on device by ThrustIdx::makeHistogram from the OptiX filled sequence buffer
* recsel is populated on device by ThrustArray::repeat_to on phosel 

Formerly had no way to get buffers into Thrust other than 
going through the full pipeline. Added capability to ThrustArray 
to allocate/resize buffers allowing simpler flow:

* NumpyEvt create NPY (recsel, phosel still created early on host, but they just serve as dimension placeholders)
* allocate recsel and phosel on device with ThrustArray(NULL, NPY dimensions), populate with ThrustIdx
* ThrustArray::download into the recsel and phosel NPY 
* Scene::uploadSelection to upload with OpenGL for use from shaders 

TODO: skip redundant Thrust download, OpenGL upload using CUDA/OpenGL interop ?



C++ library versions
----------------------

::

    delta:~ blyth$ otool -L $(ggeoview-;ggeoview-deps) | grep c++
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 120.0.0)
        /usr/lib/libstdc++.6.dylib (compatibility version 7.0.0, current version 60.0.0)



Pre-cook RNG Cache
-------------------

* currently the work number must precicely match the hardcoded 
  value used for OptiXEngine::setRngMax  

  * TODO: tie these together via envvar


::

    delta:ggeoview blyth$ ggeoview-rng-prep
    cuRANDWrapper::instanciate with cache enabled : cachedir /usr/local/env/graphics/ggeoview.build/lib/rng
    cuRANDWrapper::Allocate
    cuRANDWrapper::InitFromCacheIfPossible
    cuRANDWrapper::InitFromCacheIfPossible : no cache initing and saving 
    cuRANDWrapper::Init
     init_rng_wrapper sequence_index   0  thread_offset       0  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time   138.0750 ms 
    ...



Improvement
-------------

Adopt separate minimal VBO for animation 

* single vec4 (position, time) ? 
* no need for direction as see that from the interpolation, 
* polz, wavelength, ... keep these in separate full-VBO for detailed debug 
  of small numbers of stepped photons 


Does modern OpenGL have any features that allow a better way
--------------------------------------------------------------

* http://gamedev.stackexchange.com/questions/20983/how-is-animation-handled-in-non-immediate-opengl

  * vertex blend-based animation
  * vertex blending
  * use glVertexAttribPointer to pick keyframes, 
  * shader gets two "position" attributes, 
    one for the keyframe in front of the current 
    time and one for the keyframe after and a uniform that specifies 
    how much of a blend to do between them. 

hmm not so easy for photon simulation as they all are on their own timeline :
so not like traditional animation keyframes
 


glDrawArraysIndirect introduced in 4.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* https://www.opengl.org/wiki/History_of_OpenGL#OpenGL_4.1_.282010.29
* https://www.opengl.org/wiki/Vertex_Rendering#Indirect_rendering
* http://stackoverflow.com/questions/5047286/opengl-4-0-gpu-draw-feature
* https://www.opengl.org/registry/specs/ARB/draw_indirect.txt

Indirect rendering is the process of issuing a drawing command to OpenGL,
except that most of the parameters to that command come from GPU storage
provided by a Buffer Object.
The idea is to avoid the GPU->CPU->GPU round-trip; the GPU decides what range
of vertices to render with. All the CPU does is decide when to issue the
drawing command, as well as which Primitive is used with that command.

The indirect rendering functions take their data from the buffer currently
bound to the GL_DRAW_INDIRECT_BUFFER binding. Thus, any of these
functions will fail if no buffer is bound to that binding.

So can tee up a buffer of commands GPU side, following layout::

    void glDrawArraysIndirect(GLenum mode, const void *indirect);

    typedef  struct {
       GLuint  count;
       GLuint  instanceCount;
       GLuint  first;
       GLuint  baseInstance;   // MUST BE 0 IN 4.1
    } DrawArraysIndirectCommand;

Where each cmd is equivalent to::

    glDrawArraysInstancedBaseInstance(mode, cmd->first, cmd->count, cmd->instanceCount, cmd->baseInstance);

Similarly for indirect indexed drawing::

    glDrawElementsIndirect(GLenum mode, GLenum type, const void *indirect);

    typedef  struct {
        GLuint  count;
        GLuint  instanceCount;
        GLuint  firstIndex;
        GLuint  baseVertex;
        GLuint  baseInstance;
    } DrawElementsIndirectCommand;

With each cmd equivalent to:: 

    glDrawElementsInstancedBaseVertexBaseInstance(mode, cmd->count, type,
      cmd->firstIndex * size-of-type, cmd->instanceCount, cmd->baseVertex, cmd->baseInstance);

* https://www.opengl.org/sdk/docs/man/html/glDrawElementsInstancedBaseVertex.xhtml


EOU
}


ggeoview-sdir(){ echo $(env-home)/graphics/ggeoview ; }
ggeoview-idir(){ echo $(local-base)/env/graphics/ggeoview ; }
ggeoview-bdir(){ echo $(ggeoview-idir).build ; }
ggeoview-gdir(){ echo $(ggeoview-idir).generated ; }

#ggeoview-rng-dir(){ echo $(ggeoview-bdir)/lib/rng ; }  gets deleted too often for keeping RNG 
ggeoview-rng-dir(){ echo $(ggeoview-idir)/cache/rng ; }

ggeoview-ptx-dir(){ echo $(ggeoview-bdir)/lib/ptx ; }
ggeoview-rng-ls(){  ls -l $(ggeoview-rng-dir) ; }
ggeoview-ptx-ls(){  ls -l $(ggeoview-ptx-dir) ; }

ggeoview-scd(){  cd $(ggeoview-sdir); }
ggeoview-cd(){  cd $(ggeoview-sdir); }

ggeoview-icd(){  cd $(ggeoview-idir); }
ggeoview-bcd(){  cd $(ggeoview-bdir); }
ggeoview-name(){ echo GGeoView ; }
ggeoview-compute-name(){ echo computeTest ; }
ggeoview-loader-name(){ echo GLoaderTest ; }

ggeoview-wipe(){
   local bdir=$(ggeoview-bdir)
   rm -rf $bdir
}
ggeoview-env(){     
    elocal- 
    optix-
    optix-export
}

ggeoview-options()
{
    case $NODE_TAG in 
      D) echo -DNPYSERVER=ON ;;
    esac
}

ggeoview-cmake(){
   local iwd=$PWD

   local bdir=$(ggeoview-bdir)
   mkdir -p $bdir
  
   ggeoview-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(ggeoview-idir) \
       -DOptiX_INSTALL_DIR=$(optix-install-dir) \
       -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
       $(ggeoview-options) \
       $(ggeoview-sdir)

   cd $iwd
}

ggeoview-make(){
   local iwd=$PWD

   ggeoview-bcd 
   make $*

   cd $iwd
}

ggeoview-install(){
   printf "********************** $FUNCNAME "
   ggeoview-make install
}

ggeoview-bin(){ echo ${GGEOVIEW_BINARY:-$(ggeoview-idir)/bin/$(ggeoview-name)} ; }
ggeoview-compute-bin(){ echo $(ggeoview-idir)/bin/$(ggeoview-compute-name) ; }
ggeoview-loader-bin(){  echo $(ggeoview-idir)/bin/$(ggeoview-loader-name) ; }

ggeoview-accelcache()
{
    ggeoview-export
    ls -l ${DAE_NAME_DYB/.dae}.*.accelcache
}
ggeoview-accelcache-rm()
{
    ggeoview-export
    rm ${DAE_NAME_DYB/.dae}.*.accelcache
}

ggeoview-rng-max()
{
   # maximal number of photons that can be handled : move to cudawrap- ?
    echo $(( 1000*1000*3 ))
}

ggeoview-rng-prep()
{
   cudawrap-
   CUDAWRAP_RNG_DIR=$(ggeoview-rng-dir) CUDAWRAP_RNG_MAX=$(ggeoview-rng-max) $(cudawrap-ibin)
}



ggeoview-idpath()
{
   ggeoview-
   ggeoview-run --idpath 2>/dev/null 
}

ggeoview-steal-bookmarks()
{
   local idpath=$(ggeoview-idpath)
   cp ~/.g4daeview/dyb/bookmarks20141128-2053.cfg $idpath/bookmarks.ini
}



ggeoview-export()
{
   export-
   export-export

   [ "$GGEOVIEW_GEOKEY" == "" ] && echo $msg MISSING ENVVAR GGEOVIEW_GEOKEY && sleep 10000000
   [ "$GGEOVIEW_QUERY"  == "" ] && echo $msg MISSING ENVVAR GGEOVIEW_QUERY && sleep 10000000
   #[ "$GGEOVIEW_CTRL" == "" ]   && echo $msg MISSING ENVVAR GGEOVIEW_CTRL && sleep 10000000

   unset SHADER_DIR 
   unset SHADER_DYNAMIC_DIR 
   unset SHADER_INCL_PATH

   unset RAYTRACE_PTX_DIR
   unset RAYTRACE_RNG_DIR

   export CUDAWRAP_RNG_MAX=$(ggeoview-rng-max)
} 

ggeoview-export-dump()
{
   env | grep GGEOVIEW
   env | grep SHADER
   env | grep RAYTRACE
   env | grep CUDAWRAP

}

ggeoview-run(){ 
   local bin=$(ggeoview-bin)
   ggeoview-export
   $bin $*
}

ggeoview-compute(){ 
   local bin=$(ggeoview-compute-bin)
   ggeoview-export
   $bin $*
}

ggeoview-compute-lldb(){ 
   local bin=$(ggeoview-compute-bin)
   ggeoview-export
   lldb $bin $*
}

ggeoview-compute-gdb(){ 
   local bin=$(ggeoview-compute-bin)
   ggeoview-export
   gdb --args $bin $*
}



ggeoview-vrun(){ 
   local bin=$(ggeoview-bin)
   ggeoview-export
   vglrun $bin $*
}

ggeoview-gdb(){ 
   local bin=$(ggeoview-bin)
   ggeoview-export
   gdb --args $bin $*
}

ggeoview-valgrind(){ 
   local bin=$(ggeoview-bin)
   ggeoview-export
   valgrind $bin $*
}

ggeoview-lldb()
{
   local bin=$(ggeoview-bin)
   ggeoview-export
   lldb $bin -- $*
}

ggeoview-dbg()
{
   case $(uname) in
     Darwin) ggeoview-lldb $* ;;
          *) ggeoview-gdb  $* ;;
   esac
}


ggeoview--()
{
    ggeoview-wipe
    ggeoview-cmake
    ggeoview-make
    ggeoview-install
}


ggeoview-depinstall()
{
    bcfg-
    bcfg-install
    bregex-
    bregex-install
    npy-
    npy-install
    ggeo-
    ggeo-install
    assimpwrap- 
    assimpwrap-install
    oglrap-
    oglrap-install
    cudawrap-
    cudawrap-install 
    thrustrap-
    thrustrap-install 
    optixrap-
    optixrap-install 
    ggeoview-
    ggeoview-install  
}

ggeoview-depcmake()
{
   local dep
   ggeoview-deps- | while read dep ; do
       $dep-
       $dep-cmake
   done
}

ggeoview-deps-(){ cat << EOD
bcfg
bregex
npy
ggeo
assimpwrap
oglrap
optixrap
cudawrap
thrustrap
EOD
}

ggeoview-deps(){
   local suffix=${1:-dylib}
   local dep
   $FUNCNAME- | while read dep ; do
       $dep-
       #printf "%30s %30s \n" $dep $($dep-idir) 
       echo $($dep-idir)/lib/*.${suffix}
   done
}

ggeoview-ls(){   ls -1 $(ggeoview-;ggeoview-deps) ; }
ggeoview-libs(){ otool -L $(ggeoview-;ggeoview-deps) ; }

ggeoview-linux-setup() {
    local dep
    local edeps="boost glew glfw imgui glm assimp"
    local deps="$edeps bcfg bregex npy ggeo assimpwrap ppm oglrap cudawrap optixrap thrustrap"
    for dep in $deps
    do
        $dep-
        [ -d "$($dep-idir)/lib" ] &&  export LD_LIBRARY_PATH=$($dep-idir)/lib:$LD_LIBRARY_PATH
        [ -d "$($dep-idir)/lib64" ] &&  export LD_LIBRARY_PATH=$($dep-idir)/lib64:$LD_LIBRARY_PATH
    done

    assimp-
    export LD_LIBRARY_PATH=$(assimp-prefix)/lib:$LD_LIBRARY_PATH
}

ggeoview-linux-install-external() {
    local edeps="glew glfw imgui glm assimp"
    local edep
    for edep in $edeps
    do
        ${edep}-
        ${edep}-get
        ${edep}--
    done
}
ggeoview-linux-install() {

    local deps="bcfg bregex npy ggeo assimpwrap ppm oglrap cudawrap optixrap thrustrap"
    local dep

    for dep in $deps
    do
        $dep-
        $dep--
    done
}
