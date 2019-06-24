tboolean-proxy-scan-LV10-coincidence-speckle
==================================================


Context
------------

* :doc:`tboolean-proxy-scan-LV10-coincidence-speckle`


Command shortcuts
---------------------

::

    lv(){ echo 21 ; }
    # default geometry LV index to test 

    ts(){  PROXYLV=${1:-$(lv)} tboolean.sh --align --dbgskipclearzero --dbgnojumpzero --dbgkludgeflatzero $* ; } 
    # **simulate** : aligned bi-simulation creating OK+G4 events 

    tv(){  PROXYLV=${1:-$(lv)} tboolean.sh --load $* ; } 
    # **visualize** : load events and visualize the propagation

    tv4(){ tv --vizg4 $* ; }
    # **visualize** the geant4 propagation 

    ta(){  tboolean-;PROXYLV=${1:-$(lv)} tboolean-proxy-ip ; } 
    # **analyse** : load events and analyse the propagation


rejigged shortcuts moving the above options within tboolean-lv
------------------------------------------------------------------

::

    [blyth@localhost ana]$ t opticks-tboolean-shortcuts
    opticks-tboolean-shortcuts is a function
    opticks-tboolean-shortcuts () 
    { 
        : default geometry LV index or tboolean-geomname eg "box" "sphere" etc..;
        function lv () 
        { 
            echo 21
        };
        : **simulate** : aligned bi-simulation creating OK+G4 events;
        function ts () 
        { 
            LV=${1:-$(lv)} tboolean.sh $*
        };
        : **visualize** : load events and visualize the propagation;
        function tv () 
        { 
            LV=${1:-$(lv)} tboolean.sh --load $*
        };
        : **visualize** the geant4 propagation;
        function tv4 () 
        { 
            LV=${1:-$(lv)} tboolean.sh --load --vizg4 $*
        };
        : **analyse** : load events and analyse the propagation;
        function ta () 
        { 
            LV=${1:-$(lv)} tboolean.sh --ip
        }
    }




FIXED ISSUE : COINCIDENCE/SPECKLE stymies history alignment
-----------------------------------------------------------------

* large deviations from a few photons failing to stay in history alignment


tv 10 
~~~~~~~~
     
* shows large flat box with cylinder hole with coincidence speckle problem in the hole
* selecting the Opticks only history "TO BT BT BT SA" can see all 23 photons go into the speckle zone 


ta 10
~~~~~~~~


ta 10, shows history dropout zero::

    tboolean-proxy-10
    .
    ab.his
    .                seqhis_ana  1:tboolean-proxy-10:tboolean-proxy-10   -1:tboolean-proxy-10:tboolean-proxy-10        c2        ab        ba 
    .                              10000     10000         0.09/13 =  0.01  (pval:1.000 prob:0.000)  
    0000             8ccd      7585      7610             0.04        0.997 +- 0.011        1.003 +- 0.012  [4 ] TO BT BT SA
    0001              8bd       510       510             0.00        1.000 +- 0.044        1.000 +- 0.044  [3 ] TO BR SA
    0002            8cbcd       489       492             0.01        0.994 +- 0.045        1.006 +- 0.045  [5 ] TO BT BR BT SA
    0003              86d       467       467             0.00        1.000 +- 0.046        1.000 +- 0.046  [3 ] TO SC SA
    0004            86ccd       447       449             0.00        0.996 +- 0.047        1.004 +- 0.047  [5 ] TO BT BT SC SA
    0005            8cc6d        75        75             0.00        1.000 +- 0.115        1.000 +- 0.115  [5 ] TO SC BT BT SA
    0006          8cc6ccd        67        67             0.00        1.000 +- 0.122        1.000 +- 0.122  [7 ] TO BT BT SC BT BT SA
    0007              4cd        44        44             0.00        1.000 +- 0.151        1.000 +- 0.151  [3 ] TO BT AB
    0008           866ccd        35        35             0.00        1.000 +- 0.169        1.000 +- 0.169  [6 ] TO BT BT SC SC SA
    0009             866d        30        30             0.00        1.000 +- 0.183        1.000 +- 0.183  [4 ] TO SC SC SA
    0010           8cbbcd        26        26             0.00        1.000 +- 0.196        1.000 +- 0.196  [6 ] TO BT BR BR BT SA
    0011             86bd        25        25             0.00        1.000 +- 0.200        1.000 +- 0.200  [4 ] TO BR SC SA
    0012            8cccd        23         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT BT SA
                               ^^^^^^^^^^^^^^^^^^^
    0013           86cbcd        20        20             0.00        1.000 +- 0.224        1.000 +- 0.224  [6 ] TO BT BR BT SC SA
    0014       bbbbbbb6cd        16        15             0.03        1.067 +- 0.267        0.938 +- 0.242  [10] TO BT SC BR BR BR BR BR BR BR
    0015             8b6d        14        14             0.00        1.000 +- 0.267        1.000 +- 0.267  [4 ] TO SC BR SA
    0016           8b6ccd        11        11             0.00        1.000 +- 0.302        1.000 +- 0.302  [6 ] TO BT BT SC BR SA
    0017            8c6cd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [5 ] TO BT SC BT SA
    0018           8cbc6d         8         8             0.00        1.000 +- 0.354        1.000 +- 0.354  [6 ] TO SC BT BR BT SA
    0019         8cbc6ccd         8         8             0.00        1.000 +- 0.354        1.000 +- 0.354  [8 ] TO BT BT SC BT BR BT SA
    .                              10000     10000         0.09/13 =  0.01  (pval:1.000 prob:0.000)  



Selecting those 23 photons::


    In [1]: ab.sel = "TO BT BT BT SA"
    [2019-06-22 15:58:01,776] p268631 {evt.py    :876} WARNING  - _init_selection EMPTY nsel 0 len(psel) 10000 

    In [2]: ab.his
    Out[2]: 
    ab.his
    .                seqhis_ana  1:tboolean-proxy-10:tboolean-proxy-10   -1:tboolean-proxy-10:tboolean-proxy-10        c2        ab        ba 
    .                                 23         0         0.00/-1 =  0.00  (pval:nan prob:nan)  
    0000            8cccd        23         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT BT SA
    .                                 23         0         0.00/-1 =  0.00  (pval:nan prob:nan)  

    In [3]: a.rpost()
    Out[3]: 
    A()sliced
    A([[[    39.5525,   -188.9732, -71998.8026,      0.    ],
        [    39.5525,   -188.9732,  -2500.5993,    231.8339],
        [    39.5525,   -188.9732,   1500.7991,    245.1704],
        [    39.5525,   -188.9732,   2500.5993,    251.2284],
        [    39.5525,   -188.9732,  72001.    ,    483.0622]],

       [[  -239.5126,    -92.2893, -71998.8026,      0.    ],
        [  -239.5126,    -92.2893,  -2500.5993,    231.8339],
        [  -239.5126,    -92.2893,   1500.7991,    245.1704],
        [  -239.5126,    -92.2893,   2500.5993,    251.2284],
        [  -239.5126,    -92.2893,  72001.    ,    483.0622]],

       [[   -46.1446,    -74.7103, -71998.8026,      0.    ],
        [   -46.1446,    -74.7103,  -2500.5993,    231.8339],
        [   -46.1446,    -74.7103,   1500.7991,    245.1704],
        [   -46.1446,    -74.7103,   2500.5993,    251.2284],
        [   -46.1446,    -74.7103,  72001.    ,    483.0622]],

        ... 
   
    In [4]: a.rpost().shape
    Out[4]: (23, 5, 4)


All 23 photons go straight up the middle, directly into the maw of the speckle.

::

    In [6]: (-2500.5993--71998.8026)/231.8339
    Out[6]: 299.7758451201485

    In [8]: (1500.7991--2500.5993)/(245.1704-231.8339)
    Out[8]: 300.0336220147715

    In [9]: (2500.5993-1500.7991)/(251.2284-245.1704)
    Out[9]: 165.03799933971627

    In [10]: (72001.-2500.5993)/(483.0622-251.2284)
    Out[10]: 299.7854527683193


* hmm the extra surface creates an unphysical sequence of boundaries ?



x010 : x4gen generated executable dumps geometry source code
---------------------------------------------------------------

* thin in Z box from -2500mm to 2500mm 
* cylinder hole hz is 2000mm but its shifted by -500mm  in Z 



::

        
             ------------ BT-----------------  2500



             ------.......BT...........-----  1500
                   |                |
                   |                |  
                   |                |  
                   |                |  
                   |                |  
                   |                |  
                   |                |  
                   |                |  
                   |                |  
                   |                |  
             ------|=====[BT]=======|-----  -2500
                     Maw of speckle
             .     
                          ^
                          ^
                          ^ 
                          ^

::

    [blyth@localhost issues]$ x010
    2019-06-22 16:05:10.066 INFO  [281241] [Opticks::init@313] INTEROP_MODE
    2019-06-22 16:05:10.067 INFO  [281241] [Opticks::configure@1766]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
    NCSGList::savesrc csgpath /tmp/blyth/location/x4gen/x010 verbosity 0 numTrees 2
    2019-06-22 16:05:10.081 INFO  [281241] [NCSG::savesrc@282]  treedir_ /tmp/blyth/location/x4gen/x010/0
    2019-06-22 16:05:10.082 INFO  [281241] [NCSG::savesrc@282]  treedir_ /tmp/blyth/location/x4gen/x010/1
    analytic=1_csgpath=/tmp/blyth/location/x4gen/x010
    2019-06-22 16:05:10.083 INFO  [281241] [X4CSG::dumpTestMain@253] X4CSG::dumpTestMain

    // generated by X4CSG::generateTestMain see x4gen-vi for CMakeLists.txt generation and building 
    ...


    // gdml from X4GDMLParser::ToString(G4VSolid*)  
    const std::string gdml = R"( 
    <?xml version="1.0" encoding="UTF-8" standalone="no" ?>
    <gdml xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="SchemaLocation">

      <solids>
        <box lunit="mm" name="BoxsAirTT0x5b33e60" x="48000" y="48000" z="5000"/>
        <tube aunit="deg" deltaphi="360" lunit="mm" name="Cylinder0x5b33ef0" rmax="500" rmin="0" startphi="0" z="4000"/>
        <subtraction name="sAirTT0x5b34000">
          <first ref="BoxsAirTT0x5b33e60"/>
          <second ref="Cylinder0x5b33ef0"/>
          <position name="sAirTT0x5b34000_pos" unit="mm" x="0" y="0" z="-500"/>
        </subtraction>
      </solids>

    </gdml>

    )" ; 
    // start portion generated by nnode::to_g4code 
    G4VSolid* make_solid()
    { 
        G4VSolid* b = new G4Box("BoxsAirTT0x5b33e60", 24000.000000, 24000.000000, 2500.000000) ; // 1
        G4VSolid* d = new G4Tubs("Cylinder0x5b33ef0", 0.000000, 500.000000, 2000.000000, 0.000000, CLHEP::twopi) ; // 1
        
        G4ThreeVector A(0.000000,0.000000,-500.000000);
        G4VSolid* a = new G4SubtractionSolid("sAirTT0x5b34000", b, d, NULL, A) ; // 0
        return a ; 
    } 
    // end portion generated by nnode::to_g4code 

    ...



x010 : GDML geometry change to make opticksdata-jv5
-----------------------------------------------------------

* obvious fix, increase cylinder hz to 2001mm and the shift to -501mm
  so end up with same geometry but the subtracted cylinder face is
  no longer coincident at -2500 and its at -2501 

* see geocache-j1808-v5-notes showing the GDML change  


Original::

      <solids>
        <box lunit="mm" name="BoxsAirTT0x5b33e60" x="48000" y="48000" z="5000"/>
        <tube aunit="deg" deltaphi="360" lunit="mm" name="Cylinder0x5b33ef0" rmax="500" rmin="0" startphi="0" z="4000"/>
        <subtraction name="sAirTT0x5b34000">
          <first ref="BoxsAirTT0x5b33e60"/>
          <second ref="Cylinder0x5b33ef0"/>
          <position name="sAirTT0x5b34000_pos" unit="mm" x="0" y="0" z="-500"/>
        </subtraction>
      </solids>


Amended::

      <solids>
        <box lunit="mm" name="BoxsAirTT0x5b33e60" x="48000" y="48000" z="5000"/>
        <tube aunit="deg" deltaphi="360" lunit="mm" name="Cylinder0x5b33ef0" rmax="500" rmin="0" startphi="0" z="4002"/>
        <subtraction name="sAirTT0x5b34000">
          <first ref="BoxsAirTT0x5b33e60"/>
          <second ref="Cylinder0x5b33ef0"/>
          <position name="sAirTT0x5b34000_pos" unit="mm" x="0" y="0" z="-501"/>
        </subtraction>
      </solids>



geocache-recreate
-------------------

After added v5 funcs to geocache.bash

::

    2019-06-22 17:00:18.875 INFO  [366592] [OpticksProfile::dump@170]  npy 55,1,4 /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1/source/evt/g4live/torch/Opticks.npy
    2019-06-22 17:00:18.875 INFO  [366592] [Opticks::reportGeoCacheCoordinates@727]  ok.idpath  /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/f6cc352e44243f8fa536ab483ad390ce/1
    2019-06-22 17:00:18.875 INFO  [366592] [Opticks::reportGeoCacheCoordinates@728]  ok.keyspec OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    2019-06-22 17:00:18.875 INFO  [366592] [Opticks::reportGeoCacheCoordinates@729]  To reuse this geometry: 
    2019-06-22 17:00:18.875 INFO  [366592] [Opticks::reportGeoCacheCoordinates@730]    1. set envvar OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    2019-06-22 17:00:18.875 INFO  [366592] [Opticks::reportGeoCacheCoordinates@731]    2. enable envvar sensitivity with --envkey argument to Opticks executables 
    2019-06-22 17:00:18.875 FATAL [366592] [Opticks::reportGeoCacheCoordinates@739] THE LIVE keyspec DOES NOT MATCH THAT OF THE CURRENT ENVVAR 
    2019-06-22 17:00:18.875 INFO  [366592] [Opticks::reportGeoCacheCoordinates@740]  (envvar) OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.
    2019-06-22 17:00:18.875 INFO  [366592] [Opticks::reportGeoCacheCoordinates@741]  (live)   OPTICKS_KEY=OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.f6cc352e44243f8fa536ab483ad390ce
    2019-06-22 17:00:18.875 INFO  [366592] [Opticks::dumpRC@202]  rc 0 rcmsg : -
    2019-06-22 17:00:21.537 INFO  [366592] [OContext::cleanUpCache@466]  RemoveDir /var/tmp/OptixCache
    === o-main : /home/blyth/local/opticks/lib/OKX4Test --okx4 --g4codegen --deletegeocache --gdmlpath /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v5.gdml --csgskiplv 22 --runfolder geocache-j1808-v5 --runcomment fix-lv10-coincidence-speckle ======= PWD /tmp/blyth/opticks/geocache-create- RC 0 Sat Jun 22 17:00:21 CST 2019
    echo o-postline : dummy
    o-postline : dummy


Regenerate the x4gen executables, improved to skip zero length files like x016.cc::

   x4gen--

x010 shows the expected change


rerun simulation
--------------------

::

   ts 10 


With the bad geometry::

    ab.his
    .                seqhis_ana  1:tboolean-proxy-10:tboolean-proxy-10   -1:tboolean-proxy-10:tboolean-proxy-10        c2        ab        ba 
    .                              10000     10000         0.09/13 =  0.01  (pval:1.000 prob:0.000)  
    0000             8ccd      7585      7610             0.04        0.997 +- 0.011        1.003 +- 0.012  [4 ] TO BT BT SA
    0001              8bd       510       510             0.00        1.000 +- 0.044        1.000 +- 0.044  [3 ] TO BR SA
    0002            8cbcd       489       492             0.01        0.994 +- 0.045        1.006 +- 0.045  [5 ] TO BT BR BT SA
    0003              86d       467       467             0.00        1.000 +- 0.046        1.000 +- 0.046  [3 ] TO SC SA
    0004            86ccd       447       449             0.00        0.996 +- 0.047        1.004 +- 0.047  [5 ] TO BT BT SC SA
    0005            8cc6d        75        75             0.00        1.000 +- 0.115        1.000 +- 0.115  [5 ] TO SC BT BT SA
    0006          8cc6ccd        67        67             0.00        1.000 +- 0.122        1.000 +- 0.122  [7 ] TO BT BT SC BT BT SA
    0007              4cd        44        44             0.00        1.000 +- 0.151        1.000 +- 0.151  [3 ] TO BT AB
    0008           866ccd        35        35             0.00        1.000 +- 0.169        1.000 +- 0.169  [6 ] TO BT BT SC SC SA
    0009             866d        30        30             0.00        1.000 +- 0.183        1.000 +- 0.183  [4 ] TO SC SC SA
    0010           8cbbcd        26        26             0.00        1.000 +- 0.196        1.000 +- 0.196  [6 ] TO BT BR BR BT SA
    0011             86bd        25        25             0.00        1.000 +- 0.200        1.000 +- 0.200  [4 ] TO BR SC SA
    0012            8cccd        23         0             0.00        0.000 +- 0.000        0.000 +- 0.000  [5 ] TO BT BT BT SA
                               ^^^^^^^^^^^^^^^^^^^
    0013           86cbcd        20        20             0.00        1.000 +- 0.224        1.000 +- 0.224  [6 ] TO BT BR BT SC SA
    0014       bbbbbbb6cd        16        15             0.03        1.067 +- 0.267        0.938 +- 0.242  [10] TO BT SC BR BR BR BR BR BR BR
    0015             8b6d        14        14             0.00        1.000 +- 0.267        1.000 +- 0.267  [4 ] TO SC BR SA
    0016           8b6ccd        11        11             0.00        1.000 +- 0.302        1.000 +- 0.302  [6 ] TO BT BT SC BR SA
    0017            8c6cd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [5 ] TO BT SC BT SA
    0018           8cbc6d         8         8             0.00        1.000 +- 0.354        1.000 +- 0.354  [6 ] TO SC BT BR BT SA
    0019         8cbc6ccd         8         8             0.00        1.000 +- 0.354        1.000 +- 0.354  [8 ] TO BT BT SC BT BR BT SA
    .                              10000     10000         0.09/13 =  0.01  (pval:1.000 prob:0.000)  


With ambigous coindent geometry fixed avoid the dropout (note that G4 didnt change), 
now are staying aligned except a single photon big bouncer::

    Rock//perfectAbsorbSurface/Vacuum,Vacuum///GlassSchottF2
    tboolean-proxy-10
    .
    ab.his
    .                seqhis_ana  1:tboolean-proxy-10:tboolean-proxy-10   -1:tboolean-proxy-10:tboolean-proxy-10        c2        ab        ba 
    .                              10000     10000         0.03/13 =  0.00  (pval:1.000 prob:0.000)  
    0000             8ccd      7610      7610             0.00        1.000 +- 0.011        1.000 +- 0.011  [4 ] TO BT BT SA
    0001              8bd       510       510             0.00        1.000 +- 0.044        1.000 +- 0.044  [3 ] TO BR SA
    0002            8cbcd       492       492             0.00        1.000 +- 0.045        1.000 +- 0.045  [5 ] TO BT BR BT SA
    0003              86d       467       467             0.00        1.000 +- 0.046        1.000 +- 0.046  [3 ] TO SC SA
    0004            86ccd       449       449             0.00        1.000 +- 0.047        1.000 +- 0.047  [5 ] TO BT BT SC SA
    0005            8cc6d        75        75             0.00        1.000 +- 0.115        1.000 +- 0.115  [5 ] TO SC BT BT SA
    0006          8cc6ccd        67        67             0.00        1.000 +- 0.122        1.000 +- 0.122  [7 ] TO BT BT SC BT BT SA
    0007              4cd        44        44             0.00        1.000 +- 0.151        1.000 +- 0.151  [3 ] TO BT AB
    0008           866ccd        35        35             0.00        1.000 +- 0.169        1.000 +- 0.169  [6 ] TO BT BT SC SC SA
    0009             866d        30        30             0.00        1.000 +- 0.183        1.000 +- 0.183  [4 ] TO SC SC SA
    0010           8cbbcd        26        26             0.00        1.000 +- 0.196        1.000 +- 0.196  [6 ] TO BT BR BR BT SA
    0011             86bd        25        25             0.00        1.000 +- 0.200        1.000 +- 0.200  [4 ] TO BR SC SA
    0012           86cbcd        20        20             0.00        1.000 +- 0.224        1.000 +- 0.224  [6 ] TO BT BR BT SC SA
    0013       bbbbbbb6cd        16        15             0.03        1.067 +- 0.267        0.938 +- 0.242  [10] TO BT SC BR BR BR BR BR BR BR
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    0014             8b6d        14        14             0.00        1.000 +- 0.267        1.000 +- 0.267  [4 ] TO SC BR SA
    0015           8b6ccd        11        11             0.00        1.000 +- 0.302        1.000 +- 0.302  [6 ] TO BT BT SC BR SA
    0016            8c6cd         9         9             0.00        1.000 +- 0.333        1.000 +- 0.333  [5 ] TO BT SC BT SA
    0017           8cbc6d         8         8             0.00        1.000 +- 0.354        1.000 +- 0.354  [6 ] TO SC BT BR BT SA
    0018         8cbc6ccd         8         8             0.00        1.000 +- 0.354        1.000 +- 0.354  [8 ] TO BT BT SC BT BR BT SA
    0019           86cc6d         7         7             0.00        1.000 +- 0.378        1.000 +- 0.378  [6 ] TO SC BT BT SC SA
    .                              10000     10000         0.03/13 =  0.00  (pval:1.000 prob:0.000)  
    ab.flg


Ray trace shows the speckle is gone.

Run the sim again with auto time domain::

    TMAX=-1 ts 10 


