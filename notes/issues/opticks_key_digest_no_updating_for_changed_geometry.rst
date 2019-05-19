opticks_key_digest_no_updating_for_changed_geometry
=========================================================


Context
----------

* :doc:`torus_replacement_on_the_fly`


REOPEN : Changing csgskiplv not changing digest
----------------------------------------------------

* :doc:`review-analytic-geometry`



Fix Attempt
---------------

Get the SDigest pointer passed along recursive heirarchy in
the hope of making it a full tree digest.


Issue : removed torus but still same digest
-----------------------------------------------

::

    blyth@localhost tests]$ geocache-;geocache-j1808-v3
    geocache-j1808-v3 is a function
    geocache-j1808-v3 () 
    { 
        local iwd=$PWD;
        local tmp=$(geocache-tmp $FUNCNAME);
        mkdir -p $tmp && cd_func $tmp;
        type $FUNCNAME;
        opticksdata-;
        gdb --args OKX4Test --gdmlpath $(opticksdata-jv3) --csgskiplv 22;
        cd_func $iwd
    }
    GNU gdb (GDB) Red Hat Enterprise Linux 7.6.1-114.el7

    (gdb) r
    Starting program: /home/blyth/local/opticks/lib/OKX4Test --gdmlpath /home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v3.gdml --csgskiplv 22
    G4GDML: Reading '/home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v3.gdml'...
    G4GDML: Reading definitions...
    G4GDML: Reading materials...
    G4GDML: Reading solids...
    G4GDML: Reading structure...
    G4GDML: Reading setup...
    G4GDML: Reading '/home/blyth/local/opticks/opticksdata/export/juno1808/g4_00_v3.gdml' done!
    2019-04-18 22:08:27.634 INFO  [409112] [main@86] ///////////////////////////////// 
    2019-04-18 22:08:29.073 ERROR [409112] [main@93]  SetKey OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
    ...
    2019-04-18 22:12:02.168 ERROR [409112] [OpticksHub::cleanup@991] OpticksHub::cleanup
    2019-04-18 22:12:02.235 INFO  [409112] [Opticks::cleanup@2276] Opticks::cleanup
    2019-04-18 22:12:02.235 INFO  [409112] [Opticks::cleanup@2277] Opticks.desc
                 BOpticksKey  : KEYSOURCE
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a
                     exename  : OKX4Test
             current_exename  : OKX4Test
                       class  : X4PhysicalVolume
                     volname  : lWorld0x4bc2710_PV
                      digest  : 528f4cefdac670fffe846377973af10a
                      idname  : OKX4Test_lWorld0x4bc2710_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    IdPath : /home/blyth/local/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1


::

    [blyth@localhost issues]$ echo $OPTICKS_KEY
    OKX4Test.X4PhysicalVolume.lWorld0x4bc2710_PV.528f4cefdac670fffe846377973af10a



