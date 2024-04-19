build_and_test_opticks_with_different_versions_of_externals
=============================================================


Overview
----------

It is convenient to test non-standard configs using separate user accounts 
with a simple environment setup. 

Currenty need to edit .bashrc to switch from build to usage 



R : "R" stood for Release originally, it now just means some other version config under test
----------------------------------------------------------------------------------------------

NB the "simon" account uses symbolically linked "opticks" from "blyth" account::

   opticks -> /home/blyth/opticks


::

    epsilon:sysrap blyth$ t R
    R () 
    { 
        _R
    }
    epsilon:sysrap blyth$ t _R
    _R () 
    { 
        ssh R;
        [ $? -ne 0 ] && echo \"ssh R\" gives connection refused if ssh tunnel \"tun\" is not running
    }
    epsilon:sysrap blyth$ 



Connect "simon@P" (aka "R") and check env
-------------------------------------------

::

   R        ##  "R" stood for Release originally
   vip     ## check settings

vip::

    .opticks_externals_config
    .opticks_build_config
    .opticks_usage_config


Run the fresh build::

   ./fresh_build.sh


That deletes the opticks prefix dir and does opticks-full::

     20 echo $OPTICKS_BUILDTYPE
     21 
     22 if [ "$OPTICKS_BUILDTYPE" == "Release" ]; then
     23     rm -rf local/opticks_Release
     24 elif [ "$OPTICKS_BUILDTYPE" == "Debug" ]; then
     25     rm -rf local/opticks_Debug
     26 else
     27     echo $BASH_SOURCE : UNEXPECTED OPTICKS_BUILDTYPE $OPTICKS_BUILDTYPE
     28 fi
     29 
     30 opticks-full
     31 
     32 #opticks-t
     33 
     34 opticks-tar


It only takes around 5 min, as it uses the download cache. 


Check the usage env : normally want same Release/Build that you just built, but not always
---------------------------------------------------------------------------------------------

* after changing env, keep things simple by exiting session and reconnecting with "R"



Run test : ana fail
--------------------

::

    [simon@localhost ~]$ gxt
    /home/simon/opticks/g4cx/tests
    [simon@localhost tests]$ ./G4CXTest_raindrop_CPU.sh

  
Had to config python/ipython for analysis to work:: 

     PYTHONPATH=$HOME IPYTHON=/home/blyth/local/env/tools/conda/miniconda3/bin/ipython ./G4CXTest_raindrop.sh ana

Put those into .opticks_usage_config::

    SELECT="TO BR BT SA" PICK=B ~/o/g4cx/tests/G4CXTest_raindrop.sh ana

    PICK=B MODE=3 SELECT="TO BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BR : 224.901 224.901 
    speed min/max for : 1 -> 2 : BR -> BT : 224.901 224.901 
    speed min/max for : 2 -> 3 : BT -> SA : 299.792 299.793 

Remove the fix via envvar::

    [simon@localhost ~]$ export U4Recorder__PreUserTrackingAction_Optical_DISABLE_UseGivenVelocity=1 

Yep that gives the wrong velocities::

    [simon@localhost ~]$ SELECT="TO BR BT SA" PICK=B ~/o/g4cx/tests/G4CXTest_raindrop.sh 
    ..

    PICK=B MODE=3 SELECT="TO BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BR : 224.901 224.901 
    speed min/max for : 1 -> 2 : BR -> BT : 299.792 299.793 
    speed min/max for : 2 -> 3 : BT -> SA : 224.901 224.901 
    _pos.shape (46578, 3) 


Now rebuild opticks with G4 11.2
----------------------------------

1. switch .bashrc to "build"
2. adjust .opticks_externals_config to use new Geant4, CLHEP and no Custom4 
3. ./fresh_build.sh 

After fixing a few issues to get to work with 1120:

* ~/o/notes/issues/GDXML_not_building_with_Geant4_1120.rst
* ~/o/notes/issues/U4Touchable_not_compiling_with_Geant4_1120.rst

::

    PICK=B MODE=3 SELECT="TO BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BR : 224.901 224.901 
    speed min/max for : 1 -> 2 : BR -> BT : 224.901 224.901 
    speed min/max for : 2 -> 3 : BT -> SA : 299.792 299.793 
    _pos.shape (46578, 3) 

Removing the fix, and the velocities still broken in the same way:: 

    export U4Recorder__PreUserTrackingAction_Optical_DISABLE_UseGivenVelocity=1

    PICK=B MODE=3 SELECT="TO BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    speed min/max for : 0 -> 1 : TO -> BR : 224.901 224.901 
    speed min/max for : 1 -> 2 : BR -> BT : 299.792 299.793 
    speed min/max for : 2 -> 3 : BT -> SA : 224.901 224.901 
    _pos.shape (46578, 3) 



Permissions issue : from using symbolic linked .opticks/GEOM
--------------------------------------------------------------

Hmm, permissions issue::

    2024-04-02 15:34:35.482 INFO  [52564] [U4GDML::write@285]  ekey U4GDML_GDXML_FIX_DISABLE U4GDML_GDXML_FIX_DISABLE 0 U4GDML_GDXML_FIX 1
    2024-04-02 15:34:35.482 INFO  [52564] [U4GDML::write_@308] [
    2024-04-02 15:34:35.482 FATAL [52564] [U4GDML::write_@312]  FAILED TO REMOVE PATH [/home/simon/.opticks/GEOM/RaindropRockAirWater/origin_raw.gdml] CHECK PERMISSIONS 
    2024-04-02 15:34:35.482 INFO  [52564] [U4GDML::write_@317]  path /home/simon/.opticks/GEOM/RaindropRockAirWater/origin_raw.gdml exists YES rc -1
    G4GDML: Writing '/home/simon/.opticks/GEOM/RaindropRockAirWater/origin_raw.gdml'...

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : InvalidSetup
          issued by : G4GDMLWrite::Write()
    File '/home/simon/.opticks/GEOM/RaindropRockAirWater/origin_raw.gdml' already exists!
    *** Fatal Exception *** core dump ***
     **** Track information is not available at this moment
     **** Step information is not available at this moment

    -------- EEEE -------- G4Exception-END --------- EEEE -------

::

    [simon@localhost .opticks]$ l
    total 0
    lrwxrwxrwx. 1 simon simon 25 Nov  6 15:24 GEOM -> /home/blyth/.opticks/GEOM
    lrwxrwxrwx. 1 simon simon 33 Nov  6 15:40 InputPhotons -> /home/blyth/.opticks/InputPhotons
    lrwxrwxrwx. 1 simon simon 30 Nov  6 15:40 precooked -> /home/blyth/.opticks/precooked
    lrwxrwxrwx. 1 simon simon 29 Dec  7 22:31 rngcache -> /home/blyth/.opticks/rngcache
    drwxrwxr-x. 3 simon simon 17 Jul  4  2020 rngcache_local
    drwxrwxr-x. 2 simon simon 46 Sep 23  2021 runcache
    drwxr-xr-x. 2 simon simon 25 Oct 31 19:41 scontext
    [simon@localhost .opticks]$ 


Make .opticks/GEOM reaL::

    [simon@localhost .opticks]$ rm GEOM
    [simon@localhost .opticks]$ mkdir GEOM
    [simon@localhost .opticks]$ cd GEOM
    [simon@localhost GEOM]$ ln -s /home/blyth/.opticks/GEOM/J23_1_0_rc3_ok0
    [simon@localhost GEOM]$ 


