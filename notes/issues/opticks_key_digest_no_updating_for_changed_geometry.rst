opticks_key_digest_no_updating_for_changed_geometry
=========================================================


Original Context
----------------------

* :doc:`torus_replacement_on_the_fly`


REOPEN : Changing csgskiplv not changing digest
----------------------------------------------------

* Context :doc:`review-analytic-geometry`

Currently the spec geometry digest (aka the OPTICKS_KEY) 
depends only on the input GDML geometry or rather the in-memory geometry tree 
that G4 constructs from the GDML. So its an input geometry digest, not 
an output one : as arguments such as csgskiplv can change the output geometry
resulting from the translation. 

:: 

    066 int main(int argc, char** argv)
     67 {
     68     OPTICKS_LOG(argc, argv);
     69 
     70     for(int i=0 ; i < argc ; i++)
     71         LOG(info) << i << " " << argv[i] ;
     72 
     73 
     74     const char* gdmlpath = PLOG::instance->get_arg_after("--gdmlpath", NULL) ;
     75     if( gdmlpath == NULL ) 
     76     {
     77         LOG(fatal) << " --gdmlpath existing-path : is required " ;
     78         return 0 ; 
     79     }   
     80 
     81     LOG(info) << " parsing " << gdmlpath ;
     82     G4VPhysicalVolume* top = CGDML::Parse( gdmlpath ) ;
     83     assert(top);
     84     LOG(info) << "///////////////////////////////// " ;
     85     
     86 
     87     const char* spec = X4PhysicalVolume::Key(top) ;
     88     
     89     Opticks::SetKey(spec);
     90     
     91     LOG(error) << " SetKey " << spec  ;
     92 
     93     const char* argforce = "--tracer --nogeocache --xanalytic" ;
     94     // --nogeoache to prevent GGeo booting from cache 
     95 
     96     Opticks* ok = new Opticks(argc, argv, argforce);  // Opticks instanciation must be after Opticks::SetKey
     97     ok->configure();
     98     
     99     const char* csgskiplv = ok->getCSGSkipLV();
    100     LOG(info) << " csgskiplv " << ( csgskiplv ? csgskiplv : "NONE" ) ;
    101     
    102 
    103     ok->profile("_OKX4Test:GGeo");
    104     
    105     GGeo* gg = new GGeo(ok) ;
    106     assert(gg->getMaterialLib());




Hmm : at what level to form the digest ? 
---------------------------------------------------------------------------------------------------------------------------------

Can digest at multiple levels:

1. GDML file with SDigest::DigestPath
2. in memory Geant4 tree (**currently used**, being created by traversing the Geant4 geometry tree)
3. GGeo level, bunch of mergedmesh and analytic GParts 
4. geocache level : the GParts analytic serialization buffers 
5. OGeo level, analytic GPU buffers  (THIS LOOKS THE BEST WAY : IT IS WHAT REALLY MATTERS )
6. hybrid fake : combine the current digest with digest of geometry changing arguments such as --csgskiplv 

   * this makes the distinctly flawed assumption that the code of the translation is fixed


Practical issues

1. need to be able to reconstruct the digest from a loaded geocache, 
   to check integrity (as the filepath will have the digest too) : this would suggest doing 
  
2. hmm to be able to rapidly identify a changed/or-not GDML file 


What is the digest for ?

1. to identify if some processing stage needs to be done again because the 
   input to that stage has changed


Thoughts

* seems like need to have digests at each level for maximal usefulness and clarity, and 
  to minimize processing    




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



