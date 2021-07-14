zike_reports_slow_surface_initization_with_geant4_1071_when_have_thousands_of_surfaces
=========================================================================================


Addressed this issue with:

* sysrap/SLabelCache.cc
* extg4/X4.cc   
* extg4/tests/X4SurfaceTest.cc


::

    324 SLabelCache<int>* X4::MakeSurfaceIndexCache()
    325 {
    326     SLabelCache<int>* _cache = new SLabelCache<int>(MISSING_SURFACE) ;
    327     
    328     size_t num_lbs = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ;
    329     size_t num_sks = G4LogicalSkinSurface::GetNumberOfSkinSurfaces() ; 
    330     LOG(LEVEL) << "[ " << " num_lbs " << num_lbs << " num_sks " << num_sks ;
    331     
    332     const G4LogicalBorderSurfaceTable* lbs_table = G4LogicalBorderSurface::GetSurfaceTable() ;
    333     const std::vector<G4LogicalBorderSurface*>* lbs_vec = X4LogicalBorderSurfaceTable::PrepareVector(lbs_table);
    334     assert( num_lbs == lbs_vec->size() );
    335     
    336     for(int ibs=0 ; ibs < lbs_vec->size() ; ibs++)
    337     {
    338         const G4LogicalBorderSurface* bs = (*lbs_vec)[ibs] ;
    339         _cache->add( bs, ibs ); 
    340     }   
    341     
    342     const G4LogicalSkinSurfaceTable*   sks_vec = G4LogicalSkinSurface::GetSurfaceTable() ;
    343     assert( num_sks == sks_vec->size() );
    344     for(int isk=0 ; isk < sks_vec->size() ; isk++)
    345     {
    346         const G4LogicalSkinSurface* sk = (*sks_vec)[isk] ;
    347         _cache->add( sk, isk + num_lbs ); 
    348     }
    349 
    350     LOG(LEVEL) << "]" ;
    351     return _cache ;
    352 }






Zike report::

    Hi Simon,


    Recently, I set my geometry to full size consisting of more than 50K sensitive tubes. And I get this:

    ....................

    46525 :                                                                        OPsurface :                                                                        OPsurface
    46526 :                                                                        OPsurface :                                                                        OPsurface
    46527 :                                                                        OPsurface :                                                                        OPsurface
    46528 :                                                                        OPsurface :                                                                        OPsurface
    46529 :                                                                        OPsurface :                                                                        OPsurface
    46530 :                                                                        OPsurface :                                                                        OPsurface
    46531 :                                                                        OPsurface :                                                                        OPsurface
    46532 :                                                                        OPsurface :                                                                        OPsurface
    46533 :                                                                        OPsurface :                                                                        OPsurface
    46534 :                                                                        OPsurface :                                                                        OPsurface
    46535 :                                                                        OPsurface :                                                                        OPsurface
    46536 :                                                                        OPsurface :                                                                        OPsurface
    46537 :                                                                        OPsurface :                                                                        OPsurface
    46538 :                                                                        OPsurface :                          ^C
    sblinux@30423:~/mine/workspace/buildforneT$ 

    Then, I tried to use command:

    sblinux@30423:~/mine/workspace/buildforneT$ time ./neutrinoT 0 >/dev/null 2>&1

    It still ran more than 1 hour and didn't end, so I killed the process . In my
    previous experience, this 10TeV events only cost 3 mins on my PC. I think this
    slowdown may be caused by geometry.  So, is there any way to accelerate it?



    Hi Zike, 

    When you report problems you need to do so in a way that enables me to reproduce issues.  
    The information you have given does not allow me to do that.  

    For example say the command that you are running and point me to the code and
    scripts to build and run the code. You should keep all this in a git repository 
    on bitbucket/github or elsewhere.   Where is your repository ? 

    Note that I have been making rather large changes to Opticks over the past week or so.
    It is always possible that you updated your code at an unfortuate moment, so when you 
    find troubles the first thing to do is update and try again.

    To debug hangs, you need to run under gdb and interrupt with ctrl-C and make a backtrace 
    with “bt”  then you continue with “c” and interrupt again and get another backtrace.
    Doing this repeatedly will show the cause of the hang by showing where the code is stuck.
    Then you can send me the backtraces.



gdb backtrace during hang::


    Program received signal SIGINT, Interrupt.
    0x00007ffff3abd1e7 in __GI___libc_write (fd=1, buf=0x555555686840, nbytes=172) at ../sysdeps/unix/sysv/linux/write.c:26
    26	../sysdeps/unix/sysv/linux/write.c: 没有那个文件或目录.
        at fileops.c:426
    #4  _IO_new_do_write (fp=fp@entry=0x7ffff3b986a0 <_IO_2_1_stdout_>, data=0x555555686840 "28742 :", ' ' <repeats 72 times>, "OPsurface :", ' ' <repeats 72 times>, "OPsurface\n       GROUPVEL \n   1 :     "..., 
        to_do=172) at fileops.c:423
    #5  0x00007ffff3a40013 in _IO_new_file_overflow (f=0x7ffff3b986a0 <_IO_2_1_stdout_>, ch=10) at fileops.c:784
    #6  0x00007ffff3e60259 in std::ostream::put(char) () from /lib/x86_64-linux-gnu/libstdc++.so.6
    #7  0x00007ffff3e604d8 in std::basic_ostream<char, std::char_traits<char> >& std::endl<char, std::char_traits<char> >(std::basic_ostream<char, std::char_traits<char> >&) ()
       from /lib/x86_64-linux-gnu/libstdc++.so.6
    #8  0x00007ffff5332d45 in X4NameOrder<G4LogicalBorderSurface>::Dump (msg=0x7ffff5368448 "X4LogicalBorderSurfaceTable::PrepareVector after sort", a=std::vector of length 55016, capacity 65536 = {...})
        at /home/sblinux/opticks/extg4/X4NameOrder.hh:51
    #9  0x00007ffff53323e8 in X4LogicalBorderSurfaceTable::PrepareVector (tab=0x555555ab6530) at /home/sblinux/opticks/extg4/X4LogicalBorderSurfaceTable.cc:89
    #10 0x00007ffff530ca83 in X4::GetOpticksIndex (surf=0x5555565d5cc0) at /home/sblinux/opticks/extg4/X4.cc:315
    #11 0x00007ffff53353d3 in X4LogicalBorderSurface::Convert (src=0x5555565d5cc0, mode=71 'G') at /home/sblinux/opticks/extg4/X4LogicalBorderSurface.cc:44
    #12 0x00007ffff53326a0 in X4LogicalBorderSurfaceTable::init (this=0x7fffffffba90) at /home/sblinux/opticks/extg4/X4LogicalBorderSurfaceTable.cc:124
    #13 0x00007ffff5332457 in X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable (this=0x7fffffffba90, dst=0x555558ecc910, mode=71 'G') at /home/sblinux/opticks/extg4/X4LogicalBorderSurfaceTable.cc:107
    #14 0x00007ffff5332241 in X4LogicalBorderSurfaceTable::Convert (dst=0x555558ecc910, mode=71 'G') at /home/sblinux/opticks/extg4/X4LogicalBorderSurfaceTable.cc:43
    #15 0x00007ffff5340cc1 in X4PhysicalVolume::convertSurfaces (this=0x7fffffffc020) at /home/sblinux/opticks/extg4/X4PhysicalVolume.cc:493
    #16 0x00007ffff533f422 in X4PhysicalVolume::init (this=0x7fffffffc020) at /home/sblinux/opticks/extg4/X4PhysicalVolume.cc:195
    #17 0x00007ffff533f150 in X4PhysicalVolume::X4PhysicalVolume (this=0x7fffffffc020, ggeo=0x555563761c50, top=0x555555ab05a0) at /home/sblinux/opticks/extg4/X4PhysicalVolume.cc:178
    #18 0x00007ffff7993899 in G4Opticks::translateGeometry (this=0x555558223bc0, top=0x555555ab05a0) at /home/sblinux/opticks/g4ok/G4Opticks.cc:949
    #19 0x00007ffff799210a in G4Opticks::setGeometry (this=0x555558223bc0, world=0x555555ab05a0) at /home/sblinux/opticks/g4ok/G4Opticks.cc:593
    #20 0x00007ffff7991f80 in G4Opticks::setGeometry (this=0x555558223bc0, world=0x555555ab05a0, standardize_geant4_materials=false) at /home/sblinux/opticks/g4ok/G4Opticks.cc:585
    #21 0x00005555555838b8 in neTRunAction::BeginOfRunAction (this=0x555555a192f0) at /home/sblinux/mine/workspace/neT/src/neTRunAction.cc:42
    #22 0x00007ffff7928396 in G4RunManager::RunInitialization() () from /home/sblinux/cernsoftware/geant4/10.7.p01-gcc7/release/lib/libG4run.so
    #23 0x00007ffff79206d6 in G4RunManager::BeamOn(int, char const*, int) () from /home/sblinux/cernsoftware/geant4/10.7.p01-gcc7/release/lib/libG4run.so
    #24 0x0000555555568ad0 in main (argc=2, argv=0x7fffffffcd18) at /home/sblinux/mine/workspace/neT/neutrinoT.cc:123
    Continuing.










X4PhysicalVolume::convertSurfaces
-----------------------------------

::

     643 void X4PhysicalVolume::convertSurfaces()
     644 {
     645     LOG(LEVEL) << "[" ;
     646 
     647     size_t num_surf0, num_surf1 ;
     648     num_surf0 = m_slib->getNumSurfaces() ;
     649     assert( num_surf0 == 0 );
     650 
     651     char mode_g4interpolate = 'G' ;
     652     //char mode_oldstandardize = 'S' ; 
     653     //char mode_asis = 'A' ; 
     654     char mode = mode_g4interpolate ;
     655 
     656     X4LogicalBorderSurfaceTable::Convert(m_slib, mode);
     657     num_surf1 = m_slib->getNumSurfaces() ;
     658 
     659     size_t num_lbs = num_surf1 - num_surf0 ; num_surf0 = num_surf1 ;
     660 
     661     X4LogicalSkinSurfaceTable::Convert(m_slib, mode);
     662     num_surf1 = m_slib->getNumSurfaces() ;
     663 
     664     size_t num_sks = num_surf1 - num_surf0 ; num_surf0 = num_surf1 ;
     665 
     666     const G4VPhysicalVolume* pv = m_top ;
     667     int depth = 0 ;
     668     convertImplicitSurfaces_r(pv, depth);
     669     num_surf1 = m_slib->getNumSurfaces() ;
     670 
     671     size_t num_ibs = num_surf1 - num_surf0 ; num_surf0 = num_surf1 ;
     672 
     673 
     674     m_slib->dumpImplicitBorderSurfaces("X4PhysicalVolume::convertSurfaces");
     675 
     676     m_slib->addPerfectSurfaces();
     677     m_slib->dumpSurfaces("X4PhysicalVolume::convertSurfaces");
     678 
     679     m_slib->collectSensorIndices();
     680     m_slib->dumpSensorIndices("X4PhysicalVolume::convertSurfaces");
     681 
     682     LOG(LEVEL)
     683         << "]"
     684         << " num_lbs " << num_lbs
     685         << " num_sks " << num_sks
     686         << " num_ibs " << num_ibs
     687         ;
     688 
     689 }



cause of slow init 
--------------------

Initialization is slow in 1070+ because X4LogicalBorderSurfaceTable::PrepareVector is being run 
for every surface when it only needs to be run once for the entire geometry.

::

    293 /**
    294 size_t X4::GetOpticksIndex( const G4LogicalSurface* const surf )
    295 ==================================================================
    296 
    297 Border and skin surfaces are listed separately by G4 but together by Opticks
    298 so need to define the following convention for surface indices: 
    299 
    300 * border surfaces follow the Geant4 order with matched indices
    301 * skin surfaces follow Geant4 order but with indices offset by the number of border surfaces 
    302 
    303 * NB for these indices to remain valid, clearly must not add/remove 
    304   surfaces after accessing the indices 
    305 
    306  
    307 **/
    308 
    309 size_t X4::GetOpticksIndex( const G4LogicalSurface* const surf )
    310 {
    311     size_t num_lbs = G4LogicalBorderSurface::GetNumberOfBorderSurfaces() ;
    312     size_t num_sks = G4LogicalSkinSurface::GetNumberOfSkinSurfaces() ;
    313 
    314     const G4LogicalBorderSurfaceTable* lbs_table = G4LogicalBorderSurface::GetSurfaceTable() ;
    315     const std::vector<G4LogicalBorderSurface*>* lbs_vec = X4LogicalBorderSurfaceTable::PrepareVector(lbs_table);
    316 
    317     const G4LogicalSkinSurfaceTable*   sks_vec = G4LogicalSkinSurface::GetSurfaceTable() ;
    318 
    319     assert( num_lbs == lbs_vec->size() );
    320     assert( num_sks == sks_vec->size() );
    321 
    322     const G4LogicalBorderSurface* const lbs = dynamic_cast<const G4LogicalBorderSurface* const>(surf);
    323     const G4LogicalSkinSurface*   const sks = dynamic_cast<const G4LogicalSkinSurface* const>(surf);
    324 
    325     assert( (lbs == NULL) ^ (sks == NULL) );   // one or other must be NULL, but not both   
    326 
    327     int idx_lbs = lbs ? GetItemIndex<G4LogicalBorderSurface>( lbs_vec  , lbs ) : -1 ;
    328     int idx_sks = sks ? GetItemIndex<G4LogicalSkinSurface>(   sks_vec  , sks ) : -1 ;
    329 
    330     assert( (idx_lbs == -1) ^ (idx_sks == -1) ); // one or other must be -1, but not both 
    331 
    332     return idx_lbs > -1 ? idx_lbs : idx_sks + num_lbs ;
    333 }



Check Opticks with 1071 using "Francis" account which is dedicated to this
------------------------------------------------------------------------------


* /Users/francis/opticks is symbolic link to /Users/blyth/opticks
* /Users/francis/local/opticks_externals is symbolic link to /usr/local/opticks_externals which includes several Geant4 builds 
* BUT: /Users/francis/local/opticks is distinct allowing a separate build of Opticks against different externals 

::

    epsilon:issues blyth$ ssh F
    Last login: Tue Jan 19 16:44:44 2021 from 127.0.0.1
    epsilon:~ francis$ l
    total 8
    -rw-r--r--   1 francis  staff    73 Jan 17 16:46 SOKConfTest.log
    drwxr-xr-x   5 francis  staff   160 Jan 16 20:08 local
    lrwxr-xr-x   1 francis  staff    20 Jan 16 19:04 opticks -> /Users/blyth/opticks
    epsilon:~ francis$ 


    epsilon:local francis$ l
    total 0
    drwxr-xr-x  12 francis  staff  384 Jan 16 23:04 opticks
    lrwxr-xr-x   1 francis  staff   28 Jan 16 20:05 opticks_externals -> /usr/local/opticks_externals
    epsilon:local francis$ 



update francis Opticks build against the non-standard externals configured via CMAKE_PREFIX_PATH envvar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


/Users/francis/.opticks_config::

     23 ## hookup paths to access "foreign" externals 
     24 opticks-prepend-prefix /usr/local/opticks_externals/clhep_2440
     25 opticks-prepend-prefix /usr/local/opticks_externals/xercesc
     26 opticks-prepend-prefix /usr/local/opticks_externals/g4_1070
     27 opticks-prepend-prefix /usr/local/opticks_externals/boost
     28 
     29 export OPTICKS_GEANT4_VER=1070



::

   epsilon:opticks francis$ oo




