ckm-okg4-material-rindex-mismatch
======================================



Issue : material energy range persisted in the genstep mismatches that read from the G4 material, tripping an assert
------------------------------------------------------------------------------------------------------------------------

* this is due to an inconsistent application of standardization in the two executables 

  * TODO: review the material info flow in both cases, to decide where to standardize 

::

    199 
    200     G4MaterialPropertyVector* Rindex = GetRINDEX(materialIndex) ;  // NB straight G4, no range standardization
    201 
    202     G4double Pmin2 = Rindex->GetMinLowEdgeEnergy();
    203     G4double Pmax2 = Rindex->GetMaxLowEdgeEnergy();
    204     G4double dp2 = Pmax2 - Pmin2;
    205 
    206     G4double epsilon = 1e-6 ;
    207     bool Pmin_match = std::abs( Pmin2 - Pmin ) < epsilon ;
    208     bool Pmax_match = std::abs( Pmax2 - Pmax ) < epsilon ;
    209 
    210     if(!Pmin_match || !Pmax_match)
    211         LOG(fatal)
    212             << " Pmin " << Pmin
    213             << " Pmin2 (MinLowEdgeEnergy) " << Pmin2
    214             << " dif " << std::abs( Pmin2 - Pmin )
    215             << " epsilon " << epsilon
    216             << " Pmin(nm) " << h_Planck*c_light/Pmin/nm
    217             << " Pmin2(nm) " << h_Planck*c_light/Pmin2/nm
    218             ;
    219 
    220     if(!Pmax_match || !Pmin_match)
    221         LOG(fatal)
    222             << " Pmax " << Pmax
    223             << " Pmax2 (MaxLowEdgeEnergy) " << Pmax2
    224             << " dif " << std::abs( Pmax2 - Pmax )
    225             << " epsilon " << epsilon
    226             << " Pmax(nm) " << h_Planck*c_light/Pmax/nm
    227             << " Pmax2(nm) " << h_Planck*c_light/Pmax2/nm



ckm-okg4
-----------

::

    ckm-okg4 () 
    { 
        OPTICKS_KEY=$(ckm-key) $(ckm-dbg) OKG4Test --compute --envkey --embedded --save --natural
    }



::

    [blyth@localhost issues]$ DEBUG=1 ckm-okg4

    (gdb) r
    Starting program: /home/blyth/local/opticks/lib/OKG4Test --compute --envkey --embedded --save --natural


    2019-05-29 22:31:56.672 INFO  [195702] [CCerenkovGenerator::GeneratePhotonsFromGenstep@135]  genstep_idx 0 num_gs 1 materialLine 7 materialIndex 1      post  0.000   0.000   0.000   0.000 

    2019-05-29 22:31:56.672 INFO  [195702] [CCerenkovGenerator::GeneratePhotonsFromGenstep@168]  From Genstep :  Pmin 1.512e-06 Pmax 2.0664e-05 wavelength_min(nm) 60 wavelength_max(nm) 820 preVelocity 276.074 postVelocity 273.253
    2019-05-29 22:31:56.672 ERROR [195702] [CCerenkovGenerator::GetRINDEX@73]  aMaterial 0x9d5310 aMaterial.Name Water materialIndex 1 num_material 3 Rindex 0x9d6930 Rindex2 0x9d6930
    2019-05-29 22:31:56.672 FATAL [195702] [CCerenkovGenerator::GeneratePhotonsFromGenstep@211]  Pmin 1.512e-06 Pmin2 (MinLowEdgeEnergy) 2.034e-06 dif 5.21998e-07 epsilon 1e-06 Pmin(nm) 820 Pmin2(nm) 609.558
    2019-05-29 22:31:56.672 FATAL [195702] [CCerenkovGenerator::GeneratePhotonsFromGenstep@221]  Pmax 2.0664e-05 Pmax2 (MaxLowEdgeEnergy) 4.136e-06 dif 1.6528e-05 epsilon 1e-06 Pmax(nm) 60 Pmax2(nm) 299.768
    OKG4Test: /home/blyth/opticks/cfg4/CCerenkovGenerator.cc:234: static G4VParticleChange* CCerenkovGenerator::GeneratePhotonsFromGenstep(const OpticksGenstep*, unsigned int): Assertion `Pmax_match && "material mismatches genstep source material"' failed.

    Program received signal SIGABRT, Aborted.
    0x00007fffe2038207 in raise () from /lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 expat-2.1.0-10.el7_3.x86_64 glfw-3.2.1-2.el7.x86_64 glibc-2.17-260.el7_6.3.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-37.el7_6.x86_64 libX11-1.6.5-2.el7.x86_64 libXau-1.0.8-2.1.el7.x86_64 libXcursor-1.1.15-1.el7.x86_64 libXext-1.3.3-3.el7.x86_64 libXfixes-5.0.3-1.el7.x86_64 libXinerama-1.1.3-2.1.el7.x86_64 libXrandr-1.5.1-2.el7.x86_64 libXrender-0.9.10-1.el7.x86_64 libXxf86vm-1.1.4-1.el7.x86_64 libcom_err-1.42.9-13.el7.x86_64 libgcc-4.8.5-36.el7_6.1.x86_64 libglvnd-1.0.1-0.8.git5baa1e5.el7.x86_64 libglvnd-glx-1.0.1-0.8.git5baa1e5.el7.x86_64 libicu-50.1.2-17.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-36.el7_6.1.x86_64 libxcb-1.13-1.el7.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 xerces-c-3.1.1-9.el7.x86_64 zlib-1.2.7-18.el7.x86_64
    (gdb) bt
    #0  0x00007fffe2038207 in raise () from /lib64/libc.so.6
    #1  0x00007fffe20398f8 in abort () from /lib64/libc.so.6
    #2  0x00007fffe2031026 in __assert_fail_base () from /lib64/libc.so.6
    #3  0x00007fffe20310d2 in __assert_fail () from /lib64/libc.so.6
    #4  0x00007fffefd6d182 in CCerenkovGenerator::GeneratePhotonsFromGenstep (gs=0x8f2470, idx=0) at /home/blyth/opticks/cfg4/CCerenkovGenerator.cc:234
    #5  0x00007fffefdf2000 in CGenstepSource::generatePhotonsFromOneGenstep (this=0x933c80) at /home/blyth/opticks/cfg4/CGenstepSource.cc:94
    #6  0x00007fffefdf1f19 in CGenstepSource::GeneratePrimaryVertex (this=0x933c80, event=0x21636b0) at /home/blyth/opticks/cfg4/CGenstepSource.cc:70
    #7  0x00007fffefdc5940 in CPrimaryGeneratorAction::GeneratePrimaries (this=0x8f35f0, event=0x21636b0) at /home/blyth/opticks/cfg4/CPrimaryGeneratorAction.cc:15
    #8  0x00007fffec6b5ba7 in G4RunManager::GenerateEvent (this=0x706cd0, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:460
    #9  0x00007fffec6b563c in G4RunManager::ProcessOneEvent (this=0x706cd0, i_event=0) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:398
    #10 0x00007fffec6b54d7 in G4RunManager::DoEventLoop (this=0x706cd0, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:367
    #11 0x00007fffec6b4d2d in G4RunManager::BeamOn (this=0x706cd0, n_event=1, macroFile=0x0, n_select=-1) at /home/blyth/local/opticks/externals/g4/geant4.10.04.p02/source/run/src/G4RunManager.cc:273
    #12 0x00007fffefdeec4f in CG4::propagate (this=0x7003b0) at /home/blyth/opticks/cfg4/CG4.cc:331
    #13 0x00007ffff7bd570f in OKG4Mgr::propagate_ (this=0x7fffffffd760) at /home/blyth/opticks/okg4/OKG4Mgr.cc:144
    #14 0x00007ffff7bd55cf in OKG4Mgr::propagate (this=0x7fffffffd760) at /home/blyth/opticks/okg4/OKG4Mgr.cc:84
    #15 0x00000000004039a7 in main (argc=6, argv=0x7fffffffda98) at /home/blyth/opticks/okg4/tests/OKG4Test.cc:9
    (gdb) 

    (gdb) f 5
    #5  0x00007fffefdf2000 in CGenstepSource::generatePhotonsFromOneGenstep (this=0x933c80) at /home/blyth/opticks/cfg4/CGenstepSource.cc:94
    94          case CERENKOV:      pc = CCerenkovGenerator::GeneratePhotonsFromGenstep(m_gs,m_idx) ; break ; 
    (gdb) l
    89      unsigned gencode = m_gs->getGencode(m_idx) ; 
    90      G4VParticleChange* pc = NULL ; 
    91  
    92      switch( gencode )
    93      { 
    94          case CERENKOV:      pc = CCerenkovGenerator::GeneratePhotonsFromGenstep(m_gs,m_idx) ; break ; 
    95          case SCINTILLATION: pc = NULL                                                       ; break ;  
    96          default:            pc = NULL ; 
    97      }
    98  
    (gdb) 

    (gdb) l
    229 
    230     bool with_key = Opticks::HasKey() ; 
    231     if(with_key)
    232     {
    233         assert( Pmin_match && "material mismatches genstep source material" ); 
    234         assert( Pmax_match && "material mismatches genstep source material" ); 
    235     }
    236     else
    237     {
    238         LOG(warning) << "permissive generation for legacy gensteps " ;
    (gdb) 

    (gdb) p Pmin2
    $1 = 2.0339999999999999e-06
    (gdb) p Pmin
    $2 = 1.5120023135750671e-06
    (gdb) p Pmax2
    $3 = 4.1359999999999999e-06
    (gdb) p Pmax
    $4 = 2.0664030671468936e-05
    (gdb) 


