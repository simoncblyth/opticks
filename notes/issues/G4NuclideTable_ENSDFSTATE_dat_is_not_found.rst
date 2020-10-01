G4NuclideTable_ENSDFSTATE_dat_is_not_found
============================================

FIXED by moving aside the stale::

    [blyth@localhost opticks]$ ls -l /home/blyth/local/opticks/externals/config/
    total 8
    -rw-rw-r--. 1 blyth blyth 967 Sep 28  2019 geant4.bash
    -rw-rw-r--. 1 blyth blyth 797 May  7 22:09 geant4.ini
    [blyth@localhost opticks]$ mv /home/blyth/local/opticks/externals/config/geant4.ini /home/blyth/local/opticks/externals/config/geant4.ini.stale
    [blyth@localhost opticks]$ 



CTestDetectorTest not failing on laptop (Darwin), but ERROR from OpticksResource::SetupG4Environment::

    epsilon:opticks blyth$ CTestDetectorTest
    2020-10-01 15:39:04.491 INFO  [741301] [main@44] CTestDetectorTest
    2020-10-01 15:39:04.492 INFO  [741301] [BOpticksKey::SetKey@75]  spec OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3
    2020-10-01 15:39:04.493 INFO  [741301] [Opticks::init@405] INTEROP_MODE hostname epsilon.local
    2020-10-01 15:39:04.493 INFO  [741301] [Opticks::init@414]  non-legacy mode : ie mandatory keyed access to geometry, opticksaux 
    2020-10-01 15:39:04.493 ERROR [741301] [OpticksResource::SetupG4Environment@519] inipath /usr/local/opticks/externals/config/geant4.ini
    2020-10-01 15:39:04.493 ERROR [741301] [OpticksResource::SetupG4Environment@528]  MISSING inipath /usr/local/opticks/externals/config/geant4.ini (create it with bash functions: g4-;g4-export-ini ) 


Linux CTestDetectorTest is failing with familiar Geant4 error from G4NuclideTable::


    [blyth@localhost opticks]$ CTestDetectorTest
    2020-10-01 22:41:24.914 INFO  [405813] [main@44] CTestDetectorTest
    2020-10-01 22:41:24.914 INFO  [405813] [BOpticksKey::SetKey@75]  spec OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.5aa828335373870398bf4f738781da6c
    2020-10-01 22:41:24.916 INFO  [405813] [Opticks::init@405] COMPUTE_MODE forced_compute  hostname localhost.localdomain
    2020-10-01 22:41:24.916 INFO  [405813] [Opticks::init@414]  non-legacy mode : ie mandatory keyed access to geometry, opticksaux 
    2020-10-01 22:41:24.916 ERROR [405813] [OpticksResource::SetupG4Environment@519] inipath /home/blyth/local/opticks/externals/config/geant4.ini
    2020-10-01 22:41:24.922 INFO  [405813] [BOpticksResource::setupViaKey@832] 

    2020-10-01 22:41:29.970 ERROR [405813] [OpticksResource::SetupG4Environment@519] inipath /home/blyth/local/opticks/externals/config/geant4.ini

    -------- EEEE ------- G4Exception-START -------- EEEE -------
    *** G4Exception : PART70001
          issued by : G4NuclideTable
    ENSDFSTATE.dat is not found.
    *** Fatal Exception *** core dump ***
    Segmentation fault (core dumped)
    [blyth@localhost opticks]$ l /home/blyth/local/opticks/externals/config/geant4.ini
    -rw-rw-r--. 1 blyth blyth 797 May  7 22:09 /home/blyth/local/opticks/externals/config/geant4.ini
    [blyth@localhost opticks]$ echo $OPTICKS_PREFIX
    /home/blyth/local/opticks



Darwin succeeds with traditional env based Geant4 config::

    epsilon:opticks blyth$ env | grep G4
    G4ENSDFSTATEDATA=/usr/local/opticks_externals/g4/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2
    G4PIIDATA=/usr/local/opticks_externals/g4/share/Geant4-10.4.2/data/G4PII1.3
    G4NEUTRONXSDATA=/usr/local/opticks_externals/g4/share/Geant4-10.4.2/data/G4NEUTRONXS1.4
    G4LEDATA=/usr/local/opticks_externals/g4/share/Geant4-10.4.2/data/G4EMLOW7.3
    G4NEUTRONHPDATA=/usr/local/opticks_externals/g4/share/Geant4-10.4.2/data/G4NDL4.5
    G4SAIDXSDATA=/usr/local/opticks_externals/g4/share/Geant4-10.4.2/data/G4SAIDDATA1.1
    G4REALSURFACEDATA=/usr/local/opticks_externals/g4/share/Geant4-10.4.2/data/RealSurface2.1.1
    G4ABLADATA=/usr/local/opticks_externals/g4/share/Geant4-10.4.2/data/G4ABLA3.1
    G4LEVELGAMMADATA=/usr/local/opticks_externals/g4/share/Geant4-10.4.2/data/PhotonEvaporation5.2
    G4RADIOACTIVEDATA=/usr/local/opticks_externals/g4/share/Geant4-10.4.2/data/RadioactiveDecay5.2
    epsilon:opticks blyth$ 


Linux fails due to a stale /home/blyth/local/opticks/externals/config/geant4.ini::


    [blyth@localhost opticks]$ cat /home/blyth/local/opticks/externals/config/geant4.ini
    G4NEUTRONXSDATA=$OPTICKS_PREFIX/externals/share/Geant4-10.4.2/data/G4NEUTRONXS1.4
    G4LEDATA=$OPTICKS_PREFIX/externals/share/Geant4-10.4.2/data/G4EMLOW7.3
    G4REALSURFACEDATA=$OPTICKS_PREFIX/externals/share/Geant4-10.4.2/data/RealSurface2.1.1
    G4PIIDATA=$OPTICKS_PREFIX/externals/share/Geant4-10.4.2/data/G4PII1.3
    G4ENSDFSTATEDATA=$OPTICKS_PREFIX/externals/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2
    G4LEVELGAMMADATA=$OPTICKS_PREFIX/externals/share/Geant4-10.4.2/data/PhotonEvaporation5.2
    G4NEUTRONHPDATA=$OPTICKS_PREFIX/externals/share/Geant4-10.4.2/data/G4NDL4.5
    G4ABLADATA=$OPTICKS_PREFIX/externals/share/Geant4-10.4.2/data/G4ABLA3.1
    G4SAIDXSDATA=$OPTICKS_PREFIX/externals/share/Geant4-10.4.2/data/G4SAIDDATA1.1
    G4RADIOACTIVEDATA=$OPTICKS_PREFIX/externals/share/Geant4-10.4.2/data/RadioactiveDecay5.2
    [blyth@localhost opticks]$ 

    [blyth@localhost opticks]$ ls -l /home/blyth/local/opticks/externals/config/
    total 8
    -rw-rw-r--. 1 blyth blyth 967 Sep 28  2019 geant4.bash
    -rw-rw-r--. 1 blyth blyth 797 May  7 22:09 geant4.ini
    [blyth@localhost opticks]$ mv /home/blyth/local/opticks/externals/config/geant4.ini /home/blyth/local/opticks/externals/config/geant4.ini.stale
    [blyth@localhost opticks]$ 



