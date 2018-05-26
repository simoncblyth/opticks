UseUseG4DAE
=============

Locate gdml file and convert it into G4DAE COLLADA::

    epsilon:UseUseG4DAE blyth$ g4-find-gdml | grep test.gdml
    /usr/local/opticks/externals/g4/geant4_10_02_p01/examples/extended/persistency/gdml/G02/test.gdml
         ## find some gdml

    rm -rf /tmp/test.dae   
         ## delete any preexisting output

    UseUseG4DAE /usr/local/opticks/externals/g4/geant4_10_02_p01/examples/extended/persistency/gdml/G02/test.gdml /tmp/test.dae 
         ## run the conversion 

    open /tmp/test.dae
         ## (macOS) visualize COLLADA geometry 

