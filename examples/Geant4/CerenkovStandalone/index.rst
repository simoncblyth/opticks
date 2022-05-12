CerenkovStandalone Index
===========================

scripts
---------


G4Cerenkov_modifiedTest.sh
    uses cks.bash to build and run executable G4Cerenkov_modifiedTest 
    and analyse results using G4Cerenkov_modifiedTest.py 
    Does BetaInverse scanning to compare the standard g4 G4Cerenkov::GetAverageNumberOfPhotons
    with an _s2 variation of how the numerical integral is done. 

L4CerenkovTest.sh
    builds and runs L4CerenkovTest executable which instruments Cerenkov generation 
    with values saved into NP arrays which are analysed and presented as plots with 
    L4CerenkovTest.py 

scan.sh
    invokes ./L4CerenkovTest.sh with 11 BetaInverse value arguments from 1.0 to 2.0 

OpticksRandomTest.sh
    builds and runs OpticksRandomTest executable 
    checking the OpticksRandom hijacking of the G4UniformRand 

OpticksUtilTest.sh
    builds and runs OpticksUtilTest executable checking array concatenation 

RINDEXTest.sh
    builds and runs RINDEXTest executable checking the OpticksUtil (TODO:rename to U4Util and relocate into U4) 
    property handling 
    

copy.sh
    output png copying 


