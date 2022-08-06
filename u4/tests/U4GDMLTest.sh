#!/bin/bash -l 

usage(){  cat << EOU 

G4GDML: Reading '/tmp/blyth/opticks/ntds3/G4CXOpticks/origin.gdml'...
G4GDML: Reading definitions...

-------- EEEE ------- G4Exception-START -------- EEEE -------

*** ExceptionHandler is not defined ***
*** G4Exception : InvalidSize
      issued by : G4GDMLEvaluator::DefineMatrix()
Matrix 'PPOABSLENGTH0x56f1750' is not filled correctly!
*** Fatal Exception ***
-------- EEEE -------- G4Exception-END --------- EEEE -------

EOU 

}


logging(){
   export GDXML=INFO
   export GDXMLRead=INFO
   export GDXMLWrite=INFO
}
logging

default=/tmp/$USER/opticks/ntds3/G4CXOpticks/origin.gdml
GDMLPATH=${GDMLPATH:-$default}

if [ ! -f "$GDMLPATH" ]; then 

   echo $BASH_SOURCE GDMLPATH $GDMLPATH does not exit 
   exit 1
fi 

U4GDMLTest $GDMLPATH
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 

exit 0 




