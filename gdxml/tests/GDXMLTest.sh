#!/bin/bash -l 

logging(){
   export GDXML=INFO
   export GDXMLRead=INFO
   export GDXMLWrite=INFO
}
#logging 
  

default=/tmp/$USER/opticks/ntds3/G4CXOpticks/origin.gdml
GDMLPATH=${GDMLPATH:-$default}

if [ ! -f "$GDMLPATH" ]; then 

   echo $BASH_SOURCE GDMLPATH $GDMLPATH does not exit 
   exit 1
fi 

GDXMLTest $GDMLPATH
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 

exit 0 
