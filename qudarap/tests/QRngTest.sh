#/bin/bash 
usage(){ cat << EOU
QRngTest.sh
==============

~/o/qudarap/tests/QRngTest.sh 

OPTICKS_MAX_PHOTON=M4 ~/o/qudarap/tests/QRngTest.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh 

name=QRngTest
bin=$name
script=$name.py 

export FOLD=$TMP/$name  

#test=ctor
test=generate
#test=ALL

export TEST=${TEST:-$test}

defarg="info_run_ana"
arg=${1:-$defarg}

vars="BASH_SOURCE defarg arg FOLD bin script test TEST"

logging(){
   type $FUNCNAME
   export QRng=INFO
   export QRng__init_VERBOSE=1
   export NP__save_VERBOSE=1
   #export SCurandChunk__ParseName_DEBUG=1
}

[ -n "$LOG" ] && logging 


if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1
   rm $name.log
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   gdb__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
   rm $name.log
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${PYTHON:-python} $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi 

if [ "${arg/pdb}" != "$arg" ]; then 
    ${IPYTHON:-ipython} -i --pdb $script
   [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 4
fi 

exit 0 

