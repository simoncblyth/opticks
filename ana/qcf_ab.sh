#!/bin/bash
usage(){ cat << EOU
qcf_ab.sh
==============

::

    hookup_conda_ok

    source qcf_ab.sh 

    source qcf_ab.sh  pdb 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
which f2py

defarg="info_build_install_pdb"
arg=${1:-$defarg}

name=qcf_ab
dest=$OPTICKS_PREFIX/py/opticks/ana

vv="BASH_SOURCE defarg arg name dest"


if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%20s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
   f2py -c --backend meson -m $name $name.f90
   [ $? -ne 0 ] && echo $BASH_SOURCE build error 
fi 

if [ "${arg/install}" != "$arg" ]; then
   echo [ install the module
   mod=$(ls -1 $name.cpython*.so)
   if [ -d "$dest" ]; then
      cmd="mv $mod $dest/"
      echo $cmd
      eval $cmd
      ls -alst $dest/$name*
   fi
   echo ] install the module
fi 

if [ "${arg/pdb}" != "$arg" ]; then
   echo [ test module
   PYTHONPATH=$OPTICKS_PREFIX/py ipython --pdb -i ${name}_test.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb error 
   echo ] test module
fi




