clhep-source(){   echo $BASH_SOURCE ; }
clhep-vi(){       vi $BASH_SOURCE ; }
clhep-env(){      olocal- ; }
clhep-usage(){ cat << EOU
CLHEP
=====

Note that the Geant4 distribution (10.4.2) 
includes an ancient builtin CLHEP libG4clhep of unspecified version.  

It is unhealthy to have both this CLHEP and the Geant4 inbuilt one
in your environment simultaneously. 

To prevent building that ancient CLHEP use -DGEANT4_USE_SYSTEM_CLHEP=ON 
when building Geant4.


https://proj-clhep.web.cern.ch/proj-clhep/

The latest releases are::

   2.4.4.0, released November 9, 2020.
   2.3.4.6, released February 15, 2018.

Geant4 1070 requires 2.4.4.0

EOU
}

clhep-prefix-default(){  echo $(opticks-prefix)_externals/clhep_$(clhep-version)  ; }
clhep-prefix(){  echo ${OPTICKS_CLHEP_PREFIX:-$(clhep-prefix-default)}  ; }
#clhep-ver(){     echo 2.4.1.0 ; }
clhep-ver(){     echo 2.4.4.0 ; }
clhep-version(){  local v=$(clhep-ver) ; echo ${v//./} ; }

#clhep-url(){     echo http://proj-clhep.web.cern.ch/proj-clhep/DISTRIBUTION/tarFiles/clhep-$(clhep-ver).tgz ; }
clhep-url(){     echo https://proj-clhep.web.cern.ch/proj-clhep/dist1/clhep-$(clhep-ver).tgz ; }

clhep-dstname(){ echo $(clhep-ver) ; }     
clhep-dir(){     echo $(clhep-prefix).build/$(clhep-dstname)/CLHEP ; }  
clhep-bdir(){    echo $(clhep-dir).build ; }

clhep-info(){ cat << EOI


   clhep-ver     : $(clhep-ver)
   clhep-url     : $(clhep-url)
   clhep-dstname : $(clhep-dstname)    name of the directory created by exploding the distribution
   clhep-prefix  : $(clhep-prefix)
   clhep-dir     : $(clhep-dir)        exploded distribution dir 
   clhep-bdir    : $(clhep-bdir)

EOI
}
clhep-cd(){  cd $(clhep-dir); }

clhep-get(){

   local dir=$(dirname $(dirname $(clhep-dir))) &&  mkdir -p $dir && cd $dir
   local url=$(clhep-url)
   local dst=$(basename $url)
   local nam=$(clhep-dstname)

   [ ! -f "$dst" ] && echo getting $url && curl -L -O $url

   if [ "${dst/.zip}" != "${dst}" ]; then
        [ ! -d "$nam" ] && unzip $dst
   fi
   if [ "${dst/.tar.gz}" != "${dst}" -o  "${dst/.tgz}" != "${dst}" ]; then
        [ ! -d "$nam" ] && tar zxvf $dst
   fi

   [ -d "$nam" ]
}

clhep-bcd(){
   local bdir=$(clhep-bdir)
   mkdir -p $bdir && cd $bdir 
}

clhep-configure(){
   clhep-bcd
   local sdir=$(clhep-dir)
   cmake -DCMAKE_INSTALL_PREFIX=$(clhep-prefix) $sdir   
}

clhep-build()
{
   clhep-bcd
   make
}

clhep-install()
{
   clhep-bcd
   make install
}

clhep--()
{
    local msg="=== $FUNCNAME :"
    clhep-get
    [ $? -ne 0 ] && echo $msg get FAIL && return 1
    clhep-configure 
    [ $? -ne 0 ] && echo $msg configure FAIL && return 2
    clhep-build
    [ $? -ne 0 ] && echo $msg build FAIL && return 3
    clhep-install
    [ $? -ne 0 ] && echo $msg install FAIL && return 3
    return 0 
}

clhep-pc()
{
    local msg="=== $FUNCNAME :"
    echo $msg nothing to do : assuming g4-pc has got it covered 
}


