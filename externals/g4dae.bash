g4dae-source(){   echo $BASH_SOURCE ; }
g4dae-vi(){       vi $(g4dae-source) ; }
g4dae-usage(){ cat << EOU

G4DAE : G4 COLLADA/DAE Geometry Export as an Opticks external 
=====================================================================

* see also ~/g4dae/g4d-

EOU
}

g4dae-env(){     
    olocal- 
    g4- 
}

g4dae-prefix(){  echo $(opticks-prefix)/externals ; }
g4dae-base(){    echo $(g4dae-prefix)/g4dae ;  }

g4dae-dir(){     echo $(g4dae-base)/g4dae ; }
g4dae-sdir(){    echo $(g4dae-base)/g4dae ; }
g4dae-bdir(){    echo $(g4dae-base)/g4dae.build ; }

g4dae-cd(){   cd $(g4dae-dir); }
g4dae-scd(){  cd $(g4dae-sdir); }
g4dae-bcd(){  cd $(g4dae-bdir); }

g4dae-info(){ cat << EOI

   g4dae-source  : $(g4dae-source)
   g4dae-prefix  : $(g4dae-prefix)
   g4dae-base    : $(g4dae-base)
   g4dae-dir     : $(g4dae-dir)
   g4dae-bdir    : $(g4dae-bdir)

   g4dae-url     : $(g4dae-url)

   g4-cmake-dir  : $(g4-cmake-dir)


EOI
}

g4dae-wipe(){
   local bdir=$(g4dae-bdir)
   rm -rf $bdir
}

g4dae-url(){
   case $USER in
     blyth) echo ssh://hg@bitbucket.org/simoncblyth/g4dae ;;
         *) echo https://bitbucket.org/simoncblyth/g4dae ;;
   esac
} 

g4dae-get(){
   local iwd=$PWD
   local dir=$(dirname $(g4dae-dir)) &&  mkdir -p $dir && cd $dir
   if [ ! -d "g4dae" ]; then 
       hg clone $(g4dae-url)
   fi 
   cd $iwd
}

g4dae-cmake(){
   local iwd=$PWD
   local bdir=$(g4dae-bdir)
   mkdir -p $bdir
   g4dae-bcd 
   cmake \
       -G "$(opticks-cmake-generator)" \
       -DCMAKE_INSTALL_PREFIX=$(g4dae-prefix) \
       -DGeant4_DIR=$(g4-cmake-dir) \
       -DCMAKE_BUILD_TYPE=Debug \
       $(g4dae-sdir)

   cd $iwd
}

g4dae-make(){
   local iwd=$PWD
   g4dae-bcd 
   make $*
   cd $iwd
}

g4dae-install(){
   g4dae-make install
}

g4dae--()
{
    g4dae-get
    g4dae-cmake
    g4dae-make
    g4dae-install
}

g4dae-cls(){  
   local iwd=$PWD
   g4dae-cd 
   g4-
   g4-cls- . ${1:-G4DAEParser} ; 
   cd $iwd
}


