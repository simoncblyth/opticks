lxe-src(){      echo optix/lxe/lxe.bash ; }
lxe-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(lxe-src)} ; }
lxe-vi(){       vi $(lxe-source) ; }
lxe-usage(){ cat << EOU

Bringing the Geant4 LXe example into the 
style of env- packages

EOU
}

lxe-env(){  
   elocal- 
   g4-
}


lxe-name(){ echo lxetest ; }
lxe-bin(){ echo ${CFG4_BINARY:-$(lxe-idir)/bin/$(lxe-name)} ; }

lxe-idir(){ echo $(local-base)/env/optix/lxe; } 
lxe-bdir(){ echo $(local-base)/env/optix/lxe.build ; }
lxe-sdir(){ echo $(opticks-home)/optix/lxe ; }

lxe-icd(){  cd $(lxe-idir); }
lxe-bcd(){  cd $(lxe-bdir); }
lxe-scd(){  cd $(lxe-sdir); }

lxe-dir(){  echo $(lxe-sdir) ; }
lxe-cd(){   cd $(lxe-dir); }



lxe-wipe(){
    local bdir=$(lxe-bdir)
    rm -rf $bdir
}

lxe-cmake(){
   local iwd=$PWD
   local bdir=$(lxe-bdir)
   mkdir -p $bdir
   lxe-bcd

  # -DWITH_GEANT4_UIVIS=OFF \

   cmake \
         -DGeant4_DIR=$(g4-cmake-dir) \
         -DCMAKE_INSTALL_PREFIX=$(lxe-idir) \
         -DCMAKE_BUILD_TYPE=Debug  \
         $(lxe-sdir)
   cd $iwd 
}

lxe-make(){
    local iwd=$PWD
    lxe-bcd
    make $*
    cd $iwd 
}

lxe-install(){
   lxe-make install
}

lxe--(){
   lxe-wipe
   lxe-cmake
   lxe-make
   lxe-install
}

lxe-export()
{
   g4-export
}

lxe-run(){
   local bin=$(lxe-bin)
   lxe-export
   $bin $*
}



