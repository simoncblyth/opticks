# === func-gen- : graphics/assimpwrap/assimpwrap fgp graphics/assimpwrap/assimpwrap.bash fgn assimpwrap fgh graphics/assimpwrap
assimpwrap-src(){      echo graphics/assimpwrap/assimpwrap.bash ; }
assimpwrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(assimpwrap-src)} ; }
assimpwrap-vi(){       vi $(assimpwrap-source) ; }
assimpwrap-env(){      elocal- ; }
assimpwrap-usage(){ cat << EOU

AssimpWrap
============

Wrapping the Assimp 3D Asset Importer Library

* Used by raytrace-



EOU
}
assimpwrap-idir(){ echo $(local-base)/env/graphics ; }  # prefix
assimpwrap-bdir(){ echo $(local-base)/env/graphics/assimpwrap.build ; }
assimpwrap-sdir(){ echo $(env-home)/graphics/assimpwrap ; }

assimpwrap-icd(){  cd $(assimpwrap-idir); }
assimpwrap-bcd(){  cd $(assimpwrap-bdir); }
assimpwrap-scd(){  cd $(assimpwrap-sdir); }

assimpwrap-cd(){  cd $(assimpwrap-sdir); }

assimpwrap-mate(){ mate $(assimpwrap-dir) ; }

assimpwrap-wipe(){
    local bdir=$(assimpwrap-bdir)
    rm -rf $bdir
}


assimpwrap-cmake(){
   local bdir=$(assimpwrap-bdir)
   mkdir -p $bdir
   assimpwrap-bcd
   cmake $(assimpwrap-sdir) -DCMAKE_INSTALL_PREFIX=$(assimpwrap-idir) -DCMAKE_BUILD_TYPE=Debug 
}

assimpwrap-make(){
    local iwd=$PWD
    assimpwrap-bcd
    make $*
    cd $iwd
}

assimpwrap-install(){
   assimpwrap-make install
}


assimpwrap-bbin(){ echo $(assimpwrap-bdir)/AssimpWrapTest ; }
assimpwrap-bin(){ echo $(assimpwrap-idir)/bin/AssimpWrapTest ; }

assimpwrap-export(){
    export ASSIMPWRAP_QUERY="index:1,depth:2" 
    export ASSIMPWRAP_GEOKEY="DAE_NAME_DYB_NOEXTRA"
    export-
    export-export
}

assimpwrap-run(){
    assimpwrap-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

    assimpwrap-export 
    $DEBUG $(assimpwrap-bin) $*  
}

assimpwrap--(){

    assimpwrap-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

    assimpwrap-install $*
 
    #assimpwrap-run $* 
}

assimpwrap-lldb(){
    DEBUG=lldb assimpwrap-run
}

assimpwrap-brun(){
   local bdir=$(assimpwrap-bdir)
   DYLD_LIBRARY_PATH=$bdir $bdir/AssimpWrapTest 
}


assimpwrap-otool(){
   otool -L $(assimpwrap-bin)
}
