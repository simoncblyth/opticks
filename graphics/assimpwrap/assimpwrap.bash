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







FUNCTIONS
----------





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

assimpwrap-geokey(){
    case $1 in
      extra) echo DAE_NAME_DYB ;; 
          *) echo DAE_NAME_DYB_NOEXTRA  ;;
    esac  
}

assimpwrap-material(){
    echo __dd__Geometry__AdDetails__AdSurfacesAll__ESRAirSurfaceBot 
    #echo __dd__Materials__GdDopedLS0xc2a8ed0 
}

assimpwrap-ggctrl(){
    echo __dd__
}


assimpwrap-export(){
    export ASSIMPWRAP_GEOKEY="$(assimpwrap-geokey $1)"
    export ASSIMPWRAP_QUERY="index:1,depth:2" 
    export ASSIMPWRAP_MATERIAL="$(assimpwrap-material)" 
    export ASSIMPWRAP_GGCTRL="$(assimpwrap-ggctrl)" 
    export-
    export-export
    env | grep ASSIMPWRAP
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
}

assimpwrap-lldb(){
    DEBUG=lldb assimpwrap-run
}

assimpwrap-brun(){
   echo running from bdir not idir : no install needed, but much set library path
   local bdir=$(assimpwrap-bdir)
   DYLD_LIBRARY_PATH=$bdir $DEBUG $bdir/AssimpWrapTest 
}



assimpwrap-test(){
    local arg=$1
    assimpwrap-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

    assimpwrap-export $arg
    DEBUG=lldb assimpwrap-brun
}


assimpwrap-otool(){
   otool -L $(assimpwrap-bin)
}
