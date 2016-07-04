# === func-gen- : boost/random/brandom fgp boost/random/brandom.bash fgn brandom fgh boost/random
brandom-src(){      echo boost/random/brandom.bash ; }
brandom-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(brandom-src)} ; }
brandom-vi(){       vi $(brandom-source) ; }
brandom-env(){      olocal- ; }
brandom-usage(){ cat << EOU

* http://www.boost.org/doc/libs/1_58_0/doc/html/boost_random.html



EOU
}

brandom-idir(){ echo $(local-base)/env/boost/random; }  # prefix
brandom-bdir(){ echo $(local-base)/env/boost/random.build ; }
brandom-sdir(){ echo $(opticks-home)/boost/random ; }

brandom-icd(){  cd $(brandom-idir); }
brandom-bcd(){  cd $(brandom-bdir); }
brandom-scd(){  cd $(brandom-sdir); }

brandom-cd(){  cd $(brandom-sdir); }


brandom-wipe(){
    local bdir=$(brandom-bdir)
    rm -rf $bdir
}


brandom-cmake(){
   local bdir=$(brandom-bdir)
   mkdir -p $bdir
   brandom-bcd
   cmake $(brandom-sdir) -DCMAKE_INSTALL_PREFIX=$(brandom-idir) -DCMAKE_BUILD_TYPE=Debug 
}

brandom-make(){
    local iwd=$PWD
    brandom-bcd
    make $*
    cd $iwd
}

brandom-install(){
   brandom-make install
}


brandom-bbin(){ echo $(brandom-bdir)/GGeoTest ; }
brandom-bin(){ echo $(brandom-idir)/bin/${1:-GGeoTest} ; }


brandom-export(){
    env | grep GGEO
}


brandom-run(){
    brandom-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 

    brandom-export 
    $DEBUG $(brandom-bin) $*  
}

brandom--(){
    brandom-make
    [ $? -ne 0 ] && echo $FUNCNAME ERROR && return 1 
    brandom-install $*
}

brandom-lldb(){
    DEBUG=lldb brandom-run
}

brandom-brun(){
   echo running from bdir not idir : no install needed, but much set library path
   local bdir=$(brandom-bdir)
   DYLD_LIBRARY_PATH=$bdir $DEBUG $bdir/GGeoTest 
}



