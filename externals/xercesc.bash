# === func-gen- : xml/xercesc/xercesc fgp externals/xercesc.bash fgn xercesc fgh xml/xercesc
xercesc-src(){      echo externals/xercesc.bash ; }
xercesc-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(xercesc-src)} ; }
xercesc-vi(){       vi $(xercesc-source) ; }
xercesc-env(){      olocal- ; }
xercesc-usage(){ cat << EOU

XERCESC
========

* XML handling package required for Geant4 GDML support


EOU
}

xercesc-prefix(){  
  case $(uname -s) in 
      Darwin) echo /opt/local ;;
    MINGW64*) echo /mingw64 ;;
           *) echo ${LOCAL_BASE:-/usr/local}/opticks/externals ;;
  esac  
}


xercesc-include-dir(){ 
  if [ "$NODE_TAG" == "MGB" ]; then
      ome-
      ome-xercesc-include
  else
      echo $(xercesc-prefix)/include ;
  fi
}
xercesc-library(){  

  if [ "$NODE_TAG" == "MGB" ]; then
      ome-
      ome-xercesc-lib
  else 
      case $(uname -s) in 
           Darwin) echo $(xercesc-prefix)/lib/libxerces-c.dylib   ;;
         MINGW64*) echo $(xercesc-prefix)/bin/libxerces-c-3-1.dll ;;
                *) echo $(xercesc-prefix)/lib/libxerces-c-3-1.so  ;;
      esac
  fi 
}

xercesc-geant4-export(){
  export XERCESC_INCLUDE_DIR=$(xercesc-include-dir)
  export XERCESC_LIBRARY=$(xercesc-library)
  export XERCESC_ROOT_DIR=$(xercesc-prefix)
}


xercesc-url(){ echo http://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.1.1.tar.gz ; }

xercesc--() {
    local url=$(xercesc-url)
    local tar=$(basename $url)
    local dir=${tar/.tar.gz/}

    cd $(xercesc-prefix)
    # check tar exists or not
    if [ ! -f "$tar" ]; then
        # download
        wget $url
        # uncompress
        tar zxvf $tar
    fi

    if [ ! -d "$dir" ]; then
        echo 1>&2 can not find "$dir"
        return
    fi

    cd $dir

    # build
    ./configure --prefix=$(xercesc-prefix)
    make
    make install
}

