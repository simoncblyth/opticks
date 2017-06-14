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

xercesc-prefix-old(){  

  case $(uname -s) in 
      Darwin) echo /opt/local ;;
    MINGW64*) echo /mingw64 ;;
           *) echo ${LOCAL_BASE:-/usr/local}/opticks/externals ;;
  esac  
}

xercesc-prefix(){  
    echo $(opticks-prefix)/externals
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
  elif [ "$NODE_TAG" == "X" ]; then
      echo /lib64/libxerces-c-3.1.so    
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
xercesc-dist(){ echo $(basename $(xercesc-url)); }
xercesc-name(){ local dist=$(xercesc-dist) ; echo ${dist/.tar.gz} ; }
xercesc-base(){ echo $(opticks-prefix)/externals/xercesc ; }

xercesc-dir(){  echo $(xercesc-prefix)/xercesc/$(xercesc-name) ; }
xercesc-bdir(){ echo $(xercesc-prefix)/xercesc/$(xercesc-name).build ; }

xercesc-info(){ cat << EOI

$FUNCNAME
==============

   xercesc-url    : $(xercesc-url)
   xercesc-dist   : $(xercesc-dist)
   xercesc-name   : $(xercesc-name)
   xercesc-base   : $(xercesc-base)
   xercesc-dir    : $(xercesc-dir)
   xercesc-bdir   : $(xercesc-bdir)

   xercesc-prefix  : $(xercesc-prefix)
   xercesc-library : $(xercesc-library)
   xercesc-include-dir : $(xercesc-include-dir)


EOI
}



xercesc-cd(){   cd $(xercesc-dir); }
xercesc-bcd(){  cd $(xercesc-bdir); }

xercesc-get(){
   local dir=$(dirname $(xercesc-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(xercesc-url)
   local tgz=$(xercesc-dist)
   local nam=$(xercesc-name)
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxf $tgz 
}

xercesc-configure()
{
   xercesc-cd
   ./configure --prefix=$(xercesc-prefix)
}

xercesc-make()
{
   xercesc-cd
   make ${1:-install}
}

xercesc--()
{
   xercesc-info
   xercesc-get
   xercesc-configure
   xercesc-make
}


