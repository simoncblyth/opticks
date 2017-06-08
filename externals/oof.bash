oof-src(){      echo externals/oof.bash ; }
oof-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(oof-src)} ; }
oof-vi(){       vi $(oof-source) ; }
oof-usage(){ cat << EOU

OpenFlipper
=============


EOU
}

oof-env(){  olocal- ; opticks- ; }

oof-info(){ cat << EOI

    name : $(oof-name)
    dist : $(oof-dist)



EOI
}


oof-vers(){ echo 3.1 ; }
oof-name(){ echo OpenFlipper-$(oof-vers) ; }
oof-url(){  echo http://www.openflipper.org/media/Releases/$(oof-vers)/$(oof-name).tar.gz ; }

oof-dist(){ echo $(dirname $(oof-dir))/$(basename $(oof-url)) ; }

oof-base(){ echo $(opticks-prefix)/externals/openflipper ; }
oof-prefix(){ echo $(opticks-prefix)/externals ; }
oof-idir(){ echo $(oof-prefix) ; }

oof-dir(){  echo $(oof-base)/$(oof-name) ; }
oof-bdir(){ echo $(oof-base)/$(oof-name).build ; }

oof-ecd(){  cd $(oof-edir); }
oof-cd(){   cd $(oof-dir)/$1 ; }
oof-bcd(){  cd $(oof-bdir); }
oof-icd(){  cd $(oof-idir); }


oof-get(){
   local dir=$(dirname $(oof-dir)) &&  mkdir -p $dir && cd $dir
   local url="$(oof-url)"
   local tgz=$(basename $url)
   local nam=${tgz/.tar.gz}
   [ ! -f "$tgz" ] && curl -L -O $url
   [ ! -d "$nam" ] && tar zxf $tgz 
}

oof-doc(){ oof-html ; }
oof-html(){ open $(oof-dir)/Documentation/index.html ; }

oof-find(){ oof-cd ; find src -type f -exec grep -H ${1:-DefaultTraits} {} \; ; }


oof-wipe(){
  local bdir=$(oof-bdir)
  rm -rf $bdir 

}

oof-edit(){ vi $(opticks-home)/cmake/Modules/FindOpenFlipper.cmake ; }

oof-cmake(){
  local iwd=$PWD
  local bdir=$(oof-bdir)
  mkdir -p $bdir

  [ -f "$bdir/CMakeCache.txt" ] && echo $msg already configured : oof-configure to reconfigure && return 

  oof-bcd


  # -G "$(opticks-cmake-generator)" \

  cmake $(oof-dir) \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=$(oof-prefix) \
      -DBUILD_APPS=OFF 

  cd $iwd
}

oof-configure()
{
   oof-wipe
   oof-cmake $*
}


oof-make(){
  local iwd=$PWD
  oof-bcd

  cmake --build . --config Release --target ${1:-install}

  cd $iwd
}

oof--(){
  oof-get 
  oof-cmake
  oof-make install
}




