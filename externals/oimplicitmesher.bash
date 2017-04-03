oimplicitmesher-src(){      echo externals/oimplicitmesher.bash ; }
oimplicitmesher-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(oimplicitmesher-src)} ; }
oimplicitmesher-vi(){       vi $(oimplicitmesher-source) ; }
oimplicitmesher-env(){      olocal- ; opticks- ; }
oimplicitmesher-usage(){ cat << EOU

ImplicitMesher as Opticks External 
====================================

See also env-;implicitmesher-


EOU
}

oimplicitmesher-edit(){ vi $(opticks-home)/cmake/Modules/FindImplicitMesher.cmake ; }
oimplicitmesher-url(){ echo https://bitbucket.com/simoncblyth/implicitmesher ; }

oimplicitmesher-dir(){  echo $(opticks-prefix)/externals/implicitmesher/implicitmesher ; }
oimplicitmesher-bdir(){ echo $(opticks-prefix)/externals/implicitmesher/implicitmesher.build ; }
oimplicitmesher-prefix(){ echo $(opticks-prefix)/externals ; }


oimplicitmesher-cd(){  cd $(oimplicitmesher-dir); }
oimplicitmesher-bcd(){ cd $(oimplicitmesher-bdir) ; }

oimplicitmesher-fullwipe()
{
    rm -rf $(opticks-prefix)/externals/implicitmesher 
    rm -f  $(opticks-prefix)/externals/lib/libImplicitMesher.dylib 
    rm -rf $(opticks-prefix)/externals/include/ImplicitMesher
    ## test executables not removed
}

oimplicitmesher-get(){
   local iwd=$PWD
   local dir=$(dirname $(oimplicitmesher-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d implicitmesher ] && hg clone $(oimplicitmesher-url)
   cd $iwd
}

oimplicitmesher-cmake()
{
    local iwd=$PWD
    local bdir=$(oimplicitmesher-bdir)

    mkdir -p $bdir
    #[ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already && return  
    rm -f "$bdir/CMakeCache.txt"

    oimplicitmesher-bcd   
    opticks-

    cmake \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(oimplicitmesher-prefix) \
       $* \
       $(oimplicitmesher-dir)


    cd $iwd
}

oimplicitmesher-make()
{
    local iwd=$PWD
    oimplicitmesher-bcd
    cmake --build . --config Release --target ${1:-install}
    cd $iwd
}


oimplicitmesher--()
{
   oimplicitmesher-get
   oimplicitmesher-cmake
   oimplicitmesher-make install
}

oimplicitmesher-t()
{
   oimplicitmesher-make test
}

