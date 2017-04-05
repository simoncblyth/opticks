odcs-src(){      echo externals/odcs.bash ; }
odcs-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(odcs-src)} ; }
odcs-vi(){       vi $(odcs-source) ; }
odcs-env(){      olocal- ; opticks- ; }
odcs-usage(){ cat << EOU

DualContouringSample as Opticks External 
==========================================

See also env-;dcs-


EOU
}

odcs-edit(){ vi $(opticks-home)/cmake/Modules/FindDualContouringSample.cmake ; }
odcs-url(){ echo https://github.com/simoncblyth/DualContouringSample ; }

odcs-dir(){  echo $(opticks-prefix)/externals/dualcontouringsample ; }
odcs-bdir(){ echo $(opticks-prefix)/externals/dualcontouringsample/dualcontouringsample.build ; }
odcs-prefix(){ echo $(opticks-prefix)/externals ; }


odcs-cd(){  cd $(odcs-dir); }
odcs-bcd(){ cd $(odcs-bdir) ; }

odcs-fullwipe()
{
    rm -rf $(opticks-prefix)/externals/DualContouringSample
    rm -f  $(opticks-prefix)/externals/lib/libDualContouringSample.dylib 
    rm -rf $(opticks-prefix)/externals/include/DualContouringSample
    ## test executables not removed
}

odcs-get(){
   local iwd=$PWD
   local dir=$(dirname $(odcs-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d DualContouringSample ] && git clone $(odcs-url)
   cd $iwd
}

odcs-cmake()
{
    local iwd=$PWD
    local bdir=$(odcs-bdir)

    mkdir -p $bdir
    #[ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already && return  
    rm -f "$bdir/CMakeCache.txt"

    odcs-bcd   
    opticks-

    cmake \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(odcs-prefix) \
       $* \
       $(odcs-dir)


    cd $iwd
}

odcs-make()
{
    local iwd=$PWD
    odcs-bcd
    cmake --build . --config Release --target ${1:-install}
    cd $iwd
}


odcs--()
{
   odcs-get
   odcs-cmake
   odcs-make install
}

odcs-t()
{
   odcs-make test
}

