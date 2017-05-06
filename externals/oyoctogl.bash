oyoctogl-src(){      echo externals/oyoctogl.bash ; }
oyoctogl-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(oyoctogl-src)} ; }
oyoctogl-vi(){       vi $(oyoctogl-source) ; }
oyoctogl-env(){      olocal- ; opticks- ; }
oyoctogl-usage(){ cat << EOU

Yocto-GL as Opticks External
====================================

See also env-;yoctogl-


EOU
}

oyoctogl-edit(){ vi $(opticks-home)/cmake/Modules/FindYoctoGL.cmake ; }
oyoctogl-url(){ echo https://github.com/simoncblyth/yocto-gl ; }


oyoctogl-dir(){  echo $(opticks-prefix)/externals/yoctogl/yocto-gl ; }
oyoctogl-bdir(){ echo $(opticks-prefix)/externals/yoctogl/yocto-gl.build ; }
oyoctogl-prefix(){ echo $(opticks-prefix)/externals ; }


oyoctogl-cd(){  cd $(oyoctogl-dir); }
oyoctogl-bcd(){ cd $(oyoctogl-bdir) ; }

oyoctogl-fullwipe()
{
    rm -rf $(opticks-prefix)/externals/yoctogl 
}

oyoctogl-update()
{
    oyoctogl-fullwipe
    oyoctogl-- 
}


oyoctogl-get(){
   local iwd=$PWD
   local dir=$(dirname $(oyoctogl-dir)) &&  mkdir -p $dir && cd $dir
   [ ! -d yocto-gl ] && git clone $(oyoctogl-url)
   cd $iwd
}

oyoctogl-cmake()
{
    local iwd=$PWD
    local bdir=$(oyoctogl-bdir)
    local sdir=$(oyoctogl-dir)

    #rm -rf $bdir
    mkdir -p $bdir
    #[ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already && return  
    rm -f "$bdir/CMakeCache.txt"

    oyoctogl-bcd   
    opticks-


    cmake \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(oyoctogl-prefix) \
       $* \
       $sdir


    cd $iwd
}

oyoctogl-make()
{
    local iwd=$PWD
    oyoctogl-bcd
    cmake --build . --config Release --target ${1:-install}
    cd $iwd
}


oyoctogl--()
{
   oyoctogl-get
   oyoctogl-cmake
   oyoctogl-make install
}

oyoctogl-t()
{
   # oyoctogl-make test
   ygltf_reader $TMP/nd/scene.gltf
}





