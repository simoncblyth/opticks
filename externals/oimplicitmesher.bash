oimplicitmesher-src(){      echo externals/oimplicitmesher.bash ; }
oimplicitmesher-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(oimplicitmesher-src)} ; }
oimplicitmesher-vi(){       vi $(oimplicitmesher-source) ; }
oimplicitmesher-env(){      olocal- ; opticks- ; }
oimplicitmesher-usage(){ cat << EOU

ImplicitMesher as Opticks External 
====================================

See also env-;implicitmesher-

NB uses same prefix as Opticks so that opticks/cmake/Modules/FindGLM.cmake succeeds
this has knock effect of requiring prefixing in the CMake install locations::

    install(TARGETS ${name}  DESTINATION externals/lib)
    install(FILES ${HEADERS} DESTINATION externals/include/${name})


EOU
}

oimplicitmesher-edit(){ vi $(opticks-home)/cmake/Modules/FindImplicitMesher.cmake ; }

oimplicitmesher-url-http(){ echo https://bitbucket.com/simoncblyth/implicitmesher ; }
oimplicitmesher-url-ssh(){  echo ssh://hg@bitbucket.org/simoncblyth/ImplicitMesher ; }
oimplicitmesher-url(){
   case $USER in 
      blyth) oimplicitmesher-url-ssh ;;
          *) oimplicitmesher-url-http ;; 
   esac
}



oimplicitmesher-dir(){  echo $(opticks-prefix)/externals/implicitmesher/implicitmesher ; }
oimplicitmesher-bdir(){ echo $(opticks-prefix)/externals/implicitmesher/implicitmesher.build ; }



oimplicitmesher-cd(){  cd $(oimplicitmesher-dir); }
oimplicitmesher-bcd(){ cd $(oimplicitmesher-bdir) ; }

oimplicitmesher-fullwipe()
{
   # rm -rf  $(opticks-prefix)/externals/implicitmesher
   # moving dev into here .. so dont blow it away 
 
    rm -f  $(opticks-prefix)/externals/lib/libImplicitMesher.dylib 
    rm -rf $(opticks-prefix)/externals/include/ImplicitMesher
}

oimplicitmesher-update()
{
    oimplicitmesher-fullwipe
    oimplicitmesher-- 
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
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
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

