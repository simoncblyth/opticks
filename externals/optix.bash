optix-source(){   echo ${BASH_SOURCE} ; }
optix-vi(){       vi $(optix-source) ; }
optix-env(){      olocal- ; }
optix-usage(){ cat << \EOU

NVIDIA OptiX Ray Trace Toolkit
================================== 

See Also
------------

* optixnote-  thousands of lines of lots of notes on OptiX versions and usage, that used to be here


EOU
}



optix-export(){
   export OPTIX_SDK_DIR=$(optix-sdk-dir)
   export OPTIX_INSTALL_DIR=$(optix-install-dir)
   export OPTIX_SAMPLES_INSTALL_DIR=$(optix-samples-install-dir)
}


optix-fold(){ echo ${OPTICKS_OPTIX_HOME:-$($FUNCNAME-)} ; }
optix-fold-(){ 
   case $(uname) in
     Darwin) echo /Developer ;;
          *) echo /usr/local ;; 
   esac
}

optix-fold-old-(){
   case $NODE_TAG in 
      D)  echo /Developer ;;
      G1) echo $(local-base) ;;
      LT) echo /home/ihep/data/repo/opticks ;;
     GTL) echo /afs/ihep.ac.cn/soft/juno/JUNO-ALL-SLC6/GPU/20150723 ;;
      *) echo $(local-base) ;;
   esac
}


optix-prefix(){      echo $(optix-fold)/OptiX ; }
optix-dir(){         echo $(optix-fold)/OptiX/SDK ; }
optix-sdk-dir-old(){ echo $(optix-fold)/OptiX_301/SDK ; }
optix-sdk-dir(){     echo $(optix-fold)/OptiX/SDK ; }
optix-download-dir(){ echo $(local-base)/env/cuda ; }
optix-bdir(){         echo $(local-base)/env/cuda/$(optix-name) ; }
optix-install-dir(){ echo $(dirname $(optix-sdk-dir)) ; }
optix-idir(){        echo $(dirname $(optix-sdk-dir))/include ; }
optix-sdir(){        echo $(opticks-home)/optix ; }
optix-samples-src-dir(){     echo $(local-base)/env/cuda/$(optix-name)_sdk ; }
optix-samples-install-dir(){ echo $(local-base)/env/cuda/$(optix-name)_sdk_install ; }

optix-samples-scd(){   cd $(optix-samples-src-dir)/$1 ; }
optix-samples-cd(){    cd $(optix-samples-install-dir)/$1 ; }
optix-download-cd(){   cd $(optix-download-dir) ; }

optix-ftp(){ open https://ftpservices.nvidia.com ; }

optix-info(){ cat << EOI

   optix-fold    : $(optix-fold)
   optix-dir     : $(optix-dir)
   optix-sdk-dir : $(optix-sdk-dir)

EOI
}



optix-c(){   cd $(optix-dir); }
optix-cd(){  cd $(optix-dir); }
optix-bcd(){ cd $(optix-samples-install-dir); }
optix-scd(){ cd $(optix-sdir); }
optix-icd(){ cd $(optix-idir); }
optix-doc(){ cd $(optix-fold)/OptiX/doc ; }

optix-samples-find(){    optix-samples-cppfind $* ; }
optix-samples-cufind(){  find $(optix-samples-src-dir) -name '*.cu'  -exec grep ${2:--H} ${1:-rtReportIntersection} {} \; ;}
optix-samples-hfind(){   find $(optix-samples-src-dir) -name '*.h'   -exec grep ${2:--H} ${1:-rtReportIntersection} {} \; ;}
optix-samples-cppfind(){ find $(optix-samples-src-dir) -name '*.cpp' -exec grep ${2:--H} ${1:-rtReportIntersection} {} \; ;}
optix-find(){            find $(optix-idir)            -name '*.h'   -exec grep ${2:--H} ${1:-setMiss} {} \; ; }
optix-ifind(){           find $(optix-idir)            -name '*.h'   -exec grep ${2:--H} ${1:-setMiss} {} \; ; }

optix-x(){ find $(optix-dir) -name "*.${1}" -exec grep ${3:--H} ${2:-Sampler} {} \; ; }
optix-cu(){  optix-x cu  $* ; }
optix-cpp(){ optix-x cpp $* ; }
optix-h(){   optix-x h   $* ; }

optix-find(){
   optix-cu  $*
   optix-cpp $*
   optix-h  $*
}





optix-api-(){ echo $(optix-fold)/OptiX/doc/OptiX_API_Reference_$(optix-version).pdf ; }
optix-pdf-(){ echo $(optix-fold)/OptiX/doc/OptiX_Programming_Guide_$(optix-version).pdf ; }
optix-api(){ open $(optix-api-) ; }
optix-pdf(){ open $(optix-pdf-) ; }



optix-readlink(){ readlink $(optix-fold)/OptiX ; }
optix-name(){  echo ${OPTIX_NAME:-$(optix-readlink)} ; }
optix-jump(){    
   local iwd=$PWD
   local ver=${1:-301}
   cd $(optix-fold)
   sudo ln -sfnv OptiX_$ver OptiX 
   cd $iwd
}
optix-old(){   optix-jump 301 ; }
optix-new(){   optix-jump 411 ; }
optix-beta(){  optix-jump 370b2 ; }

optix-linux-name(){
   case $1 in 
      351) echo NVIDIA-OptiX-SDK-3.5.1-PRO-linux64 ;;
      363) echo NVIDIA-OptiX-SDK-3.6.3-linux64 ;;
      370) echo NVIDIA-OptiX-SDK-3.7.0-linux64 ;;
   esac
}

optix-version(){
   case $(optix-name) in 
     OptiX_501)   echo 5.0.1 ;;
     OptiX_411)   echo 4.1.1 ;;
     OptiX_400)   echo 4.0.0 ;;
     OptiX_380)   echo 3.8.0 ;;
     OptiX_301)   echo 3.0.2 ;;
     OptiX_370b2) echo 3.7.0 ;;
  esac
}

optix-vernum(){
   case $(optix-name) in 
     OptiX_411)   echo 411 ;;
     OptiX_400)   echo 400 ;;
     OptiX_380)   echo 380 ;;
     OptiX_301)   echo 302 ;;
     OptiX_370b2) echo 370 ;;
  esac
}

optix-header(){ echo $(optix-prefix)/include/optix.h ; }
optix-hversion(){ perl -ne 's,#define OPTIX_VERSION (\d*)\s*,$1, && print "$1\n"' $(optix-header) ; }
   

optix-linux-jump(){
    local vers=${1:-351}
    local name=$(optix-linux-name $vers)
    [ -z "$name" ] && echo $msg version $vers unknown && type optix-linux-name && return 

    cd $(optix-fold)    
    ln -sfnv $name OptiX
}

   



optix-samples-names(){ cat << EON
CMakeLists.txt
sampleConfig.h.in
cuda
CMake
sample1
sample2
sample3
sample4
sample5
sample5pp
sample6
sample7
sample8
simpleAnimation
sutil
tutorial
materials
transparency
EON

## remember that after adding a name here, need to uncomment in the CMakeLists.txt for it to get built
}



optix-samples-get-all(){

   local src=$(optix-sdk-dir)
   local dst=$(optix-samples-src-dir)
 
   mkdir -p $dst

   echo $FUNCNAME copy all samples to somewhere writable 
   cp -R $src/* $dst/
 
}


optix-samples-get-selected(){
   local sdir=$(optix-samples-src-dir)
   mkdir -p $sdir

   local src=$(optix-sdk-dir)
   local dst=$sdir
   local cmd
   local name
   optix-samples-names | while read name ; do 

      if [ -d "$src/$name" ]
      then 
          if [ ! -d "$dst/$name" ] ; then 
              cmd="cp -r $src/$name $dst/"
          else
              cmd="echo destination directory exists already $dst/$name"
          fi
      elif [ -f "$src/$name" ] 
      then 
          if [ ! -f "$dst/$name" ] ; then 
              cmd="cp $src/$name $dst/$name"
          else
              cmd="echo destination file exists already $dst/$name"
          fi
      else
          cmd="echo src $src/$name missing"
      fi 
      #echo $cmd
      eval $cmd
   done
}


optix-cuda-nvcc-flags(){
    case $NODE_TAG in 
       D) echo -ccbin /usr/bin/clang --use_fast_math ;;
       *) echo --use_fast_math ;; 
    esac
}



#optix-samples-cmake-kludge(){
#    optix-samples-scd
#    grep cmake_minimum_required CMakeLists.txt 
#    perl -pi -e 's,2.8.8,2.6.4,' CMakeLists.txt 
#    grep cmake_minimum_required CMakeLists.txt 
#}


optix-samples-cmake(){
    local iwd=$PWD
    local bdir=$(optix-samples-install-dir)
    #rm -rf $bdir   # starting clean 
    mkdir -p $bdir
    optix-bcd
    cmake -DOptiX_INSTALL_DIR=$(optix-install-dir) \
          -DCUDA_NVCC_FLAGS="$(optix-cuda-nvcc-flags)" \
           "$(optix-samples-src-dir)"
    cd $iwd
}

optix-samples-make(){
    local iwd=$PWD
    optix-bcd
    make $* 
    cd $iwd
}




optix-samples-run(){
    local name=${1:-materials}
    optix-samples-make $name
    local cmd="$(optix-bdir)/bin/$name"
    echo $cmd
    eval $cmd
}

optix-tutorial(){
    local tute=${1:-10}

    optix-samples-make tutorial

    local cmd="$(optix-bdir)/bin/tutorial -T $tute --texture-path $(optix-sdk-dir)/tutorial/data"
    echo $cmd
    eval $cmd
}


optix-tutorial-cd(){
   cd $(optix-sdk-dir)/tutorial
}
optix-tutorial-vi(){
   vi $(optix-sdk-dir)/tutorial/*
}




optix-verbose(){
  export VERBOSE=1 
}
optix-unverbose(){
  unset VERBOSE
}



optix-check(){
/usr/local/cuda/bin/nvcc -ccbin /usr/bin/clang --verbose -M -D__CUDACC__ /Developer/OptiX/SDK/cuda/triangle_mesh_small.cu -o /usr/local/env/cuda/optix301/sutil/CMakeFiles/cuda_compile_ptx.dir/__/cuda/cuda_compile_ptx_generated_triangle_mesh_small.cu.ptx.NVCC-depend -ccbin /usr/bin/cc -m64 -DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA --use_fast_math -U__BLOCKS__ -DNVCC -I/usr/local/cuda/include -I/Developer/OptiX/include -I/Developer/OptiX/SDK/sutil -I/Developer/OptiX/include/optixu -I/usr/local/env/cuda/optix301 -I/usr/local/cuda/include -I/System/Library/Frameworks/GLUT.framework/Headers -I/Developer/OptiX/SDK/sutil -I/Developer/OptiX/SDK/cuda
}



optix-check-2(){

cd /usr/local/env/cuda/OptiX_301/tutorial && /usr/bin/c++   -DGLUT_FOUND -DGLUT_NO_LIB_PRAGMA -fPIC -O3 -DNDEBUG \
     -I/Developer/OptiX/include \
     -I/Users/blyth/env/cuda/optix/OptiX_301/sutil \
     -I/Developer/OptiX/include/optixu \
     -I/usr/local/env/cuda/OptiX_301 \
     -I/usr/local/cuda/include \
     -I/System/Library/Frameworks/GLUT.framework/Headers \
       -o /dev/null \
       -c /Users/blyth/env/cuda/optix/OptiX_301/tutorial/tutorial.cpp

}



optix-diff(){
   local name=${1:-sutil/MeshScene.h}
   local cmd="diff $(optix-sdk-dir-old)/$name $(optix-sdk-dir)/$name"
   echo $cmd
   eval $cmd
}

optix-rdiff(){
   local rel="sutil"
   local cmd="diff -r --brief $(optix-sdk-dir-old)/$rel $(optix-sdk-dir)/$rel"
   echo $cmd
   eval $cmd
}



optix-pkgname(){ echo NVIDIA-OptiX-SDK-$(optix-version)-mac64 ; }

optix-dmgpath()
{
    echo $(local-base)/env/cuda/$(optix-pkgname).dmg
}
optix-dmgpath-open()
{
    open $(optix-dmgpath)
}
optix-pkgpath()
{
    echo /Volumes/$(optix-pkgname)/$(optix-pkgname).pkg
}
optix-pkgpath-lsbom()
{
    #lsbom "$(pkgutil --bom "$(optix-pkgpath)")" 
    lsbom $(optix-pkgpath)/Contents/Archive.bom 
}




