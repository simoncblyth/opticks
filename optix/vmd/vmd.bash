# === func-gen- : optix/vmd/vmd fgp optix/vmd/vmd.bash fgn vmd fgh optix/vmd
vmd-src(){      echo optix/vmd/vmd.bash ; }
vmd-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(vmd-src)} ; }
vmd-vi(){       vi $(vmd-source) ; }
vmd-env(){      olocal- ; }
vmd-usage(){ cat << EOU

VMD
====

VMD is a molecular visualization program for displaying, animating, and
analyzing large biomolecular systems using 3-D graphics and built-in scripting.
VMD supports computers running MacOS X, Unix, or Windows, is distributed free
of charge, and includes source code. 

* http://www.ks.uiuc.edu/Research/vmd/
* http://www.ks.uiuc.edu/Research/vmd/vmd-1.9.2/optix.html


EOU
}
vmd-dir(){ echo $(local-base)/env/optix/vmd ; }
vmd-cd(){  cd $(vmd-dir); }
vmd-mate(){ mate $(vmd-dir) ; }
vmd-get(){
   local dir=$(dirname $(vmd-dir)) &&  mkdir -p $dir && cd $dir

   local url=http://www.ks.uiuc.edu/Research/vmd/alpha/optix/vmd-1.9.2beta1optixtest4.bin.LINUXAMD64.opengl.tar.gz
   local tgz=$(basename $url)

   rm -rf $tgz       ## no .cu in the tarball, compiled into libs already ? 
   #[ ! -f "$tgz" ] && curl -L -O $url



}
