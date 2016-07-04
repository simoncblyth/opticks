# === func-gen- : graphics/ppm/ppm fgp graphics/ppm/ppm.bash fgn ppm fgh graphics/ppm
ppm-src(){      echo graphics/ppm/ppm.bash ; }
ppm-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(ppm-src)} ; }
ppm-vi(){       vi $(ppm-source) ; }
ppm-usage(){ cat << EOU

PPM Image Loading
==================

C header only implementation reading PPM 
image file into array of unsigned char


Convert PPM to PNG
-------------------

::

    delta:ppm blyth$ libpng-
    delta:ppm blyth$ cat /System/Library/Frameworks/Tk.framework/Versions/8.5/Resources/Scripts/demos/images/teapot.ppm | libpng-wpng > /tmp/out.png
    Encoding image data...
    Done.
    delta:ppm blyth$ open /tmp/out.png 
    delta:ppm blyth$ cp /System/Library/Frameworks/Tk.framework/Versions/8.5/Resources/Scripts/demos/images/teapot.ppm /tmp/teapot.ppm


EOU
}


ppm-sdir(){ echo $(opticks-home)/graphics/ppm ; }

ppm-env(){      olocal- ; opticks- ; }


ppm-idir(){ echo $(opticks-idir) ; }
ppm-bdir(){ echo $(opticks-bdir PPM) ; }


ppm-scd(){  cd $(ppm-sdir); }
ppm-cd(){   cd $(ppm-sdir); }

ppm-icd(){  cd $(ppm-idir); }
ppm-bcd(){  cd $(ppm-bdir); }
ppm-name(){ echo PPM ; }

ppm-wipe(){
   local bdir=$(ppm-bdir)
   rm -rf $bdir
}

ppm-cmake(){
   local iwd=$PWD

   local bdir=$(ppm-bdir)
   mkdir -p $bdir
  
   ppm-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(ppm-idir) \
       $(ppm-sdir)

   cd $iwd
}

ppm-make(){
   local iwd=$PWD

   ppm-bcd
   make $*

   cd $iwd
}

ppm-install(){
   ppm-make install
}

ppm--()
{
    ppm-wipe
    ppm-cmake
    ppm-make
    ppm-install

}

