# === func-gen- : opticks/opticks fgp opticks/opticks.bash fgn opticks fgh opticks
opticks-src(){      echo opticks/opticks.bash ; }
opticks-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opticks-src)} ; }
opticks-vi(){       vi $(opticks-source) ; }
opticks-env(){      elocal- ; }
opticks-usage(){ cat << EOU



EOU
}

opticks-sdir(){ echo $(env-home)/opticks ; }
opticks-idir(){ echo $(local-base)/env/opticks ; }
opticks-bdir(){ echo $(opticks-idir).build ; }

opticks-scd(){  cd $(opticks-sdir); }
opticks-cd(){  cd $(opticks-sdir); }

opticks-icd(){  cd $(opticks-idir); }
opticks-bcd(){  cd $(opticks-bdir); }
opticks-name(){ echo Opticks ; }


opticks-wipe(){
   local bdir=$(opticks-bdir)
   rm -rf $bdir
}


opticks-cmake(){
   local iwd=$PWD

   local bdir=$(opticks-bdir)
   mkdir -p $bdir

   opticks-bcd
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-idir) \
       $(opticks-sdir)

   cd $iwd
}

opticks-make(){
   local iwd=$PWD

   opticks-bcd
   make $*

   cd $iwd
}

opticks-install(){
   opticks-make install
}


opticks--()
{
    opticks-wipe
    opticks-cmake
    opticks-make
    opticks-install
}


