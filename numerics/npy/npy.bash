# === func-gen- : numerics/npy/npy fgp numerics/npy/npy.bash fgn npy fgh numerics/npy
npy-src(){      echo numerics/npy/npy.bash ; }
npy-source(){   echo ${BASH_SOURCE:-$(env-home)/$(npy-src)} ; }
npy-vi(){       vi $(npy-source) ; }
npy-env(){      elocal- ; }
npy-usage(){ cat << EOU





EOU
}

npy-sdir(){ echo $(env-home)/numerics/npy ; }
npy-idir(){ echo $(local-base)/env/numerics/npy ; }
npy-bdir(){ echo $(local-base)/env/numerics/npy.build ; }

npy-cd(){   cd $(npy-sdir); }
npy-scd(){  cd $(npy-sdir); }
npy-icd(){  cd $(npy-idir); }
npy-bcd(){  cd $(npy-bdir); }

npy-wipe(){
   local bdir=$(npy-bdir)
   rm -rf $bdir
}

npy-cmake(){
   local iwd=$PWD

   local bdir=$(npy-bdir)
   mkdir -p $bdir

   npy-bcd
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(npy-idir) \
       $(npy-sdir)

   cd $iwd
}

npy-make(){
   local iwd=$PWD

   npy-bcd
   make $*

   cd $iwd
}

npy-install(){
   npy-make install
}

npy--()
{
    npy-wipe
    npy-cmake
    npy-make
    npy-install
}

