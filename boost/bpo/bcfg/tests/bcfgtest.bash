bcfgtest-src(){      echo boost/bpo/bcfg/test/bcfgtest.bash ; }
bcfgtest-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bcfgtest-src)} ; }
bcfgtest-vi(){       vi $(bcfgtest-source) ; }
bcfgtest-env(){      elocal- ; }
bcfgtest-usage(){ cat << EOU




EOU
}


bcfgtest-sdir(){ echo $(env-home)/boost/bpo/bcfg/test ; }
bcfgtest-idir(){ echo $(local-base)/env/boost/bpo/bcfg/test ; }
bcfgtest-bdir(){ echo $(bcfgtest-idir).build ; }

bcfgtest-scd(){  cd $(bcfgtest-sdir); }
bcfgtest-cd(){  cd $(bcfgtest-sdir); }

bcfgtest-icd(){  cd $(bcfgtest-idir); }
bcfgtest-bcd(){  cd $(bcfgtest-bdir); }
bcfgtest-name(){ echo CfgTest ; }


bcfgtest-wipe(){
   local bdir=$(bcfgtest-bdir)
   rm -rf $bdir
}


bcfgtest-cmake(){
   local iwd=$PWD

   local bdir=$(bcfgtest-bdir)
   mkdir -p $bdir

   bcfgtest-bcd
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(bcfgtest-idir) \
       $(bcfgtest-sdir)

   cd $iwd
}

bcfgtest-make(){
   local iwd=$PWD

   bcfgtest-bcd
   make $*

   cd $iwd
}

bcfgtest-install(){
   bcfgtest-make install
}

bcfgtest-bin(){ echo $(bcfgtest-idir)/bin/$(bcfgtest-name) ; }

bcfgtest-export()
{
   echo -n
}

bcfgtest-run(){
   local bin=$(bcfgtest-bin)
   bcfgtest-export
   $bin $*
}

bcfgtest-runq(){
   local bin=$(bcfgtest-bin)
   bcfgtest-export

   local parms=""
   local p
   for p in "$@" ; do
      [ "${p/ /}" == "$p" ] && parms="${parms} $p" || parms="${parms} \"${p}\""
   done

   cat << EOC  | sh 
   $bin $parms
EOC
}


bcfgtest--()
{
    bcfgtest-wipe
    bcfgtest-cmake
    bcfgtest-make
    bcfgtest-install
}




