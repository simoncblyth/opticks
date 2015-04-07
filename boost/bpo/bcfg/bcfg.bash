# === func-gen- : boost/bpo/bcfg/bcfg fgp boost/bpo/bcfg/bcfg.bash fgn bcfg fgh boost/bpo/bcfg
bcfg-src(){      echo boost/bpo/bcfg/bcfg.bash ; }
bcfg-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bcfg-src)} ; }
bcfg-vi(){       vi $(bcfg-source) ; }
bcfg-env(){      elocal- ; }
bcfg-usage(){ cat << EOU





EOU
}


bcfg-sdir(){ echo $(env-home)/boost/bpo/bcfg ; }
bcfg-idir(){ echo $(local-base)/env/boost/bpo/bcfg ; }
bcfg-bdir(){ echo $(bcfg-idir).build ; }

bcfg-scd(){  cd $(bcfg-sdir); }
bcfg-cd(){  cd $(bcfg-sdir); }

bcfg-icd(){  cd $(bcfg-idir); }
bcfg-bcd(){  cd $(bcfg-bdir); }
bcfg-name(){ echo CfgTest ; }


bcfg-wipe(){
   local bdir=$(bcfg-bdir)
   rm -rf $bdir
}


bcfg-cmake(){
   local iwd=$PWD

   local bdir=$(bcfg-bdir)
   mkdir -p $bdir

   bcfg-bcd
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(bcfg-idir) \
       $(bcfg-sdir)

   cd $iwd
}

bcfg-make(){
   local iwd=$PWD

   bcfg-bcd
   make $*

   cd $iwd
}

bcfg-install(){
   bcfg-make install
}

bcfg-bin(){ echo $(bcfg-idir)/bin/$(bcfg-name) ; }

bcfg-export()
{
   echo -n
}

bcfg-run(){
   local bin=$(bcfg-bin)
   bcfg-export
   $bin $*
}

bcfg-runq(){
   local bin=$(bcfg-bin)
   bcfg-export

   local parms=""
   local p
   for p in "$@" ; do
      [ "${p/ /}" == "$p" ] && parms="${parms} $p" || parms="${parms} \"${p}\""
   done

   cat << EOC  | sh 
   $bin $parms
EOC
}


bcfg--()
{
    bcfg-wipe
    bcfg-cmake
    bcfg-make
    bcfg-install
}




