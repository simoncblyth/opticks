# === func-gen- : boost/bregex/bregex fgp boost/bregex/bregex.bash fgn bregex fgh boost/bregex
bregex-src(){      echo boost/bregex/bregex.bash ; }
bregex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bregex-src)} ; }
bregex-vi(){       vi $(bregex-source) ; }
bregex-env(){      elocal- ; }
bregex-usage(){ cat << EOU

*bregex-test*
    for developing boost regex patterns, via search matching against cin

    ::

        bregex-test "<[^>]*>"
        bregex-test "\w*\d"


::

    simon:~ blyth$ bregex-test "\"(.*?)\""
    search cin for text matching regex "(.*?)"
    pairs plucked using regexp : 2
                            path.h :           
                      other_path.h :         




*bregex-enum*
    plucking names and values from an enum string

    ::
       
        bregex-enum "^\s*(\\w+)\s*=\s*(.*?),*\s*?$"


EOU
}

bregex-dir(){ echo $(env-home)/boost/bregex ; }
bregex-cd(){  cd $(bregex-dir); }


bregex-name(){ echo bregex ; }

bregex-sdir(){ echo $(env-home)/boost/bregex ; }
bregex-idir(){ echo $(local-base)/env/boost/bregex ; }
bregex-bdir(){ echo $(bregex-idir).build ; }

bregex-scd(){  cd $(bregex-sdir); }
bregex-cd(){  cd $(bregex-sdir); }

bregex-icd(){  cd $(bregex-idir); }
bregex-bcd(){  cd $(bregex-bdir); }


bregex-wipe(){
   local bdir=$(bregex-bdir)
   rm -rf $bdir
}
bregex-env(){    
    elocal- 
}

bregex-cmake(){
   local iwd=$PWD

   local bdir=$(bregex-bdir)
   mkdir -p $bdir
  
   bregex-bcd 
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(bregex-idir) \
       $(bregex-sdir)

   cd $iwd
}

bregex-make(){
   local iwd=$PWD

   bregex-bcd
   make $*

   cd $iwd
}

bregex-install(){
   bregex-make install
}

bregex-bin(){ echo $(bregex-idir)/bin ; }
bregex-export()
{
   echo -n
}

bregex-run(){
   local bin=$(bregex-bin)
   bregex-export
   $bin $*
}

bregex-lldb()
{
   local bin=$(bregex-bin)
   bregex-export
   lldb $bin -- $*
}

bregex--()
{
    bregex-wipe
    bregex-cmake
    bregex-make
    bregex-install
}




bregex-hello()
{
   echo "<hello>world</hello>" | $(bregex-bin)/regexsearchTest 
}

bregex-test(){ $FUNCNAME- | $(bregex-bin)/regexsearchTest "$1" ; }
bregex-test-(){ cat << EOT
   <hello>world1</hello>
   <hello>world2</hello>
   <hello>world3</hello>
   <hello>world4</hello>

    \$ENV_HOME/graphics/ggeoview/cu/photon.h

#incl "path.h"
#incl "other_path.h"

EOT
}






bregex-enum(){ $FUNCNAME- | $(bregex-bin)/regexsearchTest "$1" ; }
bregex-enum-(){ cat << EOE
enum
{
    NO_HIT                 = 0x1 << 0,
    BULK_ABSORB            = 0x1 << 1,
    SURFACE_DETECT         = 0x1 << 2,
    SURFACE_ABSORB         = 0x1 << 3,
    RAYLEIGH_SCATTER       = 0x1 << 4,
    REFLECT_DIFFUSE        = 0x1 << 5,
    REFLECT_SPECULAR       = 0x1 << 6,
    SURFACE_REEMIT         = 0x1 << 7,
    SURFACE_TRANSMIT       = 0x1 << 8,
    BULK_REEMIT            = 0x1 << 9,
    GENERATE_SCINTILLATION = 0x1 << 16, 
    GENERATE_CERENKOV      = 0x1 << 17, 
    BOUNDARY_SPOL          = 0x1 << 18, 
    BOUNDARY_PPOL          = 0x1 << 19, 
    BOUNDARY_REFLECT       = 0x1 << 20, 
    BOUNDARY_TRANSMIT      = 0x1 << 21, 
    BOUNDARY_TIR           = 0x1 << 22, 
    BOUNDARY_TIR_NOT       = 0x1 << 23, 
    NAN_ABORT              = 0x1 << 31
}; // processes

EOE
}



bregex-photon(){  $(bregex-bin)/enum_regexsearchTest \$ENV_HOME/graphics/ggeoview/cu/photon.h ; }



