# === func-gen- : boost/bregex/bregex fgp boost/bregex/bregex.bash fgn bregex fgh boost/bregex
bregex-rel(){      echo boost/bregex ; }
bregex-src(){      echo boost/bregex/bregex.bash ; }
bregex-source(){   echo ${BASH_SOURCE:-$(env-home)/$(bregex-src)} ; }
bregex-vi(){       vi $(bregex-source) ; }
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


::

    simon:thrust blyth$ bregex-flags
    extract enum pairs from file $ENV_HOME/graphics/ggeoview/cu/photon.h
    udump : 13
                          CERENKOV : 1 :          1 :          1
                     SCINTILLATION : 2 :          2 :          2
                              MISS : 3 :          4 :          4
                       BULK_ABSORB : 4 :          8 :          8
                       BULK_REEMIT : 5 :         16 :         10
                      BULK_SCATTER : 6 :         32 :         20
                    SURFACE_DETECT : 7 :         64 :         40
                    SURFACE_ABSORB : 8 :        128 :         80
                  SURFACE_DREFLECT : 9 :        256 :        100
                  SURFACE_SREFLECT : a :        512 :        200
                  BOUNDARY_REFLECT : b :       1024 :        400
                 BOUNDARY_TRANSMIT : c :       2048 :        800
                         NAN_ABORT : d :       4096 :       1000
    simon:thrust blyth$ 



EOU
}

bregex-dir(){ echo $(env-home)/boost/bregex ; }
bregex-cd(){  cd $(bregex-dir); }

bregex-env(){      elocal- ; OPTICKS- ;  }

bregex-name(){ echo Bregex ; }
bregex-sdir(){ echo $(env-home)/boost/bregex ; }

bregex-idir(){ echo $(OPTICKS-idir); }
bregex-bdir(){ echo $(OPTICKS-bdir)/$(bregex-rel) ; }

bregex-scd(){  cd $(bregex-sdir); }
bregex-cd(){  cd $(bregex-sdir); }

bregex-icd(){  cd $(bregex-idir); }
bregex-bcd(){  cd $(bregex-bdir); }

bregex-flags(){ $(bregex-idir)/bin/enum_regexsearchTest $* ; }

bregex-wipe(){
   local bdir=$(bregex-bdir)
   rm -rf $bdir
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
    bregex-make clean
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

    \$HOME/.opticks/GColors.json
    \$HOME/.opticks
    \$HOME/opticks/GColors.json


#incl "path.h"
#incl "other_path.h"

    __dd__Geometry__AdDetails__AdSurfacesAll__AdCableTraySurface 

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



