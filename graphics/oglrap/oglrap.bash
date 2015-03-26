# === func-gen- : graphics/oglrap/oglrap fgp graphics/oglrap/oglrap.bash fgn oglrap fgh graphics/oglrap
oglrap-src(){      echo graphics/oglrap/oglrap.bash ; }
oglrap-source(){   echo ${BASH_SOURCE:-$(env-home)/$(oglrap-src)} ; }
oglrap-vi(){       vi $(oglrap-source) ; }
oglrap-env(){      elocal- ; }
oglrap-usage(){ cat << EOU

Featherweight OpenGL wrapper
==============================

Just a few utility classes to make modern OpenGL 3, 4 
easier to use.




EOU
}
oglrap-dir(){  echo $(local-base)/env/graphics/oglrap ; }
oglrap-sdir(){ echo $(env-home)/graphics/oglrap ; }
oglrap-cd(){   cd $(oglrap-dir); }
oglrap-scd(){  cd $(oglrap-sdir); }



