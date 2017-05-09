tgltf-source(){   echo $(opticks-home)/tests/tgltf.bash ; }
tgltf-vi(){       vi $(tgltf-source) ; }
tgltf-usage(){ cat << \EOU

tgltf- 
======================================================


EOU
}

tgltf-env(){      olocal- ;  }
tgltf-dir(){ echo $(opticks-home)/tests ; }
tgltf-cd(){  cd $(tgltf-dir); }

join(){ local IFS="$1"; shift; echo "$*"; }

tgltf-tag(){  echo 1 ; }
tgltf-det(){  echo gltf ; }
tgltf-src(){  echo torch ; }
tgltf-args(){ echo  --det $(tgltf-det) --src $(tgltf-src) ; }

tgltf--(){

    tgltf-

    local cmdline=$*

    op.sh  \
            $cmdline \
            --debugger \
            --gltf 1 \
            --animtimemax 10 \
            --timemax 10 \
            --geocenter \
            --eye 1,0,0 \
            --dbganalytic \
            --tag $(tgltf-tag) --cat $(tgltf-det) \
            --save 
}


