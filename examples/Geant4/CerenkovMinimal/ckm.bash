ckm-source(){ echo $BASH_SOURCE ; }
ckm-vi(){ vi $(ckm-source) ; }
ckm-env(){ echo -n ; }
ckm-usage(){ cat << EOU

EOU
}
ckm-dir(){ echo $(dirname $(ckm-source)) ; }
ckm-cd(){  cd $(ckm-dir) ; }

ckm--(){ ckm-cd ; ./go.sh ; }

