yog-source(){ echo $BASH_SOURCE ; }
yog-vi(){ vi $(yog-source) ; }
yog-env(){  olocal- ; opticks- ; }
yog-usage(){ cat << EOU

YOG Usage 
===================


EOU
}

yog-dir(){ echo $(dirname $(yog-source)) ; }
yog-cd(){  cd $(yog-dir) ; } 
yog-c(){   cd $(yog-dir) ; } 
yog--(){   opticks-- $(yog-dir) ; } 



