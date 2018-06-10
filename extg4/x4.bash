x4-source(){ echo $BASH_SOURCE ; }
x4-vi(){ vi $(x4-source) ; }
x4-env(){  olocal- ; opticks- ; }
x4-usage(){ cat << EOU

X4 Usage 
===================


EOU
}


x4-dir(){ echo $(dirname $(x4-source)) ; }
x4-cd(){  cd $(x4-dir) ; }
x4--(){   opticks-- $(x4-dir) ; }





