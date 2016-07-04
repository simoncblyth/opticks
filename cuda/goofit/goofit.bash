# === func-gen- : cuda/goofit/goofit fgp cuda/goofit/goofit.bash fgn goofit fgh cuda/goofit
goofit-src(){      echo cuda/goofit/goofit.bash ; }
goofit-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(goofit-src)} ; }
goofit-vi(){       vi $(goofit-source) ; }
goofit-env(){      olocal- ; }
goofit-usage(){ cat << EOU


* https://github.com/GooFit/GooFit

Massively-parallel framework for maximum-likelihood fits, implemented in CUDA.
Modelled after RooFit.



EOU
}
goofit-dir(){ echo $(local-base)/env/cuda/GooFit ; }
goofit-cd(){  cd $(goofit-dir); }
goofit-mate(){ mate $(goofit-dir) ; }
goofit-get(){
   local dir=$(dirname $(goofit-dir)) &&  mkdir -p $dir && cd $dir


    git clone https://github.com/GooFit/GooFit

}
