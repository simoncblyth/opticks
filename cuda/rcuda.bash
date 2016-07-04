# === func-gen- : cuda/rcuda fgp cuda/rcuda.bash fgn rcuda fgh cuda
rcuda-src(){      echo cuda/rcuda.bash ; }
rcuda-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(rcuda-src)} ; }
rcuda-vi(){       vi $(rcuda-source) ; }
rcuda-env(){      olocal- ; }
rcuda-usage(){ cat << EOU

rCUDA
======

* http://www.rcuda.net/
* http://www.rcuda.net/index.php/what-s-rcuda.html
* http://www.rcuda.net/index.php/support/documentation.html
* http://en.wikipedia.org/wiki/RCUDA

rCUDA is a middleware that enables Computer Unified Device Architecture CUDA
remoting over a commodity network.

* http://www.slideshare.net/olexandr1/r-cuda-presentationibfeatures120704

   * rCUDA client(all nodes) server(nodes with GPU) within a cluster 



EOU
}
rcuda-dir(){ echo $(local-base)/env/cuda/cuda-rcuda ; }
rcuda-cd(){  cd $(rcuda-dir); }
rcuda-mate(){ mate $(rcuda-dir) ; }
rcuda-get(){
   local dir=$(dirname $(rcuda-dir)) &&  mkdir -p $dir && cd $dir

}
