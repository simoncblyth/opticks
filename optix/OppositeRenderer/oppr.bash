# === func-gen- : cuda/optix/OppositeRenderer/oppr fgp cuda/optix/OppositeRenderer/oppr.bash fgn oppr fgh cuda/optix/OppositeRenderer
oppr-src(){      echo optix/OppositeRenderer/oppr.bash ; }
oppr-source(){   echo ${BASH_SOURCE:-$(env-home)/$(oppr-src)} ; }
oppr-vi(){       vi $(oppr-source) ; }
oppr-env(){      elocal- ; }
oppr-usage(){ cat << EOU





EOU
}
oppr-dir(){ echo $(local-base)/env/cuda/optix/OppositeRenderer ; }
oppr-cd(){  cd $(oppr-dir); }
oppr-mate(){ mate $(oppr-dir) ; }
oppr-get(){
   local dir=$(dirname $(oppr-dir)) &&  mkdir -p $dir && cd $dir
   git clone https://github.com/apartridge/OppositeRenderer.git 
}

oppr-scene(){
   vi $(oppr-dir)/OppositeRenderer/RenderEngine/scene/Scene.{h,cpp}
}
oppr-renderer(){
   vi $(oppr-dir)/OppositeRenderer/RenderEngine/renderer/OptixRenderer.{h,cpp}
}


