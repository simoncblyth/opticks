# === func-gen- : cuda/cudaz/cudaz fgp cuda/cudaz/cudaz.bash fgn cudaz fgh cuda/cudaz
cudaz-src(){      echo cuda/cudaz/cudaz.bash ; }
cudaz-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(cudaz-src)} ; }
cudaz-vi(){       vi $(cudaz-source) ; }
cudaz-env(){      elocal- ; }
cudaz-usage(){ cat << EOU

CUDA-Z
=======

* http://cuda-z.sourceforge.net

CUDA-Z 0.9.231 is out at 2014.12.05.




EOU
}
cudaz-dir(){ echo $(local-base)/env/cuda/cudaz/cuda-z-0.9 ; }
cudaz-cd(){  cd $(cudaz-dir); }
cudaz-mate(){ mate $(cudaz-dir) ; }
cudaz-get(){
   local dir=$(dirname $(cudaz-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(cudaz-url)
   local zip=$(basename $url)
   local nam=${zip/.zip}
   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -d "$nam" ] && unzip $nam
}

cudaz-url(){ echo http://downloads.sourceforge.net/project/cuda-z/cuda-z/0.9/cuda-z-0.9.zip ; }


