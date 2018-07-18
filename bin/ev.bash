ev-source(){ echo $BASH_SOURCE ; }
ev-vi(){ vi $(ev-source)  ; }
ev-env(){  olocal- ; opticks- ; }
ev-usage(){ cat << EOU

ev : aiming for flexible/minimal initial event comparison
============================================================

* NB keep it simple, this is for initial comparisons only, 
  for specialized comparisons see opticks/ana

EOU
}

ev-base(){  echo  $(local-base)/opticks/evt ; }
ev-cd(){   cd $(ev-base) ; }

ev-a-tag(){ echo J ; }
ev-b-tag(){ echo E ; }

ev-get-(){
   local tag=$1
   local cmd 
   if [ "$tag" == "$NODE_TAG" ] ; then
       cmd="rsync -av $TMP/evt/ $tag/"
   else 
       cmd="rsync -av $tag:$TMP/evt/ $tag/"
   fi 
   echo $cmd
   eval $cmd
}

ev-get(){  
    local dir=$(ev-base)
    mkdir -p $dir && cd $dir
    ev-get- $(ev-a-tag)
    ev-get- $(ev-b-tag)
}

ev-a-dir(){ echo $(ev-a-tag)/dayabay/torch/1 ; }
ev-b-dir(){ echo $(ev-b-tag)/dayabay/torch/1 ; }

ev-a-(){ echo $(ev-base)/$(ev-a-dir) ; }
ev-b-(){ echo $(ev-base)/$(ev-b-dir) ; }

ev-info(){ cat << EOI

    ev-a- : $(ev-a-)
    ev-b- : $(ev-b-)

EOI
}

ev-a(){  cd $(ev-a-); }
ev-b(){  cd $(ev-b-); }

ev-diff()
{  

   ev-cd
   diff -y $(ev-a-dir)/report.txt $(ev-b-dir)/report.txt    


   ev-a 
   np.py
   md5 *

   ev-b
   np.py
   md5 *
}


