examples-vi(){ vi $BASH_SOURCE ; }
examples-dir(){ echo $(dirname $BASH_SOURCE) ; } 
examples-cd(){  cd $(examples-dir) ; }
examples-env(){ echo -n ; }

examples-use(){
   examples-cd
   grep ^Use README.rst  
}


examples-copy-goc(){

   local tdir=UseG4OKNoCMake
   ls -1d Use*NoCMake | while read dir ; do 
      echo $dir 
      if [ "$tdir" != "$dir" ]; then 
         cp $tdir/goc.sh $dir 
      fi  
   done 


}


examples-copy-gob(){

   # hmm many of the CMake ones do not use standard naming, so not so easy to gob them 

   local tdir=UseG4OK
   ls -1d Use* | grep -v NoCMake | grep -v OptiX7 | while read dir ; do 
      echo $dir 
      if [ "$tdir" != "$dir" ]; then 
         cp $tdir/gob.sh $dir 
      fi  
   done 


}

