hh-source(){ echo $BASH_SOURCE ; }
hh-vi(){ vi $(hh-source)  ; }
hh-env(){  olocal- ; opticks- ; }
hh-usage(){ cat << EOU

hh : finding headers with embedded RST documentation
============================================================


hh.py 
     finds headers lacking a docstring 

hh--



EOU
}

hh-tmp(){   echo  /tmp/$USER/opticks/hh ; }
hh-find(){  find . -name '*.hh' -exec grep -l "/\*\*" {} \; ; }

hh--(){
   local hh
   hh-find | while read hh ; do 
      local l=$(cat $hh | hh.py | wc -l)

      if [ $l -gt 30 ]; then 
          printf "%40s : %d \n" $hh $l
      fi
   done
}


