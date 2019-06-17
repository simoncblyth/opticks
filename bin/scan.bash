scan-source(){ echo $BASH_SOURCE ; }
scan-vi(){ vi $(scan-source)  ; }
scan-env(){  olocal- ; opticks- ; }
scan-usage(){ cat << EOU

scan
===================



EOU
}



#scan-mode(){ echo photons ; }
scan-mode(){ echo ${SCAN_MODE:-proxy} ; }

#scan-proxy-args(){   seq 0 39 ; }
scan-proxy-args(){   seq 0 39 ; }
scan-photons-args(){ cat << EOS | tr -d " ,"  | grep -v \#
          1
       1000
     10,000
#    50,000
    100,000
    200,000
    500,000
  1,000,000
  2,000,000
  3,000,000
EOS
}
scan-photons-cmd(){ printf "tboolean.sh box --generateoverride %s --error --cvd 1 --rtx 1\n" $1 ; }
scan-proxy-cmd(){   printf "PROXYLV=%s tboolean.sh\n" $1 ; }

scan-photons-post(){  scan.py /tmp/tboolean-box ; }
scan-proxy-post(){    echo scan.py ???????? ; }


scan-cmds(){
   local mode=$(scan-mode)
   local cmd
   local arg
   scan-$mode-args | while read arg
   do 
      scan-$mode-cmd $arg
   done
}

scan--()
{
   local cmd
   scan-cmds | while read cmd
   do
      echo $cmd
      eval $cmd  
   done
}

