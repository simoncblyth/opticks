scan-source(){ echo $BASH_SOURCE ; }
scan-vi(){ vi $(scan-source)  ; }
scan-env(){  olocal- ; opticks- ; }
scan-usage(){ cat << EOU

scan
===================



EOU
}


scan-photons(){ cat << EOS | tr -d " ,"  | grep -v \#
#         1
#      1000
#    10,000
#   100,000
#   200,000
#   500,000
# 1,000,000
  2,000,000
  3,000,000
EOS
}

scan-cmd(){ printf "tboolean.sh box --generateoverride %s --error\n" $1 ; }
scan-post(){  scan.py /tmp/tboolean-box ; }

scan-cmds(){
   local cmd
   local photons
   scan-photons | while read photons 
   do 
      scan-cmd $photons
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

