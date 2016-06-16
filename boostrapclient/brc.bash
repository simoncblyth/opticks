brc-src(){      echo boostrapclient/brc.bash ; }
brc-source(){   echo ${BASH_SOURCE:-$(env-home)/$(brc-src)} ; }
brc-vi(){       vi $(brc-source) ; }
brc-env(){      elocal- ; }
brc-usage(){ cat << EOU


BoostRap Client Testing
==========================





EOU
}
brc-dir(){ echo $(env-home)/boostrapclient ; }
brc-cd(){  cd $(brc-dir); }


brc-run()
{
   PATH=$(opticks-prefix)/lib:$(opticks-prefix)/externals/lib:"$PATH" /c/usr/local/opticks/build/boostrapclient/Debug/BoostRapClient.exe
}





