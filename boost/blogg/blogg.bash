# === func-gen- : boost/blogg/blogg fgp boost/blogg/blogg.bash fgn blogg fgh boost/blogg
blogg-src(){      echo boost/blogg/blogg.bash ; }
blogg-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(blogg-src)} ; }
blogg-vi(){       vi $(blogg-source) ; }
blogg-env(){      olocal- ; }
blogg-usage(){ cat << EOU




EOU
}
blogg-dir(){ echo $(opticks-home)/boost/blogg ; }
blogg-cd(){  cd $(blogg-dir); }

blogg-triv(){
   #clang++ -I/opt/local/include -L/opt/local/lib -DBOOST_LOG_DYN_LINK -lboost_system-mt -lboost_log-mt -lboost_log_setup-mt $(blogg-dir)/triv.cc -o /tmp/triv
   clang++ -I/opt/local/include -L/opt/local/lib -DBOOST_LOG_DYN_LINK -lboost_system-mt -lboost_thread-mt -lboost_log-mt -lboost_log_setup-mt  $(blogg-dir)/triv.cc -o /tmp/triv
}


