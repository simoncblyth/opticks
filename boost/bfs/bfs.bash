# === func-gen- : boost/bfs/bfs fgp boost/bfs/bfs.bash fgn bfs fgh boost/bfs
bfs-src(){      echo boost/bfs/bfs.bash ; }
bfs-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(bfs-src)} ; }
bfs-vi(){       vi $(bfs-source) ; }
bfs-env(){      elocal- ; }
bfs-usage(){ cat << EOU

* http://www.boost.org/doc/libs/1_57_0/libs/filesystem/doc/index.htm



EOU
}

bfs-bdir(){ echo $(local-base)/env/boost/bfs.build ; }
bfs-sdir(){ echo $(opticks-home)/boost/bfs ; }
bfs-cd(){   cd $(bfs-sdir); }
bfs-bcd(){  cd $(bfs-bdir); }

bfs-inc(){ echo /opt/local/include ; }
bfs-lib(){ echo /opt/local/lib ; }


bfs-make(){
  local cpp=${1:-tut1.cpp}
  local nam=${cpp/.cpp}  

  local bdir=$(bfs-bdir)
  mkdir -p $bdir

  local bin=$(bfs-bdir)/$nam 
  local cmd="clang++ -I$(bfs-inc) -L$(bfs-lib) -lboost_filesystem-mt -lboost_system-mt $(bfs-sdir)/$cpp -o $bin"
  echo $cmd
  eval $cmd

  echo $bin
} 


   


