geocache-source(){ echo $BASH_SOURCE ; }
geocache-vi(){ vi $(geocache-source) ; }
geocache-sdir(){ echo $(dirname $(geocache-source)) ; }
geocache-scd(){  cd $(geocache-dir) ; }
geocache-usage(){ cat << EOU


EOU
}

geocache-env(){ echo -n ; }
geocache-export()
{
   export IDPATH2=/usr/local/opticks-cmake-overhaul/geocache/CX4GDMLTest_World0xc15cfc0_PV_g4live/g4ok_gltf/828722902b5e94dab05ac248329ffebe/1
}

geocache-paths(){  echo $IDPATH/$1 $IDPATH2/$1 ; }
geocache-diff-(){  printf "\n======== $1 \n\n" ; diff -y $(geocache-paths $1) ; }
geocache-diff()
{
   geocache-export 
   geocache-diff- GItemList/GMaterialLib.txt 
   geocache-diff- GItemList/GSurfaceLib.txt 
   geocache-diff- GNodeLib/PVNames.txt
   geocache-diff- GNodeLib/LVNames.txt
}


geocache-paths-pv(){ geocache-paths GNodeLib/PVNames.txt ; }
geocache-diff-pv(){ geocache-diff- GNodeLib/PVNames.txt ; }
geocache-diff-lv(){ geocache-diff- GNodeLib/LVNames.txt ; }

geocache-info(){  cat << EOI

   IDPATH  : $IDPATH
   IDPATH2 : $IDPATH2

EOI
}

geocache-py()
{
   geocache-scd
   ipython -i geocache.py 
}



geocache-info(){ cat << EOI

  OPTICKS_KEY     :  ${OPTICKS_KEY}
  geocache-keydir : $(geocache-keydir)

EOI
}

geocache-keydir()
{
    local k=$OPTICKS_KEY
    local arr=(${k//./ })
    [ "${#arr[@]}" != "4" ] && echo $msg expecting OPTICKS_KEY envvar with four fields separate by dot && sleep 100000
    local exe=${arr[0]}
    local cls=${arr[1]}
    local top=${arr[2]}
    local dig=${arr[3]}
    echo $LOCAL_BASE/opticks/geocache/${exe}_${top}_g4live/g4ok_gltf/$dig/1 
    #echo $LOCAL_BASE/opticks/geocache/OKX4Test_lWorld0x4bc2710_PV_g4live/g4ok_gltf/528f4cefdac670fffe846377973af10a/1
}

geocache-kcd(){ cd $(geocache-keydir) ; }


