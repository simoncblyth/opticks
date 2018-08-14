ckm-source(){ echo $BASH_SOURCE ; }
ckm-vi(){ vi $(ckm-source) ; }
ckm-env(){ echo -n ; }
ckm-usage(){ cat << EOU

EOU
}
ckm-dir(){ echo $(dirname $(ckm-source)) ; }
ckm-cd(){  cd $(ckm-dir) ; }

ckm--(){ ckm-cd ; ./go.sh ; }



ckm-key(){ echo CerenkovMinimal.X4PhysicalVolume.World.44ca65ec36cb6c03f465bc38ac5c5dd4 ; }
ckm-idpath(){ echo $LOCAL_BASE/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/44ca65ec36cb6c03f465bc38ac5c5dd4/1 ; }
ckm-evpath(){ echo /tmp/blyth/opticks/evt/g4live/natural/1 ; }

ckm-load()
{
    OPTICKS_KEY=$(ckm-key) lldb -- OKTest --load --natural --envkey
    type $FUNCNAME
}
ckm-dump()
{
    OPTICKS_KEY=$(ckm-key) OpticksEventDumpTest --natural --envkey
    type $FUNCNAME
}

