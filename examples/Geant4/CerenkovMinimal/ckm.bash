ckm-source(){ echo $BASH_SOURCE ; }
ckm-vi(){ vi $(ckm-source) ; }
ckm-env(){ echo -n ; }
ckm-usage(){ cat << EOU

EOU
}
ckm-dir(){ echo $(dirname $(ckm-source)) ; }
ckm-cd(){  cd $(ckm-dir) ; }

ckm--(){ ckm-cd ; ./go.sh ; }



#ckm-dig(){ echo 44ca65ec36cb6c03f465bc38ac5c5dd4 ; }
ckm-dig(){ echo 960713d973bd4be73b1b7d9aa4838c3e ; }

ckm-key(){ echo CerenkovMinimal.X4PhysicalVolume.World.$(ckm-dig) ; }
ckm-idpath(){ echo $LOCAL_BASE/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/$(ckm-dig)/1 ; }

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


ckm-cfg4()
{
    OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --compute --envkey
}




