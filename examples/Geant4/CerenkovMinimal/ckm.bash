ckm-source(){ echo $BASH_SOURCE ; }
ckm-vi(){ vi $(ckm-source) ; }
ckm-env(){ echo -n ; }
ckm-usage(){ cat << EOU

EOU
}
ckm-dir(){ echo $(dirname $(ckm-source)) ; }
ckm-cd(){  cd $(ckm-dir) ; }
ckm-c(){  cd $(ckm-dir) ; }

ckm--(){ ckm-cd ; ./go.sh ; }



ckm-dig(){ echo c250d41454fba7cb19f3b83815b132c2 ; }

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


ckm-okg4()
{
    OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --compute --envkey --embedded --save
}
ckm-okg4-load()
{
    OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --load --envkey --embedded
}




