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

ckm-run()
{
    g4-
    g4-export   # internal envvar stuff, is it done here ? NOPE
    # CerenkovMinimal --dbgtex    ## cannot do this either, as opticks is embedded
}



#ckm-dig(){ echo c250d41454fba7cb19f3b83815b132c2 ; }
ckm-dig(){ echo 792496b5e2cc08bdf5258cc12e63de9f ; }

ckm-key(){ echo CerenkovMinimal.X4PhysicalVolume.World.$(ckm-dig) ; }
ckm-idpath(){ echo $LOCAL_BASE/opticks/geocache/CerenkovMinimal_World_g4live/g4ok_gltf/$(ckm-dig)/1 ; }

ckm-export(){  export OPTICKS_KEYDIR=$(ckm-idpath) ; }
ckm-kcd(){ cd $(ckm-idpath) ; }
ckm-ls(){  ls -l $(ckm-idpath) ; }

ckm-evpath0(){ echo $TMP/evt/g4live/natural ; }
ckm-evpath1(){ echo $TMP/evt/g4live/torch ; }
ckm-ecd0(){ cd $(ckm-evpath0) ; }
ckm-ecd1(){ cd $(ckm-evpath1) ; }

ckm-info(){ cat << EOI

    ckm-dig                : $(ckm-dig) 
    ckm-key                : $(ckm-key)

    ckm-idpath, ckm-kcd    : $(ckm-idpath)

    ckm-evpath0, ckm-ecd0  : $(ckm-evpath0)   
    ckm-evpath1, ckm-ecd1  : $(ckm-evpath1)   
 
    ckm-ls : 
EOI
    ckm-ls
}



ckm-notes(){ cat << EON

--envkey 
     option makes executables senitive to the OPTICKS_KEY envvar allowing 
     booting from the corresponding geocache 

EON
}


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
    #OPTICKS_KEY=$(ckm-key) OKG4Test --compute --envkey --embedded --save
    OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --compute --envkey --embedded --save
}
ckm-okg4-load()
{
    OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --load --envkey --embedded
}



ckm-addr2line()
{
    local addr=${1:-0x10002160e}
    PATH=/usr/bin lldb $(which CerenkovMinimal) -o "source list -a $addr"  --batch
}

