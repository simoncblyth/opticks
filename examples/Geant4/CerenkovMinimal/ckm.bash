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
    g4-export        # internal envvar stuff is not done here 
    CerenkovMinimal  # NB the Opticks is embedded via G4OK : so commandline doesnt get thru 
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
     option makes executables sensitive to the OPTICKS_KEY envvar allowing 
     booting from the corresponding geocache 

EON
}


ckm-load(){      OPTICKS_KEY=$(ckm-key) lldb -- OKTest --load --natural --envkey ;}
ckm-dump(){      OPTICKS_KEY=$(ckm-key) OpticksEventDumpTest --natural --envkey  ;}
ckm-res(){       OPTICKS_KEY=$(ckm-key) lldb -- OpticksResourceTest --natural --envkey ;}
ckm-okg4(){      OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --compute --envkey --embedded --save ;}
ckm-okg4-load(){ OPTICKS_KEY=$(ckm-key) lldb -- OKG4Test --load --envkey --embedded ;}
ckm-mlib(){      OPTICKS_KEY=$(ckm-key) CMaterialLibTest --envkey  ;}
ckm-gentest(){   OPTICKS_KEY=$(ckm-key) lldb -- CCerenkovGeneratorTest --natural --envkey ;}
ckm-okt(){       OPTICKS_KEY=$(ckm-key) lldb -- OpticksTest --natural --envkey ;}

ckm-addr2line()
{
    local addr=${1:-0x10002160e}
    PATH=/usr/bin lldb $(which CerenkovMinimal) -o "source list -a $addr"  --batch
}


ckm-genrun(){
    local iwd=$PWD
    ckm-kcd

    local func=$1
    mkdir -p $(ckm-tmp)
    local py=$(ckm-tmp)/$func.py
    $func- $* > $py 
    cat $py 
    ipython -i $py 

    cd $iwd
}

ckm-tag(){ echo 1 ; }
ckm-tmp(){   echo $TMP/ckm ; }

#############################################################


#ckm-a-dir(){   echo source/evt/g4live/natural/1 ; }  # ckm--
#ckm-a-name(){  echo ox.npy ; }         

ckm-a-dir(){   echo source/evt/g4live/natural/-1 ; }  # ckm--
ckm-a-name(){  echo so.npy ; }         


#############################################################

#ckm-b-dir(){   echo tests/CCerenkovGeneratorTest ; }  # ckm-gentest
#ckm-b-name(){  echo so.npy ; }

ckm-b-dir(){   echo source/evt/g4live/natural/1 ; }  # ckm--
ckm-b-name(){  echo ox.npy ; }

#############################################################

ckm-a(){ cd $(ckm-a-dir) ; } 
ckm-b(){ cd $(ckm-b-dir) ; } 

ckm-l(){
    local iwd=$PWD
    ckm-kcd 
    date

    echo A $(ls -l $(ckm-a-dir)/*.json)
    echo B $(ls -l $(ckm-b-dir)/*.json)

    cd $iwd
}

ckm-ls(){ 
    local iwd=$PWD
    ckm-kcd 
    date

    echo A $(ckm-a-dir)
    ls -l  $(ckm-a-dir) 
    np.py  $(ckm-a-dir)

    echo B $(ckm-b-dir)
    ls -l  $(ckm-b-dir) 
    np.py  $(ckm-b-dir)

    cd $iwd
}


ckm-xx-(){ 

    local apath=$(ckm-a-dir)/$1
    local bpath=$(ckm-b-dir)/$2

    echo \# $apath $bpath 

    cat << EOP
import numpy as np, commands

apath = "$apath"
bpath = "$bpath"

print " $FUNCNAME comparing $1 and $2 between two dirs " 

print "pwd", commands.getoutput("pwd")
print "  ", commands.getoutput("date")
print "a ", commands.getoutput("ls -l %s" % apath)
print "b ", commands.getoutput("ls -l %s" % bpath)

a = np.load(apath)
b = np.load(bpath)

print "a %s " % repr(a.shape)
print "b %s " % repr(b.shape)

dv = np.max( np.abs(a[:,:3]-b[:,:3]), axis=(1,2) )

print "max deviation %s " % dv.max() 

cuts = [1e-5, 1e-6, 1e-7, 1e-8]
for cut in cuts:
    wh = np.where( dv > cut )[0] 
    print " deviations above cut %s num_wh %d" % ( cut, len(wh) )
    for i in wh[:10]:
        print i, dv[i], "\n",np.hstack([a[i,:3],(a[i,:3]-b[i,:3])/cut,b[i,:3]])
    pass
pass


EOP
}

ckm-so-(){ ckm-xx-    $(ckm-a-name) $(ckm-b-name)  ; }
ckm-so(){  ckm-genrun $FUNCNAME ; }

ckm-so-notes(){ cat << EON

ckm-so-notes
================

Comparing Cerenkov generated photons between:

ckm-- 
    CerenkovMinimal : geant4 example app, with genstep and photon collection
    via embedded Opticks with embedded commandline 
    " --gltf 3 --compute --save --embedded --natural --dbgtex --printenabled --pindex 0 --bouncemax 0"  

    --bouncemax 0 
        means that photons are saved immediately after generation, with no propagation 

        * which means that GPU ox.npy can be matched with CPU so.npy 
        * TODO: perhaps formalize that, by copying ox.npy to so.npy when --bouncemax 0 ??
          but then what when switch bouncemax back to normal
   
    --printenabled --pindex 0
        dump kernel debug for photon 0 


    The big advantage of ckm-- is that it can look like any Geant4 example, that however 
    is also its biggest disadvantage in that this restricts it to minimally instrumented G4 
    as it does not make use of the CFG4/CRecorder : for full photon step recording : which 
    is the reason source/evt/g4live/natural/-1/ is rather empty compared
    to the fully featured source/evt/g4live/natural/1/



ckm-gentest : 
    CCerenkovGeneratorTest : genstep eating standalone CPU generator that tries to
    mimic the cerenkov process photons via verbatim code copy 


cross exe, g4-g4 "same" sim : but with float genstep transport
----------------------------------------------------------------

Comparing photons from genstep 0, 
   source/evt/g4live/natural/-1/so.npy 
   tests/CCerenkovGeneratorTest/so.npy

* initially : small deviations at 1e-5 level mostly in wavelength
* fixed precision loss issue with wavelength and omission with time :
  bringing deviations down to an unfocussed 1e-8 level 


same ckm exe, cross sim G4/OK : with --bouncemax 0 for generation only comparison
------------------------------------------------------------------------------------------

Comparing photons from genstep 0, 
    source/evt/g4live/natural/-1/so.npy 
    source/evt/g4live/natural/1/ox.npy  

* initially same level of 1e-5 level deviations, mostly in wavelength 



EON
}

