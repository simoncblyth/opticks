abe-source(){ echo $BASH_SOURCE ; }
abe-vi(){ vi $(abe-source)  ; }
abe-env(){  olocal- ; opticks- ; }
abe-usage(){ cat << EOU

abe Usage 
===================

See the similar ev- and grab things as needed

EOU
}

abe-info(){ cat << EOP

   abe-a-dir : $(abe-a-dir)
   abe-b-dir : $(abe-b-dir)

EOP
}

abe-genrun(){
    local func=$1
    mkdir -p $(abe-tmp)
    local py=$(abe-tmp)/$func.py
    $func- $* > $py
    cat $py
    ipython -i $py
}

abe-tag(){ echo 1 ; }

abe-tmp(){   echo $TMP/abe ; }
abe-a-dir(){ echo $TMP/evt/g4live/natural ; }  # ckm--
abe-b-dir(){ echo $TMP/evt/g4live/natural ; }  # ckm--
#abe-b-dir(){ echo $TMP/evt/g4live/torch   ; }  # ckm-okg4 

abe-a(){ cd $(abe-a-dir) ; }
abe-b(){ cd $(abe-b-dir) ; }

abe-l(){ 
   date
   echo A $(ls -l $(abe-a)/*.json)
   echo B $(ls -l $(abe-b)/*.json)
}

abe-ls(){ 
   date
   echo A $(abe-a-dir)
   ls -l  $(abe-a-dir) 
   echo B $(abe-b-dir)
   ls -l  $(abe-b-dir) 
}

abe-xx-(){ cat << EOP
import numpy as np, commands

apath = "$(abe-a-dir)/$1"
bpath = "$(abe-b-dir)/$2"

print " $FUNCNAME comparing $1 and $2 between two dirs " 

print "  ", commands.getoutput("date")
print "a ", commands.getoutput("ls -l %s" % apath)
print "b ", commands.getoutput("ls -l %s" % bpath)

a = np.load(apath)
b = np.load(bpath)

print "a %s " % repr(a.shape)
print "b %s " % repr(b.shape)

EOP
}

abe-gs--(){ cat << EOP


print "\n\na0/b0 : id/parentId/materialId/numPhotons \n " 
a0 = "a[:,0].view(np.int32)"
b0 = "b[:,0].view(np.int32)"

print a0, "\n"
print eval(a0), "\n"

print b0, "\n"
print eval(b0), "\n"


print "\n\na1/b1 : start position and time x0xyz, t0 \n" 
a1 = "a[:,1]"
b1 = "b[:,1]"

print a1, "\n"
print eval(a1), "\n"

print b1, "\n"
print eval(b1), "\n"


print "\n\na2/b2 : deltaPosition, stepLength \n" 
a2 = "a[:,2]"
b2 = "b[:,2]"

print a2, "\n"
print eval(a2), "\n"

print b2, "\n"
print eval(b2), "\n"


EOP
}


abe-ht4-(){ abe-xx- -1/ht.npy  -1/ht.npy  ; }
abe-ht-(){ abe-xx-   1/ht.npy  1/ht.npy  ; }
abe-so-(){ abe-xx-   -1/so.npy  1/ox.npy  ; }
abe-gs-(){ 
   abe-xx- 1/gs.npy 1/gs.npy 
   abe-gs-- 
}

abe-ht4(){ abe-genrun $FUNCNAME ; }
abe-ht(){ abe-genrun $FUNCNAME ; }
abe-gs(){ abe-genrun $FUNCNAME ; }
abe-so(){ abe-genrun $FUNCNAME ; }

abe-np(){
   local iwd=$PWD

   abe-a
   echo A 
   np.py

   abe-b
   echo B
   np.py 

   cd $iwd
}



abe-prim(){ abe-genrun $FUNCNAME ; }
abe-prim-(){ ckm- ; cat << EOP

import commands, numpy as np  
ppath = "$(ckm-idpath)/primaries.npy" 
print "  ", commands.getoutput("date")
print "p ", commands.getoutput("ls -l %s" % ppath)

p = np.load(ppath)

print "p %s " % repr(p.shape)

EOP
}




