#!/usr/bin/env python
import logging, os
log = logging.getLogger(__name__)


template_head = r"""

join(){ local IFS="$1"; shift; echo "$*"; }

tbool%(name)s-vi(){   vi $BASH_SOURCE ; }
tbool%(name)s-env(){  olocal- ;  } 
tbool%(name)s-tag(){  echo 1 ; }
tbool%(name)s-det(){  echo tbool ; }
tbool%(name)s-src(){  echo torch ; }
tbool%(name)s-args(){ echo  --det $(tbool%(name)s-det) --src $(tbool%(name)s-src) ; }

tbool%(name)s-torchconfig()
{
    local identity=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000

    #local photons=1000000
    local photons=100000
    #local photons=1

    local torch_config_sphere=(
                 type=sphere
                 photons=10000
                 frame=-1
                 transform=1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,0.000,1.000,0.000,0.000,0.000,1000.000,1.000
                 source=0,0,0
                 target=0,0,1
                 time=0.1
                 radius=100
                 distance=400
                 zenithazimuth=0,1,0,1
                 material=GdDopedLS
                 wavelength=500
               )

    echo "$(join _ ${torch_config_sphere[@]})" 
}


tbool%(name)s-op()
{
    local msg="=== $FUNCNAME :"
    local cmdline=$*
    local testconfig=$TESTCONFIG

    op.sh  \
            $cmdline \
            --animtimemax 20 \
            --timemax 20 \
            --geocenter \
            --eye 1,0,0 \
            --dbganalytic \
            --test --testconfig "$testconfig" \
            --torch --torchconfig "$(tbool%(name)s-torchconfig)" \
            --torchdbg \
            --tag $(tbool%(name)s-tag) --cat $(tbool%(name)s-det) \
            --save 
}

tbool%(name)s(){ TESTCONFIG=$($FUNCNAME-) tbool%(name)s-op $* ; }
tbool%(name)s-(){ $FUNCNAME- | python $* ; } 
tbool%(name)s--(){ cat << EOP

import logging
log = logging.getLogger(__name__)
from opticks.ana.base import opticks_main
from opticks.analytic.csg import CSG  
args = opticks_main(csgpath="$TMP/tbool/%(name)s")

CSG.boundary = args.testobject
CSG.kwa = dict(verbosity="1", poly="IM")

"""

template_tail = r"""

obj = %(root)s

con = CSG("sphere",  param=[0,0,0,10], container="1", containerscale="2", boundary=args.container , poly="HY", level="5" )
CSG.Serialize([con, obj], args.csgpath )

EOP
}

"""

template_body = r"""

%(body)s

"""


test_body = r"""

a = CSG("sphere", param=[0,0,0,500] )


"""



class TBooleanBashFunction(dict):
     def __init__(self, *args, **kwa):
         dict.__init__(self, *args, **kwa)

     head = property(lambda self:template_head % self)
     body = property(lambda self:template_body % self)
     tail = property(lambda self:template_tail % self)

     path = property(lambda self:os.path.expandvars("$TMP/tbool%(name)s.bash" % self))

     def save(self):
         log.info("saving to %s " % self.path) 
         file(self.path,"w").write(str(self))

     def test(self):
         print self.head
         print self.body
         print self.tail

     def __str__(self):
         return "\n".join([self.head, self.body, self.tail])  


if __name__ == '__main__':

     logging.basicConfig(level=logging.INFO)

     tbf = TBooleanBashFunction(name="0", root="a", body=test_body)
     #tbf.test()

     print tbf 

     tbf.save()

     



