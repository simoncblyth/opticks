#!/usr/bin/env python

import numpy as np
import os, re, logging
log = logging.getLogger(__name__)
from collections import OrderedDict as odict 
from opticks.sysrap.stag import stag 
tag = stag()

class U4Stack_item(object):
    @classmethod
    def Placeholder(cls):
        return cls(-1,"placeholder","ERROR" )

    def __init__(self, code, name, note=""):
        self.code = code 
        self.name = name 
        self.note = note 

    def __repr__(self):
        return "%2d : %10s : %s " % (self.code, self.name, self.note)

class U4Stack(object):
    PATH = "$OPTICKS_PREFIX/include/u4/U4Stack.h" 
    enum_ptn = re.compile("^\s*(\w+)\s*=\s*(.*?),*\s*?$")
    ttos_ptn = re.compile("^\s*case stag_(\w+):\s*stack = U4Stack_(\w+)\s*.*$")

    def __init__(self, path=PATH):
        path = os.path.expandvars(path)
        lines = open(path, "r").read().splitlines()
        self.path = path
        self.lines = lines
        self.items = []
        self.parse()

    def parse(self):
        self.code2item = odict()
        self.name2item = odict()
        self.tag2stack = odict()
        self.stack2tag = odict()
 
        for line in self.lines:
            enum_match = self.enum_ptn.match(line)
            ttos_match = self.ttos_ptn.match(line)
            if enum_match:
                name, val = enum_match.groups()
                pfx = "U4Stack_"
                assert name.startswith(pfx)
                sname = name[len(pfx):]
                code = int(val)

                item = U4Stack_item(code, sname, "")
                self.items.append(item)
                self.code2item[code] = item
                self.name2item[sname] = item
                log.debug(" name %20s sname %10s val %5s code %2d " % (name, sname, val, code))     

            elif ttos_match:
                tag_name, stack_name = ttos_match.groups()

                stack_item = self.name2item.get(stack_name, None)
                tag_item = tag.name2item.get(tag_name, None)

                stack_code = stack_item.code if not stack_item is None else -1
                tag_code = tag_item.code if not tag_item is None else -1

                self.tag2stack[tag_code] = stack_code
                self.stack2tag[stack_code] = tag_code

                log.info(" tag_name %15s stack_name %50s stack_item %r tag_item %r" % (tag_name, stack_name, stack_item, tag_item ))  
            else:
                pass
                log.debug(" skip :  %s " % line )
            pass 
        pass

    def label(self, st, fl=None):
        """
        In [8]: print(stack.label(bt[53])) 
         0 :  2 : ScintDiscreteReset :  
         1 :  6 : BoundaryDiscreteReset :  
         2 :  4 : RayleighDiscreteReset :  
         3 :  3 : AbsorptionDiscreteReset :  
         4 :  8 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
         5 :  7 : BoundaryDiDiTransCoeff :  

         6 :  2 : ScintDiscreteReset :  
         7 :  6 : BoundaryDiscreteReset :  
         8 :  4 : RayleighDiscreteReset :  
         9 :  3 : AbsorptionDiscreteReset :  
        10 :  8 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
        11 :  7 : BoundaryDiDiTransCoeff :  

        12 :  2 : ScintDiscreteReset :  
        13 :  6 : BoundaryDiscreteReset :  
        14 :  4 : RayleighDiscreteReset :  
        15 :  3 : AbsorptionDiscreteReset :  

        16 :  2 : ScintDiscreteReset :  
        17 :  6 : BoundaryDiscreteReset :  
        18 :  8 : BoundaryBurn_SurfaceReflectTransmitAbsorb :  
        19 :  7 : BoundaryDiDiTransCoeff :  
        ## HUH: ONLY 2 RESET NOT NORMAL 4 ? WHY ? 
        ## COULD BE PRECEEDING ZERO STEP OR SMTH LIKE THAT 

        20 :  2 : ScintDiscreteReset :  
        21 :  6 : BoundaryDiscreteReset :  
        22 :  4 : RayleighDiscreteReset :  
        23 :  3 : AbsorptionDiscreteReset :  
        """
        if not fl is None:
            assert st.shape == fl.shape
        pass
        lines = [] 
        num_zero = 0 
        for i in range(len(st)):
            code = st[i]
            flat = fl[i] if not fl is None else None
            item = self(code)
            it = repr(item)
            assert code == item.code
            if code == st[0] and i > 0:
                lines.append("")   
            pass
            label = "%2d : %s " % (i, it) if fl is None else "%2d : %10.4f : %s" % (i, flat, it) 
            lines.append(label)
            if item.code == 0:
                num_zero += 1 
            pass
            if num_zero == 2: break  
        pass
        return "\n".join(lines)

    def __call__(self, code):
        return self.code2item.get(code,U4Stack_item.Placeholder())
 
    def __str__(self):
        return "\n".join(self.lines)

    def __repr__(self):
        return "\n".join(list(map(repr,self.items)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    stack = U4Stack()
    #print(stack) 
    print(repr(stack))
    
    st = np.array([[2, 6, 4, 3, 8, 7, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2, 6, 4, 3, 8, 7, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                   [2, 6, 4, 3, 8, 7, 2, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

    print(stack.label(st[0,:10]))
    



