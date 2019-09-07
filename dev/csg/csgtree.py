#!/usr/bin/env python
#
# Copyright (c) 2019 Opticks Team. All Rights Reserved.
#
# This file is part of Opticks
# (see https://bitbucket.org/simoncblyth/opticks).
#
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software 
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and 
# limitations under the License.
#

"""

* :google:`Spatially Efficient Tree Layout for GPU Ray-tracing of Constructive Solid Geometry Scenes`


PCT 2016
Parallel Computing Technologies
Proceedings of the 10th Annual International Scientific Conference on Parallel Computing Technologies
Arkhangelsk, Russia, March 29-31, 2016.

* http://ceur-ws.org/Vol-1576/  
* http://ceur-ws.org/Vol-1576/090.pdf

D.Y. Ulyanov(1,2)
D.K. Bogolepov(2)
V.E. Turlapov(1) 

(1) University of Nizhniy Novgorod
(2) OpenCASCADE  https://www.opencascade.com



* :google:`Ulyanov Bogolepov Turlapov`


* https://otik.uk.zcu.cz/bitstream/handle/11025/10619/Bogolepov.pdf?sequence=1

* denisbogol@gmail.com      Denis Bogolepov
* danila-ulyanov@ya.ru      Danila Ulyanov
* vadim.turlapov@gmail.com  Vadim Turlapov

* https://github.com/megaton?tab=repositories

* https://github.com/megaton/csg-tools
* https://github.com/megaton/csg-tools/blob/master/src/csgviewer.cpp
* https://github.com/megaton/csg-format/blob/master/CSG-format.md


Author 
Danila Ya. Ulyanov 

Journal of instrument engineering
pribor.ifmo.ru/en/person/5983/ulyanov_danila...

Denis K. Bogolepov, 
Dmitry P. Sopin, 
Danila Ya. Ulyanov, 
Vadim E. Turlapov 
      CONSTRUCTION OF SAH BVH TREES FOR RAY TRACING WITH THE USE OF GRAPHIC PROCESSORS




Other ray trace of CSG tree implementations

* https://github.com/POV-Ray/povray/search?q=CSG

* https://cadcammodelling.wordpress.com/2011/01/23/13-steps-to-perform-csg-tree-raycasting/


* https://github.com/search?p=5&q=CSG&ref=searchresults&type=Repositories


* :google:`ray trace csg tree`

* https://www.clear.rice.edu/comp360/lectures/old/Solidstext.pdf


"""

import logging
import numpy as np
import matplotlib.pyplot as plt

from intersect import intersect_primitive, Node, Ray, UNION, INTERSECTION, DIFFERENCE, BOX, SPHERE, EMPTY, desc

from boolean import boolean_table 
from boolean import desc_state, Enter, Exit, Miss
from boolean import desc_acts, RetMiss, RetL, RetR, RetLIfCloser, RetRIfCloser, LoopL, LoopLIfCloser, LoopR, LoopRIfCloser, FlipR

# actions
GotoLft = 0x1 << 1 
GotoRgh = 0x1 << 2
LoadLft = 0x1 << 3 
LoadRgh = 0x1 << 4
Compute = 0x1 << 5  
SaveLft = 0x1 << 6 
Start   = 0x1 << 7
Return  = 0x1 << 8

def desc_action(action):
    if action is None:return "NONE"
    s = ""
    if action & GotoLft:s+="GotoLft " ; 
    if action & GotoRgh:s+="GotoRgh " ; 
    if action & LoadLft:s+="LoadLft " ; 
    if action & LoadRgh:s+="LoadRgh " ; 
    if action & Compute:s+="Compute " ; 
    if action & SaveLft:s+="SaveLft " ; 
    if action & Start:s+="Start " ; 
    if action & Return:s+="Return " ; 
    return s 

def intersectBox(node):
    return True        # skipping bbox optimization, just test against the primitive


def dump(label):
    return "%s %s tl:%s nl:%s tr:%s nr:%s " % (label, desc_action(action),repr(tl),repr(nl),repr(tr),repr(nr))


class Stack(object):
    def __init__(self, name, desc_ ):
        self.name = name
        self.desc_ = desc_
        self._stack = []

    def reset(self):
        self._stack = []

    def push(self, obj, debug=False):
        if debug and self.desc_ is not None:
            log.info("%s.push %s " % (self.name, self.desc_(obj)))
        pass
        self._stack.append(obj)

    def pop(self, debug=False):
        if len(self._stack) == 0:
            assert 0 
        obj = self._stack.pop()
        if debug and self.desc_ is not None:
            log.info("%s.pop -> %s  rest: %s  " % (self.name, self.desc_(obj), self.desc() ))

        return obj

    def count(self):
        return len(self._stack)
 
    def desc(self):
        return ",".join(map(self.desc_,self._stack))


class CSG(object):
    def __init__(self, level=1):
        self.level = level
        self.actionStack = Stack("actionStack", desc_action)
        self.tminStack = Stack("tminStack", lambda tmin:"%5.2f" % (tmin if tmin else -1))
        self.primStack = Stack("primStack", lambda _:"%5.2f:%s" % (_[0] if _[0] else -1, _[2]))
        self.typ = None
        self._traverse = {"RECURSIVE":[], "ITERATIVE":[] }
        self._actions = {"RECURSIVE":[], "ITERATIVE":[] }
        self.reset()

    def _get_debug(self):
         return self.level if self.iray in self._debug else 0
    debug = property(_get_debug)


    def reset(self, ray=None, iray=0, debug=[]):

        self.ray = ray 
        self.iray = iray
        self.count = 0 

        self.lname = None
        self.rname = None

        self._debug = debug

        self._node = None
        self._tl = None
        self._tr = None
        self._nr = None
        self._nl = None
        self._tmin = 0
        self._prev = None
        self._stage = None
        self._act = None

        if self.typ in self._traverse:
            self._traverse[self.typ] = []
            self._actions[self.typ] = []

        self.actionStack.reset()
        self.tminStack.reset()
        self.primStack.reset()

    def __repr__(self):
        return "[%d] %s %s : %s : %s -> %s " % (self.count, self.typ, self.actionStack.desc(), self.stage, self.prevname, self.nodename )

    prevname = property(lambda self:self.prev.name if self.prev else "-")
    nodename = property(lambda self:self.node.name if self.node else "-")

    def _get_act(self):
        return self._act
    def _set_act(self, act):
        self._act = act
    act = property(_get_act, _set_act)

    def _get_stage(self):
        prev = self._prev
        node = self._node
        if (prev is None):
            stage = "going down tree from prev None"
        elif (node is prev.left):
            stage = "going left down tree"
        elif (node is prev.right):
            stage = "going right down tree"
        elif node is not None and prev is node.left:
            stage = "up from left child"
        elif node is not None and prev is node.right:
            stage = "up from right child" 
        else:
            stage = "other"
        pass
        return stage 

    def _set_node(self, n):
        if n is None:
           pass
           #if self.debug: 
           #log.warning("_set_node to None from prev %s act %s " % (self._node,desc_action(self.act)) )

        #assert n 
        self._prev = self._node
        self._node = n
        self.record_traversal(n)

        self._stage = self._get_stage()

        if self.debug > 2:
            log.info("%s _set_node iray %d : %s -> %s : %s " % (self.typ, self.iray, self.prevname, self.nodename, self.stage)) 
        #pass

    def _get_node(self):
        return self._node  
    node = property(_get_node, _set_node)


    def record_traversal(self, n):
        assert self.typ in self._traverse, "bad typ %s " % self.typ
        self._traverse[self.typ].append(n)

    def record_action(self, a):
        assert self.typ in self._traverse, "bad typ %s " % self.typ
        self._actions[self.typ].append(a)


    prev = property(lambda self:self._prev)
    stage = property(lambda self:self._stage)

    def dump_traversal(self):
        for typ in self._traverse:
            log.info(" (%3d) %20s : %s " % (self.iray, typ, repr(self._traverse[typ]) ))  
        pass

    def dump_actions(self):
        for typ in self._traverse:
            log.info(" (%3d) %20s : %s " % (self.iray, typ, repr(map(desc_action,self._actions[typ])) ))  
        pass

    def compare_traversal(self):
        """
        Iterative requires some manual record_traversal for left/right primitives in Intersect

        Note that missers are coming up with different traversal: the recursive traversal 
        never returns to root, but the iterative does several times

        """
        if self.debug:
            self.dump_traversal()  
            self.dump_actions()  
           
        r = self._traverse["RECURSIVE"]
        i = self._traverse["ITERATIVE"]

        return i == r

    def _set_action(self, a):
        self._action = a
        self.record_action(a)
        if self.debug > 2:
            log.info("_set_action %s " % desc_action(a)) 
    def _get_action(self):
        return self._action  
    action = property(_get_action, _set_action)


    def _set_tl(self, tl):
        self._tl = tl
    def _get_tl(self):
        return self._tl  
    tl = property(_get_tl, _set_tl)

    def _set_tr(self, tr):
        self._tr = tr
    def _get_tr(self):
        return self._tr  
    tr = property(_get_tr, _set_tr)

    def _set_nr(self, nr):
        self._nr = nr
    def _get_nr(self):
        return self._nr  
    nr = property(_get_nr, _set_nr)

    def _set_nl(self, nl):
        self._nl = nl
    def _get_nl(self):
        return self._nl  
    nl = property(_get_nl, _set_nl)

    def _set_tmin(self, tmin):
        self._tmin = tmin
    def _get_tmin(self):
        return self._tmin  
    tmin = property(_get_tmin, _set_tmin)


    def classify(self, tt, nn, tmin):
        if tt > tmin:
            state = Enter if np.dot(nn, self.ray.direction) < 0. else Exit 
        else:
            state = Miss 
        pass
        return state

    def recursive_intersect(self, root, depth=0, tmin=0):
        """
        * minimizing use of member vars makes recursive algorithm easier to understand
        * instead use local vars, which will have different existance at the 
          different levels of the recursion
        * can think of member vars as effective globals wrt the recursion

        * loopers result in recursive call repeating the same node, with tmin advanced 
        * recursively never traverse the root again ? are always going down with root.left root.right
          never going up ?

        * although the traverse never goes up, completion of the recursive instance calls 
          does go back up once hitting primitives in the leaves 

        """
        assert root
        assert self.typ == "RECURSIVE"
        if depth == 0:
            self.top = root
        pass
        self.node = root  
        # above are just for debug comparison against iterative algo, not used below

        if root.is_primitive:
            return intersect_primitive(root, self.ray, tmin)

        elif root.is_operation:

            tl, nl, lname, lact  = self.recursive_intersect(root.left, depth=depth+1, tmin=tmin)  
            tr, nr, rname, ract  = self.recursive_intersect(root.right, depth=depth+1, tmin=tmin)

            root.rstack = (tl, nl, lname, lact, tr, nr, rname, ract) # for debug comparison with iterative

            loopcount = 0 
            looplimit = 10 
            while loopcount < looplimit:
                loopcount += 1 

                stateL = self.classify(tl, nl, tmin)   # tmin_l tmin_r ? 
                stateR = self.classify(tr, nr, tmin)

                acts = boolean_table(root.operation, stateL, stateR )

                opr = "%s(%s:%s,%s:%s)" % ( desc[root.operation],lname,desc_state[stateL], rname,desc_state[stateR] )

                act_RetMiss = (RetMiss & acts)
                act_RetL = (RetL & acts)
                act_RetR = (RetR & acts)
                act_LoopL = (LoopL & acts)
                act_LoopR = (LoopR & acts)
                act_RetLIfCloser = ((RetLIfCloser & acts) and tl <= tr)
                act_LoopLIfCloser = ((LoopLIfCloser & acts) and tl <= tr)
                act_RetRIfCloser = ((RetRIfCloser & acts) and tr < tl)
                act_LoopRIfCloser = ((LoopRIfCloser & acts) and tr < tl)

                trep = self.trep_fmt(tmin, tl, tr )
                ret = ()
                if act_RetMiss:
                    act = RetMiss
                    ret = None, None, None, act

                elif act_RetL or act_RetLIfCloser: 
                    act = RetLIfCloser if act_RetLIfCloser else RetL
                    ret = tl, nl, lname, act 

                elif act_RetR or act_RetRIfCloser: 
                    act = RetRIfCloser if act_RetRIfCloser else RetR
                    if (FlipR & acts): nr = -nr
                    ret = tr, nr, rname, act

                elif act_LoopL or act_LoopLIfCloser:
                    act = LoopLIfCloser if act_LoopLIfCloser else LoopL
                    tl, nl, lname, _ = self.recursive_intersect(root.left, depth=depth+1, tmin=tl)

                elif act_LoopR or act_LoopRIfCloser:
                    act = LoopRIfCloser if act_LoopRIfCloser else LoopR
                    tr, nr, rname, _ = self.recursive_intersect(root.right, depth=depth+1, tmin=tr)

                else:
                    log.fatal("[%d] RECURSIVE UNHANDLED acts " % (loopcount))
                    assert 0
                   
                self.act = act 

                if self.debug > 1:
                    log.info("(%d)[%d] RECURSIVE %s : %s -> %s : %s " % (self.iray,loopcount,root.name,opr,desc_acts(self.act),trep ))

                if len(ret) > 0:
                     
                    return ret
                else:
                    # only Loop-ers hang aroud here to intersect again with advanced tmin, the rest return up to caller
                    assert act in [LoopLIfCloser, LoopL, LoopRIfCloser, LoopR]
                pass


            else:
                log.fatal(" depth %d root %s root.is_operation %d root.is_primitive %d " % (depth, root,root.is_operation, root.is_primitive) )
                assert 0
        pass
        log.fatal("[%d] RECURSIVE count EXCEEDS LIMIT %d " % (loopcount, looplimit))
        assert 0  
        return None, None, None, 0


    def iterative_intersect(self, root):
        """
        Iterative CSG boolean intersection

        * https://www.hackerearth.com/practice/notes/iterative-tree-traversals/

        """
        assert self.typ == "ITERATIVE"
        self.top = root
        self.node = root

        if self.node.is_primitive:
            return intersect_primitive(self.node, self.ray, self.tmin)
 
        self.count = 0 
        limit = 10 

        self.actionStack.push(Compute)
        #self.actionStack.push(GotoLft)

        self.action = GotoLft

        do_goto = False  
        # ignoring the first GotoLft avoids the virtual root and associated termination complications

        while self.actionStack.count() > 0:
            self.count += 1 

            #self.action = self.actionStack.pop(debug=self.debug > 2)

            if self.action == SaveLft:
                self.SaveLft_()
                self.action = GotoRgh
         
            if self.action == GotoLft or self.action == GotoRgh:
                if do_goto:
                    self.GoTo()
                pass
                do_goto = True
                
            if self.action == GotoLft or self.action == GotoRgh:
                self.Intersect()
 
            if self.action == LoadLft or self.action == LoadRgh:
                self.Load_()
                self.action = Compute

            if self.action == Compute:
                self.Compute_()
            pass 

            if self.action == Return:
                break 

            if self.count == limit: 
                log.fatal("iray %d ray %s count %d reaches limit %d " % (self.iray, self.ray, self.count, limit))
                assert 0
 
        pass
        return self.Return_()
      

    def SaveLft_(self):
        self.tmin = self.tminStack.pop(debug=self.debug > 1)
        self.primStack.push((self.tl,self.nl,self.lname, self.lact), debug=self.debug > 1)

    def Load_(self):
        if self.action == LoadLft:
            self.tl, self.nl, self.lname, self.lact = self.primStack.pop(debug=self.debug > 1)
        elif self.action == LoadRgh:
            self.tr, self.nr, self.rname, self.ract = self.primStack.pop(debug=self.debug > 1)
        else:
            assert 0, action  
        pass

    def GoTo(self):
        assert self.action in [GotoLft, GotoRgh] 

        self.node = self.node.left if self.action == GotoLft else self.node.right 

        if self.debug > 3:
            log.info("GoTo: node %s after action %s from parent %r " % (self.node, desc_action(self.action),self.prev) )


        assert self.node
        if self.node is None:
            log.fatal("GoTo: node None after action %s from parent %r " % (desc_action(self.action),self.prev) )
            return 

    def Intersect(self):
        """
        # the below handling of a operation holding primitives 
        # initially seems a special case cop out, subverting the iterative approach, 
        # that is liable to to work for simple trees, but not for complex ones 
        # 
        # BUT on deeper refeclection that isnt the case, need to allow to keep going left until 
        # find primitives one level below in order to get the ball rolling 
        # and start filling the primStack, as go back upwards
        """
        action = self.action
        assert action in [GotoLft, GotoRgh]


        if self.node.is_primitive:

            tt, nn, name, act = intersect_primitive( self.node, self.ray, self.tmin )
            if action ==  GotoLft:
                self.tl = tt
                self.nl = nn
                self.lname = name
                self.lact = act
                if self.debug > 3:
                    log.info("pr.Intersect.GotoLft %s tl %5.2f lname %s " % (self.node.name, self.tl if self.tl else -1, self.lname )) 
                pass
            elif action == GotoRgh:
                self.tr = tt
                self.nr = nn
                self.rname = name
                self.ract = act
                if self.debug > 3:
                    log.info("pr.Intersect.GotoRgh %s tr %5.2f rname %s " % (self.node.name, self.tr if self.tr else -1, self.rname )) 
                pass
            pass
            self.action = Compute
            self.node = self.node.parent

        elif self.node.is_operation:

            gotoL = intersectBox(self.node.left)
            gotoR = intersectBox(self.node.right)
            
            if gotoL and self.node.left.is_primitive:
                self.record_traversal(self.node.left)
                tt, nn, name, act = intersect_primitive(self.node.left, self.ray, self.tmin)
                self.tl = tt
                self.nl = nn
                self.lname = name
                self.lact = act
                gotoL = False

                if self.debug > 3:
                    log.info("op.Intersect.gotoL %s tl %5.2f lname %s " % (self.node.left.name, self.tl, self.lname )) 

            
            if gotoR and self.node.right.is_primitive:
                self.record_traversal(self.node.right)
                tt, nn, name, act = intersect_primitive(self.node.right, self.ray, self.tmin)
                self.tr = tt
                self.nr = nn
                self.rname = name
                self.ract = act
                gotoR = False

                if self.debug > 3:
                    log.info("op.Intersect.gotoR %s tr %5.2f rname %s " % (self.node.right.name, self.tr, self.rname )) 
             
            # immediate right/left primitives are not stacked, as are ready for compute 
            if gotoL or gotoR:
                if gotoL:
                    # non-primitive subtree intersect
                    self.primStack.push((self.tl, self.nl, self.lname, self.lact), debug=self.debug > 1)
                    self.actionStack.push(LoadLft, debug=self.debug > 1)
                elif gotoR:
                    self.primStack.push((self.tr, self.nr, self.rname, self.ract), debug=self.debug > 1)
                    self.actionStack.push(LoadRgh, debug=self.debug > 1)
                pass
            else:
                # both gotoL and gotoR False means miss OR both prim intersects done, so are ready for compute
                self.tminStack.push(self.tmin, debug=self.debug > 1)
                self.actionStack.push(LoadLft,debug=self.debug > 1)
                self.actionStack.push(SaveLft,debug=self.debug > 1)
            pass
            # NB not the same as doing this interleaved within the above, as this places
            # no demands on (gotoL or gotoR)
            if gotoL:
                self.action = GotoLft
            elif gotoR:
                self.action = GotoRgh
            else:
                self.action = Compute 
            pass

            if self.debug > 3:
                log.info("Intersect -> %s  tr/tl %5.2f/%5.2f rname/lname %s/%s " % (desc_action(self.action),self.tr if self.tr else -1,self.tl if self.tl else -1,self.rname,self.lname )) 
 

        else:
            assert 0  



    @classmethod
    def trep_fmt(cls, tmin, tl, tr ):
        return "tmin/tl/tr %5.2f %5.2f %5.2f " % (tmin if tmin else -1, tl if tl else -1, tr if tr else -1 )

    def _get_trep(self):
        return self.trep_fmt(self.tmin, self.tl, self.tr)
    trep = property(_get_trep)


    def Up(self):
        """
        Up only called from RetMiss, RetL, RetLIfCloser, RetR, RetRIfCloser branches of Compute_
        which when combined with no parent node seems like a good termination signal
        ... but action stack may not be emptied ?
        """
        if self.node.parent is None:
            if self.debug > 4:
                log.info("Up setting Return... actionStack %s " % self.actionStack.desc()) 
                log.info("Up setting Return... primStack   %s " % self.primStack.desc()) 
                log.info("Up setting Return... tminStack   %s " % self.tminStack.desc()) 
            pass
            self.action = Return
            #self.actionStack.push(Return)
            #self.action = self.actionStack.pop(debug=self.debug > 1) 
        else:
            self.action = self.actionStack.pop(debug=self.debug > 1) 
            self.node = self.node.parent
        pass

    def Compute_(self):
        """
        Hmm, surely the loopers can be more simply implemented (without stacks and action), 
        they just correspond to repeating the lookup with tmin advanced for one side.
        """
        assert self.node.is_operation

        stateL = self.classify( self.tl, self.nl, self.tmin )
        stateR = self.classify( self.tr, self.nr, self.tmin )

        if hasattr(self.node, "rstack"):
            rstack = self.node.rstack
            if self.debug > 3:
                log.info("rstack %s " % repr(rstack))
            pass
        pass
        #if self.tl != rstack[0] or self.tr != rstack[3] or self.nr != rstack[1] or self.rname != rstack[2]

        acts = boolean_table(self.node.operation, stateL, stateR )
        opr = "%s(%s:%s,%s:%s)" % ( desc[self.node.operation],self.lname,desc_state[stateL], self.rname,desc_state[stateR] )

        act = 0
        act_RetMiss = (RetMiss & acts)
        act_RetL = (RetL & acts)
        act_RetR = (RetR & acts)
        act_LoopL = (LoopL & acts)
        act_LoopR = (LoopR & acts)
        act_RetLIfCloser = ((RetLIfCloser & acts) and self.tl <= self.tr)
        act_LoopLIfCloser = ((LoopLIfCloser & acts) and self.tl <= self.tr)
        act_RetRIfCloser = ((RetRIfCloser & acts) and self.tr < self.tl)
        act_LoopRIfCloser = ((LoopRIfCloser & acts) and self.tr < self.tl)

        trep = self.trep # prior to the mods below
        node = self.node  # local copy, as Up may change self.node to parent 

        if act_RetMiss:
            act = RetMiss
            self.tr = None
            self.nr = None
            self.rname = None
            self.ract = act
            self.tl = None
            self.nl = None
            self.lname = None
            self.lact = act
            self.Up()

        elif act_RetL or act_RetLIfCloser: 
            act = RetLIfCloser if act_RetLIfCloser else RetL
            self.lact = act

            self.tr = self.tl
            self.nr = self.nl
            self.ract = self.lact
            self.Up()

        elif act_RetR or act_RetRIfCloser: 
            act = RetRIfCloser if act_RetRIfCloser else RetR
            self.ract = act
            if (FlipR & acts): self.nr = -self.nr
            self.tl = self.tr
            self.nl = self.nr
            self.lact = self.ract
            self.Up()

        elif act_LoopL or act_LoopLIfCloser:
            act = LoopLIfCloser if act_LoopLIfCloser else LoopL
            self.tmin = self.tl
            self.primStack.push((self.tr,self.nr,self.rname,act), debug=self.debug > 1)
            self.actionStack.push(LoadRgh, debug=self.debug > 1)
            self.action = GotoLft

        elif act_LoopR or act_LoopRIfCloser:
            act = LoopRIfCloser if act_LoopRIfCloser else LoopR
            self.tmin = self.tr
            self.primStack.push((self.tl,self.nl,self.lname,act), debug=self.debug > 1)
            self.actionStack.push(LoadLft, debug=self.debug > 1)
            self.action = GotoRgh
        else:
            assert 0
        pass

        self.act = act 
 
        if self.debug > 1:
            log.info("(%d)[%d] ITERATIVE %s : %s -> %s : %s" % (self.iray,self.count, node.name, opr, desc_acts(act), trep))
        pass
 
    def Return_(self):
        #assert self.action == Return
        
        if self.act in [RetMiss]:
            return None, None, None, self.act
        elif self.act in [RetL, RetLIfCloser]:
            return self.tl, self.nl, self.lname, self.act
        elif self.act in [RetR, RetRIfCloser]:
            return self.tr, self.nr, self.rname, self.act
        else:
            log.warning("%d iray %d iterative returned to top with unexpected act %s " % (self.count, self.iray,desc_action(self.act)))
        pass

    def compare_intersects(self, tst):
 
        nray = len(tst.rays)
        self.ipos = np.zeros((2,nray, 3), dtype=np.float32 ) 
        self.ndir = np.zeros((2,nray, 3), dtype=np.float32 ) 
        self.tval = np.zeros((2,nray), dtype=np.float32 )
        self.aval = np.zeros((2,nray), dtype=np.int32 )

        self.prob = []
        self.trob = []
        for iray, ray in enumerate(tst.rays):
            for recursive in [1,0]:
                self.typ = "RECURSIVE" if recursive else "ITERATIVE"
                self.reset(ray=ray, iray=iray, debug=tst.debug) 

                if iray in tst.skip:
                    log.warning("skipping iray %d " % iray)
                    continue 

                if self.debug > 0:
                    log.info(" ray(%d) %r " % (iray,ray) ) 
                if self.debug > 1:
                    log.info(" %r " % (tst.root)) 

                if recursive:
                    tt, nn, nname, act = self.recursive_intersect(tst.root, depth=0)
                else:
                    tt, nn, nname, act = self.iterative_intersect(tst.root)   
                pass
                if self.debug:
                    log.info("[%d] %s intersect tt %s nn %r " % (-1, self.typ, tt, nn ))

                if not tt is None:
                    ix = 0 if self.typ == "ITERATIVE" else 1
                    self.ipos[ix,iray] = ray.position(tt)
                    self.ndir[ix,iray] = nn
                    self.tval[ix,iray] = tt
                    self.aval[ix,iray] = act 
                pass
            pass

            ok_pos = np.allclose( self.ipos[0,iray], self.ipos[1,iray] )
            ok_dir = np.allclose( self.ndir[0,iray], self.ndir[1,iray] )
            ok_tva = np.allclose( self.tval[0,iray], self.tval[1,iray] )

            if not (ok_pos and ok_dir and ok_tva):
                self.prob.append(iray)

            ok_tra = self.compare_traversal()
            if not ok_tra:
                self.trob.append(iray) 
            pass
        pass
        log.info("%10s %d/%d rays with intersect mismatches : %s " % (tst.name, len(self.prob),nray,repr(self.prob)))
        log.info("%10s %d/%d rays with traversal mismatches : %s " % (tst.name, len(self.trob),nray,repr(self.trob)))

    def plot_intersects(self, plt, normal=False):
        sc = 10 

        prob = self.trob

        for recursive in [1, 0]:
            xoff = 600 if recursive else 0
            plt.scatter( xoff + self.ipos[recursive,:,0]                        , self.ipos[recursive,:,1] )
            if normal:
                plt.scatter( xoff + self.ipos[recursive,:,0]+self.ndir[recursive,:,0]*sc , self.ipos[recursive,:,1]+self.ndir[recursive,:,1]*sc )
    
                if len(prob) > 0:
                    plt.scatter( xoff + self.ipos[recursive, prob,0], self.ipos[recursive, prob,1], c="r" )
                    plt.scatter( xoff + self.ipos[recursive, prob,0], self.ipos[recursive, prob,1], c="g" )

 


def traverse(top):
    for act in ["label","dump"]:
        node = top
        idx = 0 
        q = []
        q.append(node)
        while len(q) > 0:
            node = q.pop(0)   # bottom of q (ie fifo)

            if act == "label":
                node.name = "%s_%s%d" % (node.name, "p" if node.is_primitive else "o", idx)
            elif act == "dump":
                pass
                #log.info("[%d] %r " % (idx, node))
            else:
                pass
            if not node.is_primitive: 
                if not node.left is None:q.append(node.left)
                if not node.right is None:q.append(node.right)
            pass
            idx += 1 



def test_intersect(csg, tst):
    """
    * only 1st intersect is returned, so to see inside and outside of   
      a shape need to send rays from inside and outside

    """
    traverse(tst.root)

    csg.compare_intersects( tst )

    csg.plot_intersects( plt )

    plt.show()




class T(object):
    def __init__(self, root, debug=[], skip=[], notes="", source="aringlight,origlight", num=200, level=1):
        """
        :param root:
        :param name:
        :param debug: list of ray index for dumping
        """
        self.root = root
        self.name = root.name
        self.debug = debug
        self.skip = skip
        self.notes = notes
        self.source = source
        self.num = num
        self.level = level

    def _get_rays(self):
        rays = []
        if "xray" in self.source:
            rays += [Ray(origin=[0,0,0], direction=[1,0,0])]

        if "aringlight" in self.source:
            ary = Ray.aringlight(num=self.num, radius=1000)
            rays += Ray.make_rays(ary)

        if "origlight" in self.source:
            rays += Ray.origlight(num=self.num)

        if "lsquad" in self.source:
            rays += [Ray(origin=[-300,y,0], direction=[1,0,0]) for y in range(-50,50+1,10)]
        pass
        return rays

    rays = property(_get_rays)







if __name__ == '__main__':

    plt.ion()
    plt.close()

    logformat = "%(asctime)s %(name)s %(levelname)-8s %(message)s"
    logging.basicConfig(level=logging.INFO,format=logformat)
    log = logging.getLogger(__name__)

    ## need to clone to avoid inadvertent parent connections between different roots 
    ## TODO: manage this inside Node and think what parent connections should be when cloning

    cbox = Node(BOX, param=[0,0,0,100], name="cbox")
    lbox = Node(BOX, param=[-200,0,0,50], name="lbox")
    rbox = Node(BOX, param=[ 200,0,0,50], name="rbox")
    lrbox = Node(None,lbox.clone(),  rbox.clone(), UNION, name="lrbox") 

    bms = Node(None, Node(BOX, param=[0,0,0,200], name="box"),  Node(SPHERE,param=[0,0,0,150],name="sph"), DIFFERENCE, name="bms")
    smb = Node(None, Node(SPHERE,param=[0,0,0,200], name="sph"), Node(BOX,param=[0,0,0,150], name="box"), DIFFERENCE , name="smb")
    ubo = Node(None, bms.clone(), lrbox.clone(), UNION , name="ubo")
    bmslrbox = Node( None, Node(None, bms.clone(), rbox.clone(), UNION,name="bmsrbox"),lbox.clone(),UNION, name="bmslrbox" ) 

    bmsrbox = Node(None, bms.clone(), rbox.clone(), UNION,name="bmsrbox")
    smblbox = Node(None, smb.clone(), lbox.clone(), UNION,name="smblbox")




    # bmslrbox : 
    #         U( bms_rbox_u : 
    #                U( bms : 
    #                         D(bms_box : BX ,
    #                           bms_sph : SP ),
    #                      rbox : BX ),
    #                  lbox : BX ) 
    #


    bmsrlbox = Node( None, Node(bms.clone(), lbox.clone(), UNION,name="bms_lbox"),rbox.clone(),UNION, name="bmsrlbox" ) 


    csph = Node(SPHERE, param=[0,0,0,100], name="csph")
    lsph = Node(SPHERE, param=[-50,0,0,100], name="lsph")
    rsph = Node(SPHERE, param=[50,0,0,100], name="rsph")

    lrsph_u = Node(None, lsph.clone(), rsph.clone(), UNION, name="lrsph_u")
    lrsph_i = Node(None, lsph.clone(), rsph.clone(), INTERSECTION, name="lrsph_i")
    lrsph_d = Node(None, lsph.clone(), rsph.clone(), DIFFERENCE , name="lrsph_d")


    ok0 = [ 
             #T(lrsph_i, source="origlight"), 
             #T(lrsph_i, source="aringlight", notes="all iterative aringlight miss in actionStack while mode", level=2, debug=[0]), 
             #T(lrbox), 
             #T(bms, level=4, debug=[0]),
             #T(csph, source="origlight", debug=[0], level=2),
             T(smb, source="aringlight,origlight", debug=[23], skip=[], level=4),
          ]

    ok = [ 
             T(smb),
             T(bms),
             T(csph),
             T(cbox),
             T(lbox),
             T(rbox),
             T(lrbox),
             T(lrsph_d),
             T(lrsph_u, notes="fixed all rightside mismatched with origlight by adopting clone to avoid inadventent parent relationship to other shape"),
             T(lrsph_i),
             T(bmsrbox),
         ]


    nok = [
             #T(bmslrbox, notes="left box protrusion is missed for iterative", debug=[92], level=2),
             T(smblbox, notes="box corners are discrepantly present for iterative", debug=[23], level=2),
             #T(bmslrbox, notes="left box protrusion is missed for iterative", source="lsquad", debug=[1]),
             #T(bmsrlbox, notes="right box protrusion is missed for iterative"),
             #T(ubo, [], notes="looks to be missing most intersects???"),
          ]

    for tst in ok0:
        csg = CSG(level=tst.level)
        test_intersect(csg,tst)
    pass


