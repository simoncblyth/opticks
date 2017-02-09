#!/usr/bin/env python
"""

* :google:`Spatially Efficient Tree Layout for GPU Ray-tracing of Constructive Solid Geometry Scenes`


PCT 2016
Parallel Computing Technologies
Proceedings of the 10th Annual International Scientific Conference on Parallel Computing Technologies
Arkhangelsk, Russia, March 29-31, 2016.

* http://ceur-ws.org/Vol-1576/  
* http://ceur-ws.org/Vol-1576/090.pdf

"""

import logging
log = logging.getLogger(__name__)
import numpy as np
from intersect import intersect_node, Node, Ray, UNION, INTERSECTION, DIFFERENCE, BOX, SPHERE

CODE_JK = 3,3   # item position of shape/operation code


# intersect status
Enter = 1
Exit  = 2
Miss  = 3

desc_state = { Enter : "Enter", Exit : "Exit", Miss : "Miss" }


# acts
RetL          = 0x1 << 1 
RetR          = 0x1 << 2 
RetLIfCloser  = 0x1 << 3 
RetRIfCloser  = 0x1 << 4
LoopL         = 0x1 << 5 
LoopLIfCloser = 0x1 << 6
LoopR         = 0x1 << 7 
LoopRIfCloser = 0x1 << 8
FlipR         = 0x1 << 9


def desc_acts(acts):
    s = ""
    if acts & RetL: s+= "RetL "
    if acts & RetR: s+= "RetR "
    if acts & RetLIfCloser: s+= "RetLIfCloser "
    if acts & RetRIfCloser: s+= "RetRIfCloser "
    if acts & LoopL: s+= "LoopL "
    if acts & LoopLIfCloser: s+= "LoopLIfCloser "
    if acts & LoopR: s+= "LoopR "
    if acts & LoopRIfCloser: s+= "LoopRIfCloser "
    if acts & FlipR: s+= "FlipR "
    return s 


table_ = {
    DIFFERENCE : { 
                  Enter : {
                             Enter : RetLIfCloser | LoopR,
                              Exit : LoopLIfCloser | LoopRIfCloser,
                              Miss : RetL
                          },

                  Exit: {
                             Enter : RetLIfCloser | RetRIfCloser | FlipR,
                             Exit  : RetRIfCloser | FlipR | LoopL,
                             Miss  : RetL
                       },

                  Miss: {
                             Enter : Miss,
                             Exit : Miss,
                             Miss : Miss
                        }
               }, 

    UNION : {
                 Enter : {
                            Enter : RetLIfCloser | RetRIfCloser, 
                            Exit  : RetRIfCloser | LoopL,
                            Miss  : RetL
                         },

                 Exit  : {
                            Enter : RetLIfCloser | LoopR, 
                            Exit  : LoopLIfCloser | LoopRIfCloser,
                            Miss  : RetL
                         },

                  Miss: {
                             Enter : RetR,
                             Exit  : RetR,
                             Miss  : Miss
                        }
               },
 
   INTERSECTION : {
                         Enter : {
                                    Enter : LoopLIfCloser | LoopRIfCloser,
                                    Exit  : RetLIfCloser | LoopR ,
                                    Miss  : Miss
                                 },

                         Exit :  {
                                    Enter : RetRIfCloser | LoopL,
                                    Exit  : RetLIfCloser | RetRIfCloser,
                                    Miss  : Miss 
                                 },

                         Miss :  {
                                    Enter : Miss,  
                                    Exit  : Miss,  
                                    Miss  : Miss
                                 }
                      }
}


def table(operation, l, r):
    return table_[operation][l][r]


# actions
GotoLft = 0x1 << 1 
GotoRgh = 0x1 << 2
LoadLft = 0x1 << 3 
LoadRgh = 0x1 << 4
Compute = 0x1 << 5  
SaveLft = 0x1 << 6 

def action_desc(action):
    s = ""
    if action & GotoLft:s+="GotoLft " ; 
    if action & GotoRgh:s+="GotoRgh " ; 
    if action & LoadLft:s+="LoadLft " ; 
    if action & LoadRgh:s+="LoadRgh " ; 
    if action & Compute:s+="Compute " ; 
    if action & SaveLft:s+="SaveLft " ; 
    return s 





actionStack = []
def pushAction(action, label=None):
    global actionStack
    if not label is None:
        log.info("pushAction %s %s " % (label, action_desc(action) ))
    actionStack.append(action)
def popAction():
    global actionStack
    return actionStack.pop()
    
tminStack = []
def pushTmin(tmin):
    global tminStack
    tminStack.append(tmin)
def popTmin():
    global tminStack
    return tminStack.pop()

primitiveStack = []
def pushPrimitive(t_n, label=None):
    global primitiveStack
    if not label is None:
        log.info("pushPrimitive %s %s " % (label, repr(t_n)))
    primitiveStack.append(t_n)
def popPrimitive():
    global primitiveStack
    return primitiveStack.pop()


def intersectBox(node):
    return True        # skipping bbox optimization, just test against the primitive


def classify(tt,nn):
    if tt > ray.tmin:
        state = Enter if np.dot(nn, ray.direction) < 0 else Exit 
    else:
        state = Miss 
    pass
    return state

def dump(label):
    global action
    global tl, nl, tr, nr
    return "%s %s tl:%s nl:%s tr:%s nr%s " % (label, action_desc(action),repr(tl),repr(nl),repr(tr),repr(nr))


def GoTo():
    global tl, nl, tr, nr
    global action
    global node

    print dump("GoTo")

    if action == GotoLft:
        node = node.left
    else:
        node = node.right
    pass

    if node.is_operation:
        gotoL = intersectBox(node.left)
        gotoR = intersectBox(node.right)
 
        if gotoL and node.left.is_primitive:
            tl, nl = intersect_node(node.left, ray)
            gotoL = False

        if gotoR and node.right.is_primitive:
            tr, nr = intersect_node(node.right, ray)
            gotoR = False

        print "gotoL %s gotoR %s " % (gotoL, gotoR )

        if gotoL or gotoR:
            if gotoL:
                pushPrimitive((tl, nl), label="gotoL")
                pushAction(LoadLft, label="gotoL")
            elif gotoR:
                pushPrimitive((tr, nr), label="gotoR")
                pushAction(LoadRgh, label="gotoR")
            pass
        else:
            pushTmin(ray.tmin)
            pushAction(LoadLft, label="no-goto")
            pushAction(SaveLft, label="no-goto")
        pass

        if gotoL: 
            action = GotoLft
        elif gotoR:
            action = GotoRgh
        else:
            action = Compute 
        pass
        print dump("node.is_operation conclusion %s " % action_desc(action))

    else:   # node is a Primitive

        if action ==  GotoLft:
            tl, nl = intersect_node(node, ray)
        else:
            tr, nr = intersect_node(node, ray)

        action = Compute
        node = node.parent


def Compute_():

    global action
    global tl, nl
    global tr, nr
    global node

    dump("Compute_")

    if action == LoadLft or action == LoadRgh:
        if action == LoadLft:
            tl, nl = popPrimitive()
        else:
            tr, nr = popPrimitive()
        pass

    stateL = classify(tl,nl)
    stateR = classify(tr,nr)

    acts = table(node.operation, stateL, stateR )

    log.info("Compute stateL %s stateR %s -> acts %s  tl %s tr %s " % ( desc_state[stateL], desc_state[stateR], desc_acts(acts), tl, tr ))


    if (RetL & acts) or ((RetLIfCloser & acts) and tl <= tr): 
        tr = tl
        nr = nl
        action = popAction()
        node = node.parent
    pass

    if (RetR & acts) or ((RetRIfCloser & acts) and tr < tl): 
        if (FlipR & acts): nr = -nr
        tl = tr
        nl = nr
        action = popAction()
        node = node.parent
    elif (LoopL & acts) or ((LoopLIfCloser & acts) and tl <= tr):
        ray.tmin = tl
        pushPrimitive((tr,nr))
        pushAction(LoadRgh)
        action = GotoLft
    elif (LoopR & acts) or ((LoopRIfCloser & acts) and tr < tl):
        ray.tmin = tr
        pushPrimitive((tl,nl))
        pushAction(LoadLft)
        action = GotoRgh
    else:
        tr = None
        action = popAction()




if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)

    bms = Node(Node(BOX, param=[0,0,0,200]),  Node(SPHERE,param=[0,0,0,200]), DIFFERENCE )
    smb = Node(Node(SPHERE,param=[0,0,100,300]), Node(BOX,param=[0,0,100,300]), DIFFERENCE )
    ubo = Node(bms, smb, UNION )

    root = bms
    V = Node(left=root)    

    ray = Ray(origin=[0,0,0], direction=[1,1,0])

    ########
if 1:
    ray.tmin = 0 
    node = V  # virtual root whose left subtree is the real root
    tl, nl = None, None
    tr, nr = None, None

    pushAction(Compute)
    action = GotoLft

    count = 0 
    limit = 10

    while count < limit:
        log.info("while (%d)" % count )
        count += 1 

        if action == SaveLft:
            ray.tmin = popTmin()
            pushPrimitive((tl,nl))
            action = GotoRgh
        pass
        if action == GotoLft or action == GotoRgh:
            GoTo()
        pass
        if action == LoadLft or action == LoadRgh or action == Compute:
            Compute_()
        pass
        




