#!/usr/bin/env python
"""

* :google:`Spatially Efficient Tree Layout for GPU Ray-tracing of Constructive Solid Geometry Scenes`
* http://ceur-ws.org/Vol-1576/090.pdf

"""


SPHERE = 1
BOX = 2 
is_shape = lambda c:c in [SPHERE, BOX]

DIVIDER = 99  # between shapes and operations

UNION = 100
INTERSECTION = 101
DIFFERENCE = 102
is_operation = lambda c:c in [UNION,INTERSECTION,DIFFERENCE]

CODE_JK = 3,3   # item position of shape/operation code

desc = { SPHERE:"SPHERE", BOX:"BOX", UNION:"UNION", INTERSECTION:"INTERSECTION", DIFFERENCE:"DIFFERENCE" }






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


class Node(object):
    def __init__(self, left, right=None, operation=None, param=None):

        self.left = left
        self.right = right
        self.operation = operation
        self.param = param

        if not operation is None:
            left.parent = self 
            right.parent = self 
        pass


    is_primitive = property(lambda self:self.operation is None and self.right is None and not self.left is None)
    is_operation = property(lambda self:not self.operation is None)

    def __repr__(self):
        if self.is_primitive:
            return desc[self.left]
        else:
            return "%s(%s,%s)" % ( desc[self.operation], repr(self.left), repr(self.right) )




actionStack = []
def pushAction(action):
    global actionStack
    actionStack.append(action)
def popAction(action):
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
def pushPrimitive(t_n):
    global primitiveStack
    primitiveStack.append(t_n)
def popPrimitive():
    global primitiveStack
    return primitiveStack.pop()


def intersectBox(node):
    return True     

def intersect(node, tt):
    return 0.5,(0,0,1)

def classify(tt,nn):
    return 1 



def GoTo():
    global tmin
    global action
    global node
    print "GoTo"

    if action == GotoLft:
        node = node.left
    else:
        node = node.right
    pass

    if node.is_operation:
        gotoL = intersectBox(node.left)
        gotoR = intersectBox(node.right)
 
        if gotoL and node.left.is_primitive:
            tl, nl = intersect(node.left, tmin)
            gotoL = False

        if gotoR and node.right.is_primitive:
            tr, nr = intersect(node.right, tmin)
            gotoR = False

        if gotoL or gotoR:
            if gotoL:
                pushPrimitive((tl, nl))
                pushAction(LoadLft)
            elif gotoR:
                pushPrimitive((tr, nr))
                pushAction(LoadRgh)
            pass
        else:
            pushTmin(tmin)
            pushAction(LoadLft)
            pushAction(SaveLft)
        pass

        if gotoL: 
            action = GotoLft
        elif gotoR:
            action = GotoRgh
        else:
            action = Compute 
        pass

    else:   # node is a Primitive

        if action ==  GotoLft:
            tl, nl = intersect(node, tmin)
        else:
            tr, nr = intersect(node, tmin)

        action = Compute
        node = node.parent



def Compute_():
    print "Compute_"

    global action
    global tl, nl
    global tr, nr

    if action & (LoadLft | LoadRgh):
        tl, nl = popPrimitive()
    else:
        tr, nr = popPrimitive()

    stateL = classify(tl,nl)
    stateR = classify(tr,nr)

    acts = table(stateL, stateR )

    if (RetL & acts) or ((RetLIfCloser & acts) and tl <= tr): 
        tr = tl
        nr = nl
        action = popAction()
        node = node.parent




if __name__ == '__main__':
    action = GotoLft | GotoRgh | LoadLft | LoadRgh | Compute | SaveLft
    print action_desc(action)


    bms = Node(Node(BOX, param=[0,0,-100,200]),  Node(SPHERE,param=[0,0,-100,200]), DIFFERENCE )
    smb = Node(Node(SPHERE,param=[0,0,100,300]), Node(BOX,param=[0,0,100,300]), DIFFERENCE )
    ubo = Node(bms, smb, UNION )

    root = ubo
    V = Node(left=root)    


    ########

    tmin = 0 
    node = V  # vitual root whose left subtree is the real root
    tl, nl = None, None
    tr, nr = None, None

    pushAction(Compute)
    action = GotoLft

    while True:
        if action == SaveLft:
            tmin = popTmin()
            pushPrimitive((tl,nl))
            action = GotoRgh
        if action & ( GotoLft | GotoRgh ):
            GoTo()
        if action & ( LoadLft | LoadRgh | Compute ):
            Compute_()
        




