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
import numpy as np
import matplotlib.pyplot as plt

from intersect import intersect_primitive, Node, Ray, UNION, INTERSECTION, DIFFERENCE, BOX, SPHERE, desc

CODE_JK = 3,3   # item position of shape/operation code


# intersect status
Enter = 1
Exit  = 2
Miss  = 3
desc_state = { Enter : "Enter", Exit : "Exit", Miss : "Miss" }

# acts
RetMiss       = 0x1 << 0 
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
    if acts & RetMiss:      s+= "RetMiss "
    if acts & RetL:         s+= "RetL "
    if acts & RetR:         s+= "RetR "
    if acts & RetLIfCloser: s+= "RetLIfCloser "
    if acts & RetRIfCloser: s+= "RetRIfCloser "
    if acts & LoopL:        s+= "LoopL "
    if acts & LoopLIfCloser:s+= "LoopLIfCloser "
    if acts & LoopR:        s+= "LoopR "
    if acts & LoopRIfCloser:s+= "LoopRIfCloser "
    if acts & FlipR:        s+= "FlipR "
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
                             Enter : RetMiss,
                             Exit : RetMiss,
                             Miss : RetMiss
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
                             Miss  : RetMiss
                        }
               },
 
   INTERSECTION : {
                         Enter : {
                                    Enter : LoopLIfCloser | LoopRIfCloser,
                                    Exit  : RetLIfCloser | LoopR ,
                                    Miss  : RetMiss
                                 },

                         Exit :  {
                                    Enter : RetRIfCloser | LoopL,
                                    Exit  : RetLIfCloser | RetRIfCloser,
                                    Miss  : RetMiss 
                                 },

                         Miss :  {
                                    Enter : RetMiss,  
                                    Exit  : RetMiss,  
                                    Miss  : RetMiss
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
    if action is None:return "NONE"
    s = ""
    if action & GotoLft:s+="GotoLft " ; 
    if action & GotoRgh:s+="GotoRgh " ; 
    if action & LoadLft:s+="LoadLft " ; 
    if action & LoadRgh:s+="LoadRgh " ; 
    if action & Compute:s+="Compute " ; 
    if action & SaveLft:s+="SaveLft " ; 
    return s 


count = 0 

actionStack = []
def pushAction(action, label=None):
    global actionStack
    if not label is None:
        log.debug("pushAction %s %s " % (label, action_desc(action) ))
    actionStack.append(action)
def popAction():
    global actionStack
    if len(actionStack) == 0:
       log.warning("popAction empty stack ")
       return None
    pass
    return actionStack.pop()
    
tminStack = []
def pushTmin(tmin):
    global tminStack
    tminStack.append(tmin)
def popTmin():
    global tminStack
    return tminStack.pop()

primitiveStack = []
def pushPrimitive(t_n_n, label=None):
    global primitiveStack
    if not label is None:
        log.debug("pushPrimitive %s %s " % (label, repr(t_n_n)))
    primitiveStack.append(t_n_n)
def popPrimitive():
    global primitiveStack
    return primitiveStack.pop()

def intersectBox(node):
    return True        # skipping bbox optimization, just test against the primitive

def classify(tt,nn):
    if tt > tmin:
        state = Enter if np.dot(nn, ray.direction) < 0. else Exit 
    else:
        state = Miss 
    pass
    return state

def dump(label):
    return "%s %s tl:%s nl:%s tr:%s nr:%s " % (label, action_desc(action),repr(tl),repr(nl),repr(tr),repr(nr))


def rintersect_node(node, ray):
    """
    Recursive CSG boolean intersection

    * maybe need to split tmin for l and r ?

    """
    global count, limit, debug, tl, nl, tr, nr, tmin, lname, rname
    if node.is_primitive:
        return intersect_primitive(node, ray, tmin)
    else:
        tl, nl, lname = rintersect_node(node.left, ray)
        tr, nr, rname = rintersect_node(node.right, ray)

        stateL = classify(tl, nl)
        stateR = classify(tr, nr)

        while count < limit:
            count += 1 
            acts = table(node.operation, stateL, stateR )

            if debug:
                log.info("[%d] RECURSIVE %s(%s:%s,%s:%s) -> %s: %s " % ( count, desc[node.operation],lname,desc_state[stateL], rname,desc_state[stateR], desc_acts(acts), trep() ))

            if RetMiss & acts:
                if debug:
                    log.info("[%d] RECURSIVE RetMiss : %s " % (count,trep() ))
                return None, None, None
            elif (RetL & acts) or ((RetLIfCloser & acts) and tl <= tr): 
                if debug:
                    log.info("[%d] RECURSIVE RetL/RetLIfCloser : %s " % (count,trep()))
                return tl, nl, lname 
            elif (RetR & acts) or ((RetRIfCloser & acts) and tr < tl): 
                if debug:
                    log.info("[%d] RECURSIVE RetR/RetRIfCloser : %s " % (count,trep() ))
                if (FlipR & acts): nr = -nr
                return tr, nr, rname
            elif (LoopL & acts) or ((LoopLIfCloser & acts) and tl <= tr):
                if debug:
                    log.info("[%d] RECURSIVE LoopL/LoopLIfCloser : %s  " % (count,trep() ))
                tmin = tl
                tl, nl, lname = rintersect_node(node.left, ray)
                stateL = classify(tl,nl)
            elif (LoopR & acts) or ((LoopRIfCloser & acts) and tr < tl):
                if debug:
                    log.info("[%d] RECURSIVE LoopR/LoopRIfCloser : %s " % (count,trep() ))
                tmin = tr
                tr, nr, rname = rintersect_node(node.right, ray)
                stateR = classify(tr,nr)
            else:
                assert 0
                return None, None, None
        pass
        return None, None, None


def iintersect_node(ray):
    """
    Iterative CSG boolean intersection
    """
    global debug, count, limit, node, action, tl, nl, tr, nr, tmin, lname, rname

    count = 0 

    #if node.is_primitive:
    #    return intersect_primitive(node, ray, tmin)
    #else:
    if True:
        pushAction(Compute)
        action = GotoLft

        while count < limit:
            count += 1 

            #if debug:
            #    log.info("[%d] ITERATIVE while node %r " % (count,node) )

            if action == SaveLft:
                tmp = popTmin()
                if debug:
                   log.info("(SaveLft) popTmin setting tmin %5.2f -> %5.2f " % (tmin, tmp))

                tmin = tmp
                pushPrimitive((tl,nl,lname))
                action = GotoRgh
            pass
            if action == GotoLft or action == GotoRgh:
                GoTo()
            pass
            if action == LoadLft or action == LoadRgh or action == Compute:
                Compute_()
            pass

            #if debug:
            #    log.info("[%d] ITERATIVE while tail node %r " % (count, node) )

            if node is None or node.right is None:
                return tl, nl, lname

            #if node is None:
            #    #log.info(" None at node %r tl/tr %s %s nl/nr %r/%r " % ( node, tl, tr, nl, nr ))
            #    #assert tl == tr, (tl, tr) 
            #    #assert np.allclose(nl, nr)
            #    return tl, nl 
            #pass
        pass
    pass
    return None,None,None 


def GoTo():
    global debug, count, node, action, tl, nl, tr, nr, tmin, lname, rname
    assert action in [GotoLft, GotoRgh] 
    pnode = node
    node = node.left if action == GotoLft else node.right 

    #if debug:
    #    log.info("[%d] ITERATIVE %s -> %s  " % ( count, action_desc(action), node ))

    if node is None:
       log.fatal("node None after action %s from parent %r " % (action_desc(action),pnode) )
       return 

    if node.is_operation:
        gotoL = intersectBox(node.left)
        gotoR = intersectBox(node.right)
 
        if gotoL and node.left.is_primitive:
            tl, nl, lname = intersect_primitive(node.left, ray, tmin)
            gotoL = False

        if gotoR and node.right.is_primitive:
            tr, nr, rname = intersect_primitive(node.right, ray, tmin)
            gotoR = False

        log.debug("gotoL %s gotoR %s " % (gotoL, gotoR ))

        if gotoL or gotoR:
            if gotoL:
                pushPrimitive((tl, nl, lname), label="gotoL")
                pushAction(LoadLft, label="gotoL")
                action = GotoLft
            elif gotoR:
                pushPrimitive((tr, nr, rname), label="gotoR")
                pushAction(LoadRgh, label="gotoR")
                action = GotoRgh
            pass
        else:
            # both gotoL and gotoR False means miss OR both prim intersects done, so are ready for compute
            pushTmin(tmin)
            pushAction(LoadLft, label="no-goto")
            pushAction(SaveLft, label="no-goto")
            action = Compute 
        pass
        log.debug(dump("node.is_operation conclusion %s " % action_desc(action)))

    else:   # node is a Primitive

        if action ==  GotoLft:
            tl, nl, lname = intersect_primitive(node, ray, tmin)
            
        else:
            tr, nr, rname = intersect_primitive(node, ray, tmin)

        action = Compute
        node = node.parent



def trep():
    global tl, tr, tmin 
    return "tmin/tl/tr %5.2f %5.2f %5.2f " % (tmin if tmin else -1, tl if tl else -1, tr if tr else -1 )


def Compute_():
    global debug, count, node, action, tl, nl, tr, nr, tmin, lname, rname
    dump("Compute_")
    assert node.is_operation

    if action == LoadLft or action == LoadRgh:
        if action == LoadLft:
            lll = popPrimitive()
            assert len(lll) == 3, lll
            tl, nl, lname = lll
        else:
            tr, nr, rname = popPrimitive()
        pass

    stateL = classify(tl,nl)
    stateR = classify(tr,nr)

    #if debug:
    #    log.info("[%d] ITERATIVE %r -> %s op %d " % ( count, node, desc[node.operation], node.operation ))

    acts = table(node.operation, stateL, stateR )

    if debug:
        log.info("[%d] ITERATIVE %s(%s:%s,%s:%s) -> %s: %s" % ( count, desc[node.operation],lname,desc_state[stateL], rname,desc_state[stateR], desc_acts(acts), trep() ))

    if (RetMiss & acts):
        tr = None
        action = Compute # popAction()  ################### ???????????
        node = node.parent
    elif (RetL & acts) or ((RetLIfCloser & acts) and tl <= tr): 
        if debug:
            log.info("[%d] ITERATIVE RetL/RetLIfCloser : %s" % (count, trep()))
        tr = tl
        nr = nl
        action = popAction()
        node = node.parent
    elif (RetR & acts) or ((RetRIfCloser & acts) and tr < tl): 
        if debug:
            log.info("[%d] ITERATIVE RetR/RetRIfCloser : %s" % (count, trep()))
        if (FlipR & acts): nr = -nr
        tl = tr
        nl = nr
        action = popAction()
        node = node.parent
    elif (LoopL & acts) or ((LoopLIfCloser & acts) and tl <= tr):
        if debug:
            log.info("[%d] ITERATIVE LoopL/LoopLIfCloser : %s" % (count, trep()))
        tmin = tl
        pushPrimitive((tr,nr,rname))
        pushAction(LoadRgh)
        action = GotoLft
    elif (LoopR & acts) or ((LoopRIfCloser & acts) and tr < tl):
        if debug:
            log.info("[%d] ITERATIVE LoopR/LoopRIfCloser : %s" % (count, trep()))
        tmin = tr
        pushPrimitive((tl,nl,lname))
        pushAction(LoadLft)
        action = GotoRgh
    else:
        assert 0



def reset_globals(debug_list=[]):
    global debug, count, limit, node, iray, action, tl, nl, tr, nr, tmin, lname, rname
    debug = True if iray in debug_list else False
    limit = 20
    count = 0 
    node = None
    action = None
    tl, nl = None, None
    tr, nr = None, None
    tmin = 0 
    lname = "?"
    rname = "?"



def test_intersect(tst):
    """
    * top virtual node distinguishable as an "operation" but node.right is None 

    * only 1st intersect is returned, so to see inside and outside of   
      a shape need to send rays from inside and outside


    Union of two offset spheres, OK from aringlight but not origlight


    """
    global debug, count, limit, node, ray, iray, action, tl, nl, tr, nr, tmin 

    virtual_root = Node(left=tst.root,right=None,operation=UNION)   

    rays = []

    if "aringlight" in tst.source:
        ary = Ray.aringlight(num=tst.num, radius=1000)
        rays += Ray.make_rays(ary)
    pass 

    if "origlight" in tst.source:
        rays += Ray.origlight(num=tst.num)
    pass

    if "lsquad" in tst.source:
        rays += [Ray(origin=[-300,y,0], direction=[1,0,0]) for y in range(-50,50+1,10)]
    pass

    ipos = np.zeros((2,len(rays), 3), dtype=np.float32 ) 
    ndir = np.zeros((2,len(rays), 3), dtype=np.float32 ) 
    tval = np.zeros((2,len(rays)), dtype=np.float32 )

    prob = []
    for iray, ray in enumerate(rays):

        for recursive in [0,1]:
            reset_globals(tst.debug_list)

            if debug:
                log.info(" ray(%d) %r " % (iray,ray) ) 

            if recursive:
                node = virtual_root.left
                tt, nn, nname = rintersect_node(node, ray)
            else:
                node = virtual_root  
                tmp = iintersect_node(ray)   
                assert len(tmp) == 3, tmp
                tt, nn, nname = tmp
            pass
            typ = "RECURSIVE" if recursive else "ITERATIVE"
            if debug:
                log.info("[%d] %s intersect %r tt %s nn %r " % (count, typ, ray, tt, nn ))
            if not tt is None:
                ipos[recursive,iray] = ray.position(tt)
                ndir[recursive,iray] = nn
                tval[recursive,iray] = tt
            pass
        pass

        ok_pos = np.allclose( ipos[0,iray], ipos[1,iray] )
        ok_dir = np.allclose( ndir[0,iray], ndir[1,iray] )
        ok_tva = np.allclose( tval[0,iray], tval[1,iray] )

        if not (ok_pos and ok_dir and ok_tva):
            prob.append(iray)
        pass
    pass

    if len(prob) > 0:
        log.warning("%10s %d/%d rays with mismatches : %s " % (tst.name, len(prob),len(rays),repr(prob)))
    else:
        log.info("%10s %d/%d rays with mismatches : %s " % (tst.name, len(prob),len(rays),repr(prob)))

    sc = 10 
    for recursive in [0, 1]:
        xoff = 600 if recursive else 0
        plt.scatter( xoff + ipos[recursive,:,0]                        , ipos[recursive,:,1] )
        #plt.scatter( xoff + ipos[recursive,:,0]+ndir[recursive,:,0]*sc , ipos[recursive,:,1]+ndir[recursive,:,1]*sc )
        #plt.scatter( xoff + ary[:,0,0], ary[:,0,1] )

        if len(prob) > 0:
            plt.scatter( xoff + ipos[recursive, prob,0], ipos[recursive, prob,1], c="r" )
            plt.scatter( xoff + ipos[recursive, prob,0], ipos[recursive, prob,1], c="g" )


    plt.show()

    return prob, tval, ipos, ndir



class T(object):
    def __init__(self, root, name, debug_list=[], notes="", source="aringlight,origlight", num=200):
        self.root = root
        self.name = name
        self.debug_list = debug_list
        self.notes = notes
        self.source = source
        self.num = num



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
    lrbox = Node(lbox.clone(),  rbox.clone(), UNION, name="lrbox") 

    bms = Node(Node(BOX, param=[0,0,0,200], name="bms_box"),  Node(SPHERE,param=[0,0,0,150],name="bms_sph"), DIFFERENCE, name="bms")
    smb = Node(Node(SPHERE,param=[0,0,0,200]), Node(BOX,param=[0,0,0,150]), DIFFERENCE , name="smb")
    ubo = Node(bms.clone(), lrbox.clone(), UNION , name="ubo")
    bmslrbox = Node( Node(bms.clone(), rbox.clone(), UNION,name="bms_rbox_u"),lbox.clone(),UNION, name="bmslrbox" ) 
    bmsrlbox = Node( Node(bms.clone(), lbox.clone(), UNION,name="bms_lbox_u"),rbox.clone(),UNION, name="bmsrlbox" ) 

    csph = Node(SPHERE, param=[0,0,0,100], name="csph")
    lsph = Node(SPHERE, param=[-50,0,0,100], name="lsph")
    rsph = Node(SPHERE, param=[50,0,0,100], name="rsph")

    lrsph_u = Node(lsph.clone(), rsph.clone(), UNION, name="lrsph_u")
    lrsph_i = Node(lsph.clone(), rsph.clone(), INTERSECTION, name="lrsph_i")
    lrsph_d = Node(lsph.clone(), rsph.clone(), DIFFERENCE , name="lrsph_d")

    ok = [ 
             T(smb, "smb"),
             T(bms, "bms"),
             T(csph, "csph"),
             T(cbox, "cbox"),
             T(lbox, "lbox"),
             T(rbox, "rbox"),
             T(lrbox, "lrbox"),
             T(lrsph_d, "lrsph_d"),
             T(lrsph_u, "lrsph_u", notes="fixed all rightside mismatched with origlight by adopting clone to avoid inadventent parent relationship to other shape"),
             T(lrsph_i, "lrsph_i"),
         ]

    nok = [
             T(bmslrbox, "bmslrbox", notes="left box protrusion is missed for iterative", source="lsquad", debug_list=[1]),
             #T(bmsrlbox, "bmsrlbox", notes="right box protrusion is missed for iterative"),
             #T(ubo, "ubo", [], notes="looks to be missing most intersects???"),
          ]

    for tst in nok:
        prob, tval, ipos, ndir = test_intersect(tst)


