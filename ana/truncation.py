#!/usr/bin/env python
"""
truncation.py
===============

Pseudo duplication of oxrap/cu/generate.cu for easy thinking about truncation, 
and implementing seqhis steered propagations.

record_max
     has a hard limit at 16, from 4bits*16 = 64 bits  
     hopefully before this becomes a problem GPU/CUDA/OptiX 
     will natively support 128bit uints
 
     There are workarounds using uint4 and inline PTX asm assembly however 

     * :google:`cuda 128 bit int`
     * http://stackoverflow.com/questions/6162140/128-bit-integer-on-cuda

bounce_max
     is more flexible, but for clean truncation using bounce_max = record_max - 1  
     is highly favored (ie maximally 15)

bounce
     not really the right good word as SC,AB,RE dont involve a "bounce", 
     trace is more appropriate as a raytrace is always done for every turn of the loop  


Why use bounce at all, why not just use slot in the loop ?
   as step-by-step recording is just a temporary measure whilst debugging,   
   the real thing in the photon that gets recorded after the bounce loop


"""
import logging
log = logging.getLogger(__name__)
index = 0 

BREAK = "BREAK"
CONTINUE = "CONTINUE"
PASS = "PASS"

SEQHIS = "TO BT BT SC BT BT SA"
SQ = SEQHIS.split()
tru = 0 

def truth():
    global tru
    try:
       sq = SQ[tru]
    except IndexError:
       sq = None
    pass
    tru += 1 
    return sq 


def RSAVE(msg, slot, slot_offset, bounce):
    global index  
    log.info(" [%2d] (%10s) slot %2d slot_offset %2d bounce %2d " % (index, msg, slot, slot_offset, bounce))
    index += 1 


def propagate_to_boundary(sq):
    if sq == "AB":
        rc = BREAK 
    elif sq == "RE" or sq == "SC":
        rc = CONTINUE
    else:
        rc = PASS
    pass
    log.info(" propagate_to_boundary sq %s rc %s " % (sq, rc) )
    return rc 

 
def propagate_at_surface(sq):
    if sq in "SD SA".split():
        rc = BREAK
    elif sq in "DR SR".split():
        rc = CONTINUE 
    else:
        assert False, sq 
         
    log.info(" propagate_at_surface sq %s rc %s " % (sq, rc) )
    return rc 

def propagate_at_boundary(sq):
    assert sq in "BR BT".split()
    rc = CONTINUE
    log.info(" propagate_at_boundary sq %s rc %s " % (sq, rc) )
    return rc


def generate():
    record_max = 16 
    bounce_max = 15
    MAXREC = record_max 

    photon_id = 1000 
    slot_min = photon_id*MAXREC
    slot_max = slot_min + MAXREC - 1
    log.info("photon_id %7d slot_min %2d slot_max %2d " % (photon_id, slot_min, slot_max))

    bounce = 0 
    slot = 0 

    flag = truth()
    assert flag == 'TO'


    while bounce < bounce_max:
        bounce += 1

        # raytrace happens here 

        if slot < MAXREC: 
            slot_offset = slot_min + slot 
        else:
            slot_offset = slot_max ;
        pass
        RSAVE("inside", slot, slot_offset, bounce)
        slot += 1

        flag = truth()
        
        command = propagate_to_boundary(flag)
        if command == BREAK: break
        if command == CONTINUE: continue
        if command == PASS: pass

        if flag in "SD SA DR SR".split():
            command = propagate_at_surface(flag)
            if command == BREAK: break
            if command == CONTINUE: continue
        elif flag in "BR BT".split():
            propagate_at_boundary(flag)
        else:
            assert 0, flag
    pass

    if slot < MAXREC: 
        slot_offset = slot_min + slot 
    else:
        slot_offset = slot_max ;
    pass
    RSAVE("after", slot, slot_offset, bounce)
    pass




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate()

