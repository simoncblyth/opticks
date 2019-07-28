#!/usr/bin/env python


def RSAVE(slot, bounce):
    print( " slot %2d   bounce  %2d " % ( slot, bounce) )
pass

bounce_max = 9 
bounce = 0 
slot = 0 


while bounce < bounce_max:
    bounce += 1 
    RSAVE(slot, bounce)
    slot += 1 
pass

RSAVE(slot, bounce)



