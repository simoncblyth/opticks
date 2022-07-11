#pragma once

struct SRandom
{
    virtual int    getFlatCursor() const = 0 ; 
    virtual double getFlatPrior() const = 0 ; 

};



