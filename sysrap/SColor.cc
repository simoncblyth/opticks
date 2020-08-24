#include "SColor.hh"

const SColor SColors::red = { 0xff, 0x00, 0x00 };
const SColor SColors::green = { 0x00, 0xff, 0x00 };
const SColor SColors::blue  = { 0x00, 0x00, 0xff };
const SColor SColors::cyan   = { 0x00, 0xff, 0xff };
const SColor SColors::magenta = { 0xff, 0x00, 0xff };
const SColor SColors::yellow  = { 0x00, 0xff, 0xff };
const SColor SColors::white = { 0xff, 0xff, 0xff };
const SColor SColors::black = { 0x00, 0x00, 0x00 };

unsigned char SColor::get(unsigned i) const 
{
    unsigned char c = 0xff ; 
    switch(i)
    {
        case 0: c = r ; break ; 
        case 1: c = g ; break ; 
        case 2: c = b ; break ; 
    }
    return c ; 
}


