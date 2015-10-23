#include "GColors.hh"
#include "stdlib.h"

int main(int argc, char** argv)
{
    GColors* m_colors = GColors::load("$HOME/.opticks","GColors.json");
    m_colors->dump();
    m_colors->test(); 


    GItemIndex* materials = NULL ; 
    GItemIndex* surfaces  = NULL ; 
    GItemIndex* flags = NULL ; 

    // canonically done in GLoader
    m_colors->setupCompositeColorBuffer(materials, surfaces, flags);   
    GBuffer* m_color_buffer = m_colors->getCompositeBuffer(); 

    // really its uchar4 but aoba wull probably not handle that, so use unsigned int
    assert(sizeof(unsigned char)*4 == sizeof(unsigned int));

    m_color_buffer->save<unsigned int>("/tmp/colors.npy");

    m_color_buffer->Summary();


}


/*

::

    In [1]: c = np.load("/tmp/colors.npy")

    In [2]: c.shape
    Out[2]: (256, 1)

    In [3]: c
    Out[3]: 
    array([[1145324612],
           [1145324612],
           [1145324612],
           [1145324612],
    ...
           [1145324612],
           [1145324612],
           [1145324612]], dtype=uint32)


View as smaller type splits the items::

    In [4]: u = c.view(np.uint8)

    In [5]: u.shape
    Out[5]: (256, 4)

    In [6]: u
    Out[6]: 
    array([[68, 68, 68, 68],
           [68, 68, 68, 68],
           [68, 68, 68, 68],
           ..., 
           [68, 68, 68, 68],
           [68, 68, 68, 68],
           [68, 68, 68, 68]], dtype=uint8)

    In [7]: hex(68)
    Out[7]: '0x44'

*/
