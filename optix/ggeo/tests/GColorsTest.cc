#include <cstdlib>
#include <cassert>

#include <vector>


#include "NGLM.hpp"
#include "NPY.hpp"

#include "OpticksColors.hh"

#include "PLOG.hh"

// see tests/GColorsTest.py 


int main(int argc, char** argv)
{
    PLOG_(argc, argv);    


    OpticksColors* m_colors = OpticksColors::load("$HOME/.opticks/GCache","GColors.json");
    assert(m_colors);

    m_colors->dump();
    m_colors->test(); 

    std::vector<unsigned int> material_codes ; 
    std::vector<unsigned int> flag_codes ; 

    // canonically done in GLoader
    m_colors->setupCompositeColorBuffer(material_codes, flag_codes);   
    
    //GBuffer* m_color_buffer = m_colors->getCompositeBuffer(); 

    NPY<unsigned char>* m_color_buffer = m_colors->getCompositeBuffer(); 


    // really its uchar4 but aoba wull probably not handle that, so use unsigned int
    assert(sizeof(unsigned char)*4 == sizeof(unsigned int));

    //m_color_buffer->save<unsigned int>("/tmp/colors_GBuffer.npy");

    m_color_buffer->Summary();

    m_color_buffer->save("/tmp/colors_NPY.npy");


}


/*

color_dump 63 : 137 137 137 137 
color_dump 64 : 0 0 0 0 
color_dump 65 : 0 0 129 0 
color_dump 66 : 0 0 140 0 
color_dump 67 : 0 0 206 0 
color_dump 68 : 0 0 0 0 
color_dump 69 : 0 201 0 0 
color_dump 70 : 0 129 0 0 
color_dump 71 : 0 129 129 0 
color_dump 72 : 0 140 140 0 
color_dump 73 : 0 192 0 0 
color_dump 74 : 0 207 210 0 
color_dump 75 : 0 251 155 0 
color_dump 76 : 0 0 0 0 
color_dump 77 : 0 0 255 0 
color_dump 78 : 0 0 0 0 
color_dump 79 : 0 0 0 0 
color_dump 80 : 201 201 225 0 
color_dump 81 : 241 145 0 0 
color_dump 82 : 129 179 171 0 
color_dump 83 : 137 140 137 0 
color_dump 84 : 185 140 175 0 
color_dump 85 : 189 159 159 0 
color_dump 86 : 189 159 159 0 
color_dump 87 : 201 206 201 0 
color_dump 88 : 241 180 227 0 
color_dump 89 : 129 225 209 0 
color_dump 90 : 131 211 226 0 
color_dump 91 : 141 131 181 0 
color_dump 92 : 145 245 140 0 
color_dump 93 : 145 210 205 0 
color_dump 94 : 151 0 131 0 
color_dump 95 : 171 215 189 0 
color_dump 96 : 191 159 161 0 
color_dump 97 : 201 150 238 0 
color_dump 98 : 205 206 171 0 
color_dump 99 : 211 211 211 0 
color_dump 100 : 211 211 211 0 
color_dump 101 : 213 181 206 0 
color_dump 102 : 215 143 141 0 
color_dump 103 : 225 129 145 0 
color_dump 104 : 225 129 145 0 
color_dump 105 : 239 137 154 0 
color_dump 106 : 239 137 154 0 


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
