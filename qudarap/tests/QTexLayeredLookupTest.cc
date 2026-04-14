#include "NP.hh"
#include "QTexLayered.h"
#include "QTexLayeredLookup.h"

/**

In [5]: f.origin[0].reshape(5,10)
Out[5]:
array([[  0.,  10.,  20.,  30.,  40.,  50.,  60.,  70.,  80.,  90.],
       [100., 110., 120., 130., 140., 150., 160., 170., 180., 190.],
       [200., 210., 220., 230., 240., 250., 260., 270., 280., 290.],
       [300., 310., 320., 330., 340., 350., 360., 370., 380., 390.],
       [400., 410., 420., 430., 440., 450., 460., 470., 480., 490.]], dtype=float32)

In [6]: f.origin[1].reshape(5,10)
Out[6]:
array([[1000., 1010., 1020., 1030., 1040., 1050., 1060., 1070., 1080., 1090.],
       [1100., 1110., 1120., 1130., 1140., 1150., 1160., 1170., 1180., 1190.],
       [1200., 1210., 1220., 1230., 1240., 1250., 1260., 1270., 1280., 1290.],
       [1300., 1310., 1320., 1330., 1340., 1350., 1360., 1370., 1380., 1390.],
       [1400., 1410., 1420., 1430., 1440., 1450., 1460., 1470., 1480., 1490.]], dtype=float32)


In [7]: f.origin.shape
Out[7]: (2, 5, 10, 1)

In [8]: f.lookup.shape
Out[8]: (2, 5, 10, 1)

**/

NP* make_array_1(unsigned nl, unsigned ni, unsigned nj, unsigned nk)
{
    std::vector<float> src ;

    for(unsigned l=0 ; l < nl ; l++)
    for(unsigned i=0 ; i < ni ; i++)
    for(unsigned j=0 ; j < nj ; j++)
    for(unsigned k=0 ; k < nk ; k++)
    {
        float val = float(l*1000 + i*100 + j*10 + k) ;
        src.push_back(val);
    }

    NP* a = NP::Make<float>( nl, ni, nj, nk );
    a->read(src.data()) ;

    return a ;
}

NP* make_array_2(unsigned nl, unsigned ni, unsigned nj, unsigned nk)
{
    // bit perfect equivalent to make_array_1
    NP* a = NP::Make<float>( nl, ni, nj, nk );
    float* aa = a->values<float>();

    for(unsigned l=0 ; l < nl ; l++)
    for(unsigned i=0 ; i < ni ; i++)
    for(unsigned j=0 ; j < nj ; j++)
    for(unsigned k=0 ; k < nk ; k++)
    {
        unsigned index = l*ni*nj*nk + i*nj*nk + j*nk + k ;
        float value = float(l*1000 + i*100 + j*10 + k) ;
        aa[index] = value ;
    }
    return a ;
}

int main()
{
    unsigned nl = 2 ;   // layers
    unsigned ny = 5 ;   // height
    unsigned nx = 10 ;  // width
    unsigned np = 1 ;   // payload

    const NP* origin = make_array_2( nl, ny, nx, np );
    char filterMode = 'P' ; // point for roundtripping the texture

    QTexLayered<float>* tex = new QTexLayered<float>(origin, filterMode );
    tex->uploadMeta();

    QTexLayeredLookup<float> look(tex);
    NP* out = look.lookup();

    out->save("$FOLD/lookup.npy");
    origin->save("$FOLD/origin.npy");

    return 0 ;
}

