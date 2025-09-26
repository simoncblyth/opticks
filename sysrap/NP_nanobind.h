#pragma once
/**
NP_nanobind.h
==============


**/

#include "NP.hh"
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

struct NP_nanobind
{
    static NP*  NP_copy_of_numpy_array(          nanobind::ndarray<nanobind::numpy> a) ;
    static NP*  NP_copy_of_numpy_array_with_meta(nanobind::ndarray<nanobind::numpy> a, nanobind::str meta) ;

    static nanobind::dlpack::dtype      dtype_for_NP(char uifc, size_t ebyte);
    static nanobind::capsule*           owner_for_NP(char uifc, size_t ebyte, void* data);

    static nanobind::ndarray<nanobind::numpy> numpy_array_view_of_NP(const NP* a);
    static nanobind::tuple                    numpy_array_view_of_NP_with_meta(const NP* a );

    static nanobind::ndarray<nanobind::numpy> example_numpy_array_view_of_NP(int code);

    static nanobind::ndarray<nanobind::numpy> roundtrip_numpy_array_via_NP(nanobind::ndarray<nanobind::numpy> src);
};


/**
NP_copy_of_numpy_array : python -> C++
---------------------------------------------

Currently just copying. Can that be avoided ?

**/

inline NP* NP_nanobind::NP_copy_of_numpy_array(nanobind::ndarray<nanobind::numpy> _a) // static
{
    void* data = _a.data();
    size_t ndim = _a.ndim();
    nanobind::dlpack::dtype _dtype = _a.dtype();

    std::string dtype ;
    if(      _dtype == nanobind::dtype<float>() )        dtype = descr_<float>::dtype()    ;
    else if( _dtype == nanobind::dtype<double>())        dtype = descr_<double>::dtype()   ;
    else if( _dtype == nanobind::dtype<double>())        dtype = descr_<double>::dtype()   ;
    else if( _dtype == nanobind::dtype<int>())           dtype = descr_<int>::dtype()      ;
    else if( _dtype == nanobind::dtype<long>())          dtype = descr_<long>::dtype()     ;
    else if( _dtype == nanobind::dtype<unsigned>())      dtype = descr_<unsigned>::dtype() ;
    else if( _dtype == nanobind::dtype<unsigned long>()) dtype = descr_<unsigned long>::dtype() ;

    std::vector<NP::INT> shape(ndim);
    for(size_t i=0 ; i < ndim ; i++ ) shape[i] = _a.shape(i) ;

    NP* a = new NP(dtype.c_str(), shape );
    assert( a->uarr_bytes() == _a.nbytes() );
    a->read_bytes( (char*)data );

    return a ;
}

inline NP* NP_nanobind::NP_copy_of_numpy_array_with_meta(nanobind::ndarray<nanobind::numpy> _a, nanobind::str _meta) // static
{
    NP* a = NP_copy_of_numpy_array(_a);
    a->meta = nanobind::cast<std::string>(_meta);
    return a ;
}



inline nanobind::dlpack::dtype NP_nanobind::dtype_for_NP(char uifc, size_t ebyte) // static
{
    nanobind::dlpack::dtype dtype ;
    if(      uifc == 'f' && ebyte == 4 ) dtype = nanobind::dtype<float>();
    else if( uifc == 'f' && ebyte == 8 ) dtype = nanobind::dtype<double>();
    else if( uifc == 'u' && ebyte == 4 ) dtype = nanobind::dtype<unsigned>();
    else if( uifc == 'u' && ebyte == 8 ) dtype = nanobind::dtype<unsigned long>();
    else if( uifc == 'i' && ebyte == 4 ) dtype = nanobind::dtype<int>();
    else if( uifc == 'i' && ebyte == 8 ) dtype = nanobind::dtype<long>();
   return dtype ;
}

inline nanobind::capsule* NP_nanobind::owner_for_NP(char uifc, size_t ebyte, void* data) // static
{
    nanobind::capsule* owner = nullptr ;
    if(      uifc == 'f' && ebyte == 4 ) owner = new nanobind::capsule(data, [](void *p) noexcept { delete[] (float *)p ; });
    else if( uifc == 'f' && ebyte == 8 ) owner = new nanobind::capsule(data, [](void *p) noexcept { delete[] (double *)p ; });
    else if( uifc == 'u' && ebyte == 4 ) owner = new nanobind::capsule(data, [](void *p) noexcept { delete[] (unsigned *)p ; });
    else if( uifc == 'u' && ebyte == 8 ) owner = new nanobind::capsule(data, [](void *p) noexcept { delete[] (unsigned long *)p ; });
    else if( uifc == 'i' && ebyte == 4 ) owner = new nanobind::capsule(data, [](void *p) noexcept { delete[] (int *)p ; });
    else if( uifc == 'i' && ebyte == 8 ) owner = new nanobind::capsule(data, [](void *p) noexcept { delete[] (long *)p ; });
    return owner ;
}


/**
numpy_array_view_of_NP : C++ -> python
--------------------------------------------

No copying, just adopts the same data pointer.




**/

inline nanobind::ndarray<nanobind::numpy> NP_nanobind::numpy_array_view_of_NP(const NP* a) // static
{
    void* data = (void*)a->bytes() ;
    std::vector<size_t> sh ;
    a->get_shape(sh);
    size_t ndim = sh.size();
    const size_t* shape = sh.data();
    nanobind::capsule* owner = owner_for_NP(a->uifc, a->ebyte, data);
    const int64_t* strides = nullptr ;
    nanobind::dlpack::dtype dtype = dtype_for_NP(a->uifc, a->ebyte);
    int device_type = nanobind::device::cpu::value ;
    int device_id = 0 ;
    char order = 'C' ;
    return nanobind::ndarray<nanobind::numpy>(data, ndim, shape, *owner, strides, dtype, device_type, device_id, order);
}

inline nanobind::tuple NP_nanobind::numpy_array_view_of_NP_with_meta(const NP* a )
{
    nanobind::ndarray<nanobind::numpy> arr = numpy_array_view_of_NP(a);
    return nanobind::make_tuple(arr, a->meta );
}


inline nanobind::ndarray<nanobind::numpy> NP_nanobind::example_numpy_array_view_of_NP(int code) // static
{
    NP* a = nullptr ;
    switch(code)
    {
        case 0: a = NP::Make<float>(3,6,4);         break ;
        case 1: a = NP::Make<double>(3,6,4);        break ;
        case 2: a = NP::Make<int>(3,6,4);           break ;
        case 3: a = NP::Make<long>(3,6,4);          break ;
        case 4: a = NP::Make<unsigned>(3,6,4);      break ;
        case 5: a = NP::Make<unsigned long>(3,6,4); break ;
    }
    a->fillIndexFlat();
    return numpy_array_view_of_NP(a);
}


inline nanobind::ndarray<nanobind::numpy> NP_nanobind::roundtrip_numpy_array_via_NP(nanobind::ndarray<nanobind::numpy> src) // static
{
    NP* a_src = NP_copy_of_numpy_array(src);
    nanobind::ndarray<nanobind::numpy> a_dst = numpy_array_view_of_NP(a_src) ;
    return a_dst ;
}



