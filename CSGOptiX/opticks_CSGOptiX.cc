/**
opticks_CSGOptiX.cc
=====================

https://nanobind.readthedocs.io/en/latest/basics.html#basics
https://nanobind.readthedocs.io/en/latest/ndarray.html


Experiement with C++ Python binding with nanobind

TODO:

* move NP mechanics into separate header in sysrap (actually np repo makes more sense) 
  as unreasonable to live up here, should be down at minimum dependency 

**/


#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/ndarray.h>

#include "CSGOptiXService.h"


namespace nb = nanobind;
using namespace nb::literals;


using RGBImage = nb::ndarray<uint8_t, nb::shape<-1, -1, 3>, nb::device::cpu>;

void process3(RGBImage data)
{
    // treble brightness of the MxNx3 RGB image
    for (size_t y = 0; y < data.shape(0); ++y)
        for (size_t x = 0; x < data.shape(1); ++x)
            for (size_t ch = 0; ch < 3; ++ch)
                data(y, x, ch) = (uint8_t) std::min(255, data(y, x, ch) * 3);
}



nb::ndarray<nb::numpy, float> create_3d(size_t rows, size_t cols, size_t depth)
{
    size_t sz = rows * cols * depth ;
    float* data = new float[sz];
    for (size_t i = 0; i < sz; ++i) data[i] = (float)i;

    // delete data when owner capsule expires
    nb::capsule owner(data, [](void *p) noexcept { delete[] (float *)p ; });

    std::initializer_list<size_t> shape = {rows, cols, depth };

    return nb::ndarray<nb::numpy, float>( data, shape, owner );
}

/**
create_NP_from_numpy_array : python -> C++
---------------------------------------------

Currently just copying. Can that be avoided ?

**/


NP* create_NP_from_numpy_array(nb::ndarray<nb::numpy>& a)
{
    void* data = a.data();
    size_t ndim = a.ndim();
    nb::dlpack::dtype a_dtype = a.dtype();

    std::string dtype ;
    if(      a_dtype == nb::dtype<float>() )        dtype = descr_<float>::dtype()    ;
    else if( a_dtype == nb::dtype<double>())        dtype = descr_<double>::dtype()   ;
    else if( a_dtype == nb::dtype<double>())        dtype = descr_<double>::dtype()   ;
    else if( a_dtype == nb::dtype<int>())           dtype = descr_<int>::dtype()      ;
    else if( a_dtype == nb::dtype<long>())          dtype = descr_<long>::dtype()     ;
    else if( a_dtype == nb::dtype<unsigned>())      dtype = descr_<unsigned>::dtype() ;
    else if( a_dtype == nb::dtype<unsigned long>()) dtype = descr_<unsigned long>::dtype() ;

    std::vector<NP::INT> shape(ndim);
    for(size_t i=0 ; i < ndim ; i++ ) shape[i] = a.shape(i) ;

    NP* n = new NP(dtype.c_str(), shape );
    assert( n->uarr_bytes() == a.nbytes() );
    n->read_bytes( (char*)data );

    return n ;
}


/**
create_numpy_array_from_NP : C++ -> python
--------------------------------------------

No copying, just adopts the same data pointer.

**/

nb::ndarray<nb::numpy> create_numpy_array_from_NP(const NP* a)
{
    void* data = (void*)a->bytes() ;
    char x = a->uifc ;
    size_t b = a->ebyte ;

    std::vector<size_t> sh ;
    a->get_shape(sh);
    size_t ndim = sh.size();
    const size_t* shape = sh.data();

    const int64_t* strides = nullptr ;

    int device_type = nb::device::cpu::value ;
    int device_id = 0 ;
    char order = 'C' ;

    nb::dlpack::dtype dtype ;
    nb::capsule* owner = nullptr ;

    if( x == 'f' && b == 4 )
    {
        dtype = nb::dtype<float>();
        owner = new nb::capsule(data, [](void *p) noexcept { delete[] (float *)p ; });
    }
    else if( x == 'f' && b == 8 )
    {
        dtype = nb::dtype<double>();
        owner = new nb::capsule(data, [](void *p) noexcept { delete[] (double *)p ; });
    }
    else if( x == 'u' && b == 4 )
    {
        dtype = nb::dtype<unsigned>();
        owner = new nb::capsule(data, [](void *p) noexcept { delete[] (unsigned *)p ; });
    }
    else if( x == 'u' && b == 8 )
    {
        dtype = nb::dtype<unsigned long>();
        owner = new nb::capsule(data, [](void *p) noexcept { delete[] (unsigned long *)p ; });
    }
    else if( x == 'i' && b == 4 )
    {
        dtype = nb::dtype<int>();
        owner = new nb::capsule(data, [](void *p) noexcept { delete[] (int *)p ; });
    }
    else if( x == 'i' && b == 8 )
    {
        dtype = nb::dtype<long>();
        owner = new nb::capsule(data, [](void *p) noexcept { delete[] (long *)p ; });
    }
    return nb::ndarray<nb::numpy>(data, ndim, shape, *owner, strides, dtype, device_type, device_id, order);
}

nb::ndarray<nb::numpy> create_from_NP(int code)
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
    return create_numpy_array_from_NP(a);
}


nb::ndarray<nb::numpy> roundtrip_numpy_array_via_NP(nb::ndarray<nb::numpy>& src)
{
    NP* a_src = create_NP_from_numpy_array(src);
    nb::ndarray<nb::numpy> a_dst = create_numpy_array_from_NP(a_src) ;
    return a_dst ;
}





// module name must match first arg to nanobind_add_module in CMakeLists.txt
NB_MODULE(opticks_CSGOptiX, m)
{
    //m.def("add", &add, "a"_a, "b"_a = 1, "Adds two numbers and increments if only one is provided.");
    m.attr("the_answer") = 42;
    m.doc() = "A simple example python extension";

    nb::class_<Dog>(m, "Dog")
        .def(nb::init<>())
        .def(nb::init<const std::string &>())
        .def("bark", &Dog::bark)
        .def_rw("name", &Dog::name)
        .def("__repr__", [](const Dog &p) { return "<my_ext.Dog named '" + p.name + "'>"; })
        .def("bark_later", [](const Dog &p) {
                 auto callback = [name = p.name] { nb::print(nb::str("{}: woof!").format(name));};
                 return nb::cpp_function(callback);
         })
         ;


    nb::class_<CSGOptiXService>(m,"CSGOptiXService")
        .def(nb::init<>())
        .def("__repr__", [](const CSGOptiXService& svc) { return svc.desc(); })
        ;

    m.def("inspect", [](const nb::ndarray<>& a) {
        printf("Array data pointer : %p\n", a.data());
        printf("Array dimension : %zu\n", a.ndim());
        for (size_t i = 0; i < a.ndim(); ++i) {
            printf("Array dimension [%zu] : %zu\n", i, a.shape(i));
            printf("Array stride    [%zu] : %ld\n", i, a.stride(i));
        }
        printf("Device ID = %u (cpu=%i, cuda=%i)\n", a.device_id(),
            int(a.device_type() == nb::device::cpu::value),
            int(a.device_type() == nb::device::cuda::value)
        );
        printf("Array dtype: int16=%i, uint32=%i, float32=%i\n",
            a.dtype() == nb::dtype<int16_t>(),
            a.dtype() == nb::dtype<uint32_t>(),
            a.dtype() == nb::dtype<float>()
        );
    });

    m.def("process", [](RGBImage data) {
        // Double brightness of the MxNx3 RGB image
        for (size_t y = 0; y < data.shape(0); ++y)
            for (size_t x = 0; x < data.shape(1); ++x)
                for (size_t ch = 0; ch < 3; ++ch)
                    data(y, x, ch) = (uint8_t) std::min(255, data(y, x, ch) * 2);
    });

    m.def("process3", &process3 );

    m.def("create_3d", &create_3d );

    m.def("create_from_NP", &create_from_NP );

    m.def("roundtrip_numpy_array_via_NP", &roundtrip_numpy_array_via_NP );


    m.def("create_2d",
        [](size_t rows, size_t cols)
        {
            // Allocate a memory region and initialize it
            float *data = new float[rows * cols];
            for (size_t i = 0; i < rows * cols; ++i) data[i] = (float) i;

            // Delete 'data' when the 'owner' capsule expires
            nb::capsule owner(data,
               [](void *p) noexcept
               {
                    delete[] (float *) p;
               }
            );

            return nb::ndarray<nb::numpy, float, nb::ndim<2>>(
                   data,
                   { rows, cols },
                   owner
            );
        }
    );
}


