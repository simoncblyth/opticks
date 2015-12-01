// based on ufunc-/logit/single_type_logit.c 

#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

#include "ciexyz.h"
#include "blackbody.h"


#define FLOAT_FUNC(XNAME,INAME) \
static void XNAME(char **args, npy_intp *dimensions, npy_intp* steps, void* data) \
{ \
    npy_intp i; \
    npy_intp n = dimensions[0]; \
    char *in = args[0], *out = args[1]; \
    npy_intp in_step = steps[0], out_step = steps[1]; \
                     \
    float tmp;       \
                        \
    for (i = 0; i < n; i++) {   \
                               \
        tmp = *(float *)in;     \
        *((float *)out) = INAME(tmp) ; \
    \
        in += in_step; \
        out += out_step; \
    }  \
}  \



#define DOUBLE_COERCED_FUNC(XNAME,INAME) \
static void XNAME(char **args, npy_intp *dimensions, npy_intp* steps, void* data) \
{ \
    npy_intp i; \
    npy_intp n = dimensions[0]; \
    char *in = args[0], *out = args[1]; \
    npy_intp in_step = steps[0], out_step = steps[1]; \
                     \
    double tmp;       \
    float ftmp;       \
    float fout;       \
                        \
    for (i = 0; i < n; i++) {   \
                               \
        tmp = *(double *)in;     \
                                 \
        ftmp = (float)tmp ;      \
        fout = INAME(ftmp) ;     \
                                   \
        *((double *)out) = (double)(fout) ; \
    \
        in += in_step; \
        out += out_step; \
    }  \
}  \






FLOAT_FUNC(float_cieX, xFit_1931) ;
FLOAT_FUNC(float_cieY, yFit_1931) ;
FLOAT_FUNC(float_cieZ, zFit_1931) ;
FLOAT_FUNC(float_bb5k,   bb5k) ;
FLOAT_FUNC(float_bb6k,   bb6k) ;

// using double is unnecessary, but provide for calling convenience

DOUBLE_COERCED_FUNC(double_cieX, xFit_1931) ;
DOUBLE_COERCED_FUNC(double_cieY, yFit_1931) ;
DOUBLE_COERCED_FUNC(double_cieZ, zFit_1931) ;
DOUBLE_COERCED_FUNC(double_bb5k,   bb5k) ;
DOUBLE_COERCED_FUNC(double_bb6k,   bb6k) ;


PyUFuncGenericFunction x_funcs[2] = { &float_cieX, &double_cieX };
static char x_types[4] = { NPY_FLOAT, NPY_FLOAT, NPY_DOUBLE, NPY_DOUBLE, };
static void * x_data[2] = { NULL, NULL } ;

PyUFuncGenericFunction y_funcs[2] = { &float_cieY, &double_cieY };
static char y_types[4] = { NPY_FLOAT, NPY_FLOAT, NPY_DOUBLE, NPY_DOUBLE, };
static void * y_data[2] = { NULL, NULL } ;

PyUFuncGenericFunction z_funcs[2] = { &float_cieZ, &double_cieZ };
static char z_types[4] = { NPY_FLOAT, NPY_FLOAT, NPY_DOUBLE, NPY_DOUBLE, };
static void * z_data[2] = { NULL, NULL } ;




PyUFuncGenericFunction bb5k_funcs[2] = { &float_bb5k, &double_bb5k };
static char bb5k_types[4] = { NPY_FLOAT, NPY_FLOAT, NPY_DOUBLE, NPY_DOUBLE, };
static void * bb5k_data[2] = { NULL, NULL } ;

PyUFuncGenericFunction bb6k_funcs[2] = { &float_bb6k, &double_bb6k };
static char bb6k_types[4] = { NPY_FLOAT, NPY_FLOAT, NPY_DOUBLE, NPY_DOUBLE, };
static void * bb6k_data[2] = { NULL, NULL } ;




static PyMethodDef CiexyzMethods[] = {
        {NULL, NULL, 0, NULL}
};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ciexyz",
    NULL,
    -1,
    CiexyzMethods,
    NULL,
    NULL,
    NULL,
    NULL
};
#endif


#if PY_VERSION_HEX >= 0x03000000
PyMODINIT_FUNC PyInit_ciexyz(void)
{
    PyObject *m ;
    m = PyModule_Create(&moduledef);
    if(!m) return NULL;
#else
PyMODINIT_FUNC initciexyz(void)
{
    PyObject *m, *d, *f ;
    m = Py_InitModule("ciexyz", CiexyzMethods);
    if(!m) return ;
#endif

    import_array();
    import_umath();

    d = PyModule_GetDict(m);

    //                                                  ntyp/ninp/nout/identity/name/docstring/unused
    f = PyUFunc_FromFuncAndData(x_funcs, x_data, x_types, 2, 1, 1, PyUFunc_None, "X", "cieX_docstring", 0);
    PyDict_SetItemString(d, "X", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndData(y_funcs, y_data, y_types, 2, 1, 1, PyUFunc_None, "Y", "cieY_docstring", 0);
    PyDict_SetItemString(d, "Y", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndData(z_funcs, z_data, z_types, 2, 1, 1, PyUFunc_None, "Z", "cieZ_docstring", 0);
    PyDict_SetItemString(d, "Z", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndData(bb5k_funcs, bb5k_data, bb5k_types, 2, 1, 1, PyUFunc_None, "BB5K", "BB5K_docstring", 0);
    PyDict_SetItemString(d, "BB5K", f);
    Py_DECREF(f);

    f = PyUFunc_FromFuncAndData(bb6k_funcs, bb6k_data, bb6k_types, 2, 1, 1, PyUFunc_None, "BB6K", "BB6K_docstring", 0);
    PyDict_SetItemString(d, "BB6K", f);
    Py_DECREF(f);


#if PY_VERSION_HEX >= 0x03000000
    return m;
#endif
}


