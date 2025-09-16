"""
check.py
=========

Huh this works, but doing very simular from FastAPI does not work

"""
import functools, operator, numpy as np
import opticks_CSGOptiX as cx



def read_arr_from_rawfile(path="$HOME/Downloads/arr", dtype_="uint8", shape_="512,512,3" ):
    """
    this suceeded to recover the numpy array from octet-stream of bytes download from /arr endpoint,
    but it is cheating regards the metadata
    """
    x = None

    dtype = getattr(np, dtype_, None)
    shape = tuple(map(int,shape_.split(",")))

    with open(os.path.expandvars(path), "rb") as fp:
        xb = fp.read()
        x = np.frombuffer(xb, dtype=dtype ).reshape(*shape)
    pass
    return x


def array_create_():
    a = np.zeros((512, 512, 3), dtype=np.uint8)
    a[0:256, 0:256] = [255, 0, 0] # red patch in upper left
    return a



def arange_check():
    print("cx\n", cx)
    _svc = cx._CSGOptiXService()

    print("repr(_svc)\n", repr(_svc))
    print("_svc\n", _svc)


    shape = (10,6,4)
    sz = functools.reduce(operator.mul,shape)
    gs = np.arange(sz, dtype=np.float32).reshape(*shape)
    print("gs\n", gs)

    hit = _svc.simulate(gs)   ## NB this is wrapper which handles numpy arrays

    print("hit\n", hit)


def check_with_non_owned_array():
    """
    --------------------------------------------------------------------------
    TypeError                                 Traceback (most recent call last)
    File ~/opticks/CSGOptiX/tests/CSGOptiXService_FastAPI_test/check.py:64
         61 buffer = data.tobytes()
         62 a0 = np.frombuffer(buffer, dtype=np.float32).reshape(2, 2)
    ---> 64 hit = cx._CSGOptiXService_Simulate(a0)

    TypeError: _CSGOptiXService_Simulate(): incompatible function arguments. The following argument types are supported:
        1. _CSGOptiXService_Simulate(input: numpy.ndarray) -> numpy.ndarray

    Invoked with types: ndarray
    """
    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    buffer = data.tobytes()
    a0 = np.frombuffer(buffer, dtype=np.float32).reshape(2, 2)
    a = a0

    hit = cx._CSGOptiXService_Simulate(a)  # Test with non-owning array




if __name__ == '__main__':


    data = np.array([[1, 2], [3, 4]], dtype=np.float32)
    buffer = data.tobytes()
    a0 = np.frombuffer(buffer, dtype=np.float32).reshape(2, 2)
    a = a0.copy()  # copying makes it owned

    hit = cx._CSGOptiXService_Simulate(a)







