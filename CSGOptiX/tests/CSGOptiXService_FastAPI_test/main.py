"""
main.py
========

make_response
   returns fastapi.Response containing array and meta string

make_numpy_array_from_magic_bytes
   so far metadata not extracted

make_numpy_array_from_magic_bytes_with_meta
   metadata is assumed to be stuffed into the buffer after header and array data

make_numpy_array_from_raw_bytes

parse_request
    extracting numpy array and values of some headers

simulate
    invokes C++ CSGOptiXService via _CSGOptiXService which straddles python/C++ barrier

array_create
    test endpoint

"""


import io
import numpy as np
from typing import Annotated, Optional
from urllib.parse import unquote
from urllib.parse import quote

#from pydantic import BaseModel
from fastapi import FastAPI, Request, Response, Header, Depends, HTTPException

import opticks_CSGOptiX as cx

_svc = cx._CSGOptiXService()

app = FastAPI()


def make_response( arr:np.ndarray, meta:str="", magic:bool=False, index:int=-1, level:int = 0):
    """
    :param arr: array to respond with over HTTP
    :param meta: metadata to append to the transport buffer after the array data

    :param magic:True
         NumPy MAGIC+dtype+shape metadata travel within the response data,
    :param magic:False
         NumPy dtype+shape metadata travel via the response headers, array data travels in body

    :param index: metadata ? event index ?
    :param level: logging verbosity
    :return response: Response

    Note that the dtype and shape headers in magic:True case are duplicating
    the internal NPY header infomation. But these headers are small anyhow and
    can allow users to know the meaning sooner and without using NP.hh or NumPy.
    NP::WriteToArrayCallback already decodes the internal NPY header and sets
    array size as soon as all the bytes of the interal header have arrived. This means
    that buffer growing and reallocation is avoided and the array already arrives off the
    network with zero copying.

    See optimization idea in::

        is_real_zero_copy_possible_for_receiving_hits_from_the_server.rst


    """
    if level > 0:
        print("make_response arr.shape:%s meta:%s magic:%s index:%s level:%s  " % (str(arr.shape), meta.replace("\n",","),magic,index,level))
    pass
    headers = {}
    headers["x-opticks-index"] = str(index)
    headers["x-opticks-level"] = str(level)
    headers["x-opticks-dtype"] = arr.dtype.name
    headers["x-opticks-shape"] = str(arr.shape)
    headers["x-opticks-meta"] = quote(meta, safe="") # safe="" means encode everything, even / and spaces

    data:bytes = b''

    if magic:
        # directly save arr bytes to buffer and append any meta string
        buffer = io.BytesIO()
        np.save(buffer, arr)
        if len(meta) > 0:
            buffer.write(meta.encode("utf-8"))
        pass
        data = buffer.getvalue()
    else:
        data = arr.tobytes('C')
    pass
    return Response(data, headers=headers, media_type="application/octet-stream" )




def make_numpy_array_from_magic_bytes(data:bytes, level:int=0):
    """
    :param data: bytes which are assumed to include the numpy magic header
    :return arr: numpy.ndarray constructed from the bytes

    NB any metadata concatenated following array data is currently ignored

    TODO: follow tests/NP_nanobind_test/meta_check.py to get meta from the bytes too

    Used from parse_request when magic enabled
    """
    buffer = io.BytesIO(data)
    buffer.seek(0)
    arr = np.load(buffer)
    return arr

def make_numpy_array_from_magic_bytes_with_meta(data:bytes, level:int=0):
    """
    :param data: bytes
    :param level: int
    :return arr,meta: tuple

    Test for this in ~/np/tests/NP_nanobind_test/meta_check.py

    1. np.load array from the data buffer
    2. calculate meta_nbytes which may be appended after array data
    3. if there are meta_nbytes decode into meta string
    4. return (arr, meta) tuple
    """

    if level > 0:print(f"[make_numpy_array_from_magic_bytes_with_meta")

    # 1. np.load array from the data buffer
    buffer = io.BytesIO(data)
    buffer.seek(0)
    arr = np.load(buffer)

    # 2. calculate meta_nbytes which may be appended after array data
    buf_nbytes = len(data) # buffer.getbuffer().nbytes
    hdr_nbytes = data.find(b'\n') + 1  # 1 + index of first newline, typically 128 but can be more for arrays with many dimensions
    arr_nbytes = arr.nbytes
    meta_nbytes = buf_nbytes - hdr_nbytes - arr_nbytes

    # 3. if there are meta_nbytes decode into meta string
    buffer.seek( hdr_nbytes + arr_nbytes )
    _meta = buffer.read(meta_nbytes) if meta_nbytes>0 else b""
    meta = _meta.decode("utf-8") if _meta else ""

    if level > 0:print(f"-make_numpy_array_from_magic_bytes_with_meta buf_nbytes:{buf_nbytes} hdr_nbytes:{hdr_nbytes} arr_nbytes:{arr_nbytes} meta_nbytes:{meta_nbytes} meta:{meta} ")
    if level > 0:print(f"]make_numpy_array_from_magic_bytes_with_meta")

    # 4. return (arr, meta) tuple
    return arr, meta


def make_numpy_array_from_raw_bytes(data:bytes, dtype_:str, shape_:str, level:int=0 ):
    """
    :param data: bytes which are assumed to just carry array data, no header or metadata
    :param dtype_: str
    :param shape_: str

    Used from parse_request when magic not enabled

    raw bytes require dtype and shape metadata strings
    """
    dtype = getattr(np, dtype_, None)
    shape = tuple(map(int,filter(None,map(str.strip,shape_.replace("(","").replace(")","").split(",")))))
    a0 = np.frombuffer(data, dtype=dtype ).reshape(*shape)
    return a0


async def parse_request(request: Request):
    """
    :param request: FastAPI Request
    :return None:

    Request headers and data are parsed and request.state is populated::

        request.state.array
        request.state.meta   ## from metadata appended after array data
        request.state.hmeta  ## from metadata transported via HTTP header
        request.state.index
        request.state.level
        request.state.magic

    """
    token_ = request.headers.get('x-opticks-token')
    if token_ != "secret":
        raise HTTPException(status_code=401, detail="x-opticks-token invalid")
    pass

    level = int(request.headers.get('x-opticks-level','0'))
    index = int(request.headers.get('x-opticks-index','0'))
    count = int(request.headers.get('x-opticks-count','0'))

    hmeta = unquote(request.headers.get("x-opticks-meta", ""))
    hdtype = request.headers.get("x-opticks-dtype","")
    hshape = request.headers.get("x-opticks-shape","")

    type_ = request.headers.get('content-type')


    if level > 0:
        print(f"[parse_request url {request.url} type_ {type_}")
        print(f"request.headers\n{request.headers}")
    pass

    data:bytes = b''
    if type_.startswith("multipart/form-data"):
        field = "upload"  # needs to match field name from client
        form = await request.form()
        filename = form[field].filename
        data = await form[field].read()
    else:
        filename = None
        data = await request.body()
    pass

    numpy_magic = b'\x93NUMPY'
    has_numpy_magic = data.startswith(numpy_magic)
    if has_numpy_magic:
        arr0, meta = make_numpy_array_from_magic_bytes_with_meta(data, level)
    else:
        arr0 = make_numpy_array_from_raw_bytes(data, hdtype, hshape, level )
        meta = hmeta
    pass
    arr = arr0 if level == 10 else arr0.copy()
    ## without the copy get runtime type error in the nanobind call across the C++ python barrier
    ## this is the not so big gensteps arriving from client

    if level > 0:
        # https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flags.html
        print("has_numpy_magic:%s" % has_numpy_magic )
        print("arr0.data.c_contiguous:%s" % arr0.data.c_contiguous )
        print("arr.data.c_contiguous:%s" % arr.data.c_contiguous )
        #print("arr0.flags:\n", arr0.flags)
        #print("arr.flags:\n", arr.flags)
        print("content-type:%s" % type_ )
        print("filename:%s" % filename )
        print("token_[%s]" % token_ )
        print("level[%d]" % level )
        print("index[%d]" % index )
        print("count[%d]" % count )
        print("hmeta[%s]" % hmeta.replace("\n",",") )
        print("meta[%s]" % meta.replace("\n",",") )
        print("hdtype[%s]" % str(hdtype) )
        print("hshape[%s]" % str(hshape) )
        #print("type(arr0)\n", type(arr0))
        #print("type(arr)\n", type(arr))
        if level > 1: print("arr[%s]" % arr )
        print("]parse_request")
    pass
    request.state.array = arr
    request.state.meta = meta
    request.state.hmeta = hmeta
    request.state.hdtype = hdtype
    request.state.hshape = hshape
    request.state.index = index
    request.state.count = count
    request.state.level = level
    request.state.magic = has_numpy_magic




@app.post('/simulate', response_class=Response, dependencies=[Depends(parse_request)])
async def simulate(request: Request):
    """
    :param request: Request
    :return response: Response

    1. parse_request dependency sets request.state values
    2. call _svc with *gs* to give *ht*
    3. return *ht* as FastAPI Response

    Test this with ~/np/tests/np_curl_test/np_curl_test.sh
    """
    gs = request.state.array
    gs_meta = request.state.meta
    index = request.state.index
    count = request.state.count
    level = request.state.level
    magic = request.state.magic

    if level > -1: print("main.py:simulate index %d count %d gs.shape %s gs_meta[%s] " % ( index, count, str(gs.shape), gs_meta.replace("\n",",") ))

    ht, ht_meta = _svc.simulate_with_meta(gs, gs_meta, index)

    response = make_response(ht, ht_meta, magic, index, level)

    return response




@app.get('/array_create', response_class=Response)
def array_create():
    """
    ::

       curl -s -D /dev/stdout http://127.0.0.1:8000/array_create  --output arr

    """
    arr = np.arange(32, dtype=np.float32).reshape(2,4,4)
    meta:str = ""
    magic:bool = True
    index:int = 1
    level:int = 1

    response = make_response( arr, meta, magic, index, level )
    return response




