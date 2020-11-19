#!/usr/bin/env python
"""
npy.py
=======

Demonstrates sending and receiving NPY arrays and dict/json metadata over TCP socket.
To test serialization/deserialization::

    npy.py --test

In one session start the server::

    npy.py --server

In the other run the client::

    npy.py --client 

The client sends an array to the server where some processing 
is done on the array after which the array is sent back to the client
together with some metadata.

* https://docs.python.org/3/howto/sockets.html

"""
import os, sys, logging, argparse, json, datetime, operator, functools
import io, struct, binascii as ba
import socket 
import numpy as np
mul_ = lambda _:functools.reduce(operator.mul, _) 

log = logging.getLogger(__name__)
x_ = lambda _:ba.hexlify(_)

HEADER_FORMAT, HEADER_SIZE, HEADER_BYTES = ">LLLL", 4, 16   # four big-endian unsigned long which is 4*4 bytes 

def npy_serialize(arr):
    """
    :param arr: ndarray
    :return buf: bytes  
    """
    fd = io.BytesIO()
    np.save( fd, arr)     # write ndarray to stream 
    buf = fd.getbuffer()
    assert type(buf) is memoryview
    log.info("serialized arr %r into memoryview buf of length %d " % (arr.shape, len(buf)))
    return buf  

def npy_deserialize(buf):
    """
    :param buf:
    :return arr:
    """  
    fd = io.BytesIO(buf) 
    arr = np.load(fd)
    return arr

def meta_serialize(meta):
    buf = json.dumps(meta).encode() 
    return buf 

def meta_deserialize(buf):
    fd = io.BytesIO(buf) 
    meta = json.load(fd)
    return meta  

def pack_prefix(*sizes):
    """
    :param arr_bytes: uint
    :param meta_bytes: uint
    :return bytes:
    """
    assert len(sizes) == HEADER_SIZE 
    return struct.pack(HEADER_FORMAT, *sizes) 

def unpack_prefix(data):
    sizes = struct.unpack(HEADER_FORMAT, data)  
    assert len(sizes) == HEADER_SIZE 
    return sizes

def serialize_with_header(arr, meta):
    """
    :param arr: numpy array
    :param meta: metadata dict 
    :return buf: memoryview containing transport header followed by serialized NPY and metadata
    """
    fd = io.BytesIO()

    prefix = pack_prefix(0,0,0,0)  # placeholder zeroes in header
    fd.write(prefix)          
  
    np.save( fd, arr)       # write ndarray to stream 
    hdr_arr_bytes = fd.tell() - len(prefix)
     
    fd.write(meta_serialize(meta))
    meta_bytes = fd.tell() - hdr_arr_bytes - len(prefix)

    fd.seek(0)  # rewind and fill in the header with the sizes 

    arr_bytes = arr.nbytes
    hdr_bytes = hdr_arr_bytes - arr_bytes
    
    fd.write(pack_prefix(hdr_bytes,arr_bytes,meta_bytes,0))

    buf = fd.getbuffer()
    assert type(buf) is memoryview
    log.info("serialized arr %r into memoryview len(buf) %d  hdr_bytes %d arr_bytes %d meta_bytes %d  " % (arr.shape, len(buf), hdr_bytes, arr_bytes, meta_bytes ))
    return buf

def deserialize_with_header(buf):
    """
    :param fd: io.BytesIO stream 
    :return arr,meta:

    Note that arr_bytes, meta_bytes not actually needed when are sure of completeness
    of the buffer, unlike when reading from a network socket where there is 
    no gaurantee of completeness of the bytes received so far.
    """
    fd = io.BytesIO(buf)
    prefix = fd.read(HEADER_BYTES)
    hdr_bytes,arr_bytes,meta_bytes,zero = unpack_prefix(prefix)
    assert zero == 0  
    arr = np.load(fd)         # fortunately np.load ignores the metadata that follows the array 
    meta = json.load(fd)
    log.info("hdr_bytes:%d arr_bytes:%d meta_bytes:%d deserialized:%r" % (hdr_bytes,arr_bytes,meta_bytes,arr.shape))
    return arr, meta


def test_serialize_deserialize(arr0, meta0, dump=False):
    buf0 = npy_serialize(arr0)
    if dump:
        print(x_(buf0))
    pass
    arr1 = npy_deserialize(buf0) 
    assert np.all(arr0 == arr1)
    meta1 = meta0

    buf1 = serialize_with_header(arr1,meta1)
    if dump:
        print(x_(buf1))
    pass

    arr2,meta2 = deserialize_with_header(buf1)
    assert np.all(arr0 == arr2)

    print("meta2:%s" % repr(meta2))
    log.info("buf0:%d buf1:%d" % (len(buf0),len(buf1)))

def recv_exactly(sock, n):
    """ 
    https://eli.thegreenplace.net/2011/08/02/length-prefix-framing-for-protocol-buffers

    Raise RuntimeError if the connection closed before n bytes were read.
    """
    buf = b""
    while n > 0:
        data = sock.recv(n)
        if data == b'':
            raise RuntimeError('unexpected connection close')
        buf += data
        n -= len(data)
    return buf

def make_array(dtype="float32", shape="10,4"):
    log.info("make_array %s %s " % (dtype, shape))
    dtype = getattr(np, dtype, np.float32)
    shape = tuple(map(int,shape.split(",")))
    size = mul_(shape)
    arr = np.arange(size, dtype=dtype).reshape(shape)
    return arr

def npy_send(sock, arr, meta):
    buf = serialize_with_header(arr, meta)   
    sock.sendall(buf)

def npy_recv(sock):
    hdr_bytes, arr_bytes, meta_bytes, zero = unpack_net_header(recv_exactly(sock, HEADER_BYTES))
    assert zero == 0 
    arr = npy_deserialize(recv_exactly(sock, hdr_bytes+arr_bytes))    
    meta = meta_deserialize(recv_exactly(sock, meta_bytes))
    log.info("hdr_bytes:%d arr_bytes:%d meta_bytes:%d arr:%s " % (hdr_bytes,arr_bytes,meta_bytes,repr(arr.shape)))
    return arr, meta

def server(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(args.addr)
    sock.listen(1)   # 1:connect requests
    while True:
        conn, addr = sock.accept()
        arr, meta = npy_recv(conn)
        arr *= 10               ## server does some processing on the array 
        meta["stamp"] = datetime.datetime.now().strftime("%c")  
        npy_send(conn, arr, meta)
    pass
    conn.close()

def client(args, arr, meta):
    log.info("client")
    log.info("\n%s"%arr)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(args.addr)
    npy_send(sock, arr, meta)
    arr2,meta2 = npy_recv(sock)
    sock.close()
    print("arr2")
    print(arr2)
    print("meta2")
    print(meta2)


def demo():
    arr = np.arange(100, dtype=np.float32)
    meta = dict(hello="world",src="npy.py")
    path = os.path.expandvars("/tmp/$USER/opticks/demo.npy")
    fold = os.path.dirname(path)
    if not os.path.isdir(fold):
        os.makedirs(fold)
    pass
    log.info("saving to %s " % path )
    np.save(path,arr)
    json.dump(meta, open(path+".json","w"))


def parse_args(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-p","--path", default=None)
    parser.add_argument("-s","--server", action="store_true", default=False)
    parser.add_argument("-c","--client", action="store_true", default=False)
    parser.add_argument("-t","--test", action="store_true", default=False)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--shape", default="10,4")
    parser.add_argument("-d","--demo", action="store_true", default=False)
    args = parser.parse_args()

    args.metapath = None
    if not args.path is None: 
        args.metapath = args.path + ".json" 
        if not os.path.exists(args.metapath): 
            args.metapath = None
        pass
    pass

    port = os.environ.get("TCP_PORT","15006")
    host = os.environ.get("TCP_HOST", "127.0.0.1" ) 
    if host == "hostname":
        host = socket.gethostname()
    pass
    addr = (host, int(port))
    args.addr = addr 
    logging.basicConfig(level=logging.INFO)
    log.info("addr: %s" % repr(addr))
    return args


if __name__ == '__main__':
    args = parse_args(__doc__)
    arr0 = make_array(args.dtype, args.shape) if args.path is None else np.load(args.path) 
    meta0 = dict(red=1,green=2,blue=3) if args.metapath is None else json.load(open(args.metapath))

    if args.server:
        server(args)
    elif args.client:
        client(args, arr0, meta0)
    elif args.test:
        test_serialize_deserialize(arr0, meta0)
    elif args.demo:
        demo()
    else:
        pass
    pass 


