#!/usr/bin/env python
"""
npy.py
=======

Demonstrates sending and receiving NPY arrays over TCP 
In one session start the server::

    npy.py --server

In the other start the client::

    npy.py --client 


The client sends an array to the server where some processing 
is done on the array after which the array is sent back to the client.

* https://docs.python.org/3/howto/sockets.html

"""
import os, sys, logging, argparse
import io, struct, binascii as ba
import socket 
import numpy as np

log = logging.getLogger(__name__)
x_ = lambda _:ba.hexlify(_)

HEADER_FORMAT, HEADER_LENGTH = ">L", 4   # big-endian unsigned long which is 4 bytes 

def serialize(arr):
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

def deserialize(buf):
    """
    :param buf:
    :return arr:
    """  
    fd = io.BytesIO(buf) 
    arr = np.load(fd)
    return arr

def pack_header(num_bytes):
    """
    :param num_bytes: int
    :return bytes:
    """
    return struct.pack(HEADER_FORMAT, num_bytes) 

def unpack_header(data):
    num_bytes, = struct.unpack(HEADER_FORMAT, data)  # tuple -> , 
    return num_bytes

def serialize_with_header(arr):
    """
    :param arr: numpy array
    :return buf: memoryview of NPY serialized bytes 
    """
    fd = io.BytesIO()

    hdr = pack_header(0)
    fd.write(hdr)              # placeholder zero in header 
    np.save( fd, arr)          # write ndarray to stream 
    tot_bytes = fd.tell()      # better than fd.getbuffer().nbytes
    body_bytes = tot_bytes - len(hdr)

    fd.seek(0)  # rewind and fill in the header now that we know the size 
    fd.write(pack_header(body_bytes))

    buf = fd.getbuffer()
    assert type(buf) is memoryview
    log.info("serialized arr %r into memoryview len(buf) %d  tot_bytes %d body_bytes %d  " % (arr.shape, len(buf), tot_bytes, body_bytes ))
    return buf

def deserialize_with_header(buf):
    """
    :param buf:
    :return arr:

    Note that body_bytes not actually needed when are sure of completeness
    of the buffer, unlike when reading from a network socket where there is 
    no gaurantee of completeness. 
    """
    fd = io.BytesIO(buf)
    hdr = fd.read(HEADER_LENGTH)
    body_bytes = unpack_header(hdr)
    arr = np.load(fd)
    log.info("body_bytes:%d deserialized:%r" % (body_bytes,arr.shape))
    return arr

def test_serialize_deserialize(arr0, dump=False):
    buf0 = serialize(arr0)
    if dump:
        print(x_(buf0))
    pass
    arr1 = deserialize(buf0) 
    assert np.all(arr0 == arr1)
    buf1 = serialize_with_header(arr1)
    if dump:
        print(x_(buf1))
    pass
    arr2 = deserialize_with_header(buf1)
    assert np.all(arr0 == arr2)
    log.info("buf0:%d buf1:%d" % (len(buf0),len(buf1)))

def read_exactly(sock, n):
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

def make_array():
    return np.arange(16, dtype=np.uint8)

def npy_send(sock, arr):
    buf = serialize_with_header(arr)   
    sock.sendall(buf)

def npy_recv(sock):
    body_bytes = unpack_header(read_exactly(sock, HEADER_LENGTH))
    buf = read_exactly(sock, body_bytes)
    arr = deserialize(buf)    
    log.info("body_bytes:%d arr:%s " % (body_bytes,repr(arr.shape)))
    return arr

def server(args):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(args.addr)
    sock.listen(1)   # 1:connect requests
    while True:
        conn, addr = sock.accept()
        arr = npy_recv(conn)
        arr *= 10               ## server does some processing on the array 
        npy_send(conn, arr)
    pass
    conn.close()

def client(args, arr):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect(args.addr)
    npy_send(sock, arr)
    arr2 = npy_recv(sock)
    sock.close()
    print("arr2")
    print(arr2)

def parse_args(doc):
    parser = argparse.ArgumentParser(doc)
    parser.add_argument("-p","--path", default=None)
    parser.add_argument("-s","--server", action="store_true", default=False)
    parser.add_argument("-c","--client", action="store_true", default=False)
    parser.add_argument("-t","--test", action="store_true", default=False)
    args = parser.parse_args()
    port = os.environ.get("TCP_PORT","15006")
    host = os.environ.get("TCP_HOST", "127.0.0.1" )  # socket.gethostname()
    addr = (host, int(port))
    args.addr = addr 
    logging.basicConfig(level=logging.INFO)
    return args

if __name__ == '__main__':
    args = parse_args(__doc__)
    arr0 = make_array() if args.path is None else np.load(args.path) 

    if args.server:
        server(args)
    elif args.client:
        client(args, arr0)
    elif args.test:
        test_serialize_deserialize(arr0)
    else:
        pass
    pass 


