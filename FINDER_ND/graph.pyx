'''
# file: graph.pyx, implementation file for the graph wrapper.
# Cython can automatically load a same-name .pxd file from the same path.
# The cimport graph command can be omitted.
# This file provides the Python-facing interface.
'''
from cython.operator cimport dereference as deref
cimport cpython.ref as cpy_ref
from libcpp.memory cimport shared_ptr
from libc.stdlib cimport malloc
from libc.stdlib cimport free
import numpy as np
# import gc

cdef class py_Graph:
    cdef shared_ptr[Graph] inner_graph  # shared_ptr is used instead of unique_ptr.
    # __cinit__ is called before __init__.
    def __cinit__(self,*arg):
        '''doing something before python calls the __init__.
        cdef C/C++ objects must be initialized inside __cinit__; otherwise
        memory is not allocated for them. Variable Python arguments are used to
        mimic overloaded constructors.'''
        #print("doing something before python calls the __init__")
        # if len(arg)==0:
        #     print("num of parameter is 0")
        self.inner_graph = shared_ptr[Graph](new Graph())
        cdef int _num_nodes
        cdef int _num_edges
        cdef int[:] edges_from
        cdef int[:] edges_to
        if len(arg)==0:
            # These two assignments prevent uninitialized memory states.
            deref(self.inner_graph).num_edges=0
            deref(self.inner_graph).num_nodes=0
        elif len(arg)==4:
            _num_nodes=arg[0]
            _num_edges=arg[1]
            edges_from = np.array([int(x) for x in arg[2]], dtype=np.int32)
            edges_to = np.array([int(x) for x in arg[3]], dtype=np.int32)
            self.reshape_Graph(_num_nodes,  _num_edges,  edges_from,  edges_to)
        else:
            print('Error: py_Graph initialization failed because the number of arguments must be 0 or 4.')
    # def __dealloc__(self):
    #     if self.inner_graph != NULL:
    #         self.inner_graph.reset()
    #         gc.collect()
    @property
    def num_nodes(self):
        return deref(self.inner_graph).num_nodes

    # @num_nodes.setter
    # def num_nodes(self):
    #     def __set__(self,num_nodes):
    #         self.setadj(adj_list)

    @property
    def num_edges(self):
        return deref(self.inner_graph).num_edges

    @property
    def adj_list(self):
        return deref(self.inner_graph).adj_list

    @property
    def edge_list(self):
        return deref(self.inner_graph).edge_list

    cdef reshape_Graph(self, int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to):
        cdef int *cint_edges_from = <int*>malloc(_num_edges*sizeof(int))
        cdef int *cint_edges_to = <int*>malloc(_num_edges*sizeof(int))
        cdef int i
        for i in range(_num_edges):
            cint_edges_from[i] = edges_from[i]
        for i in range(_num_edges):
            cint_edges_to[i] = edges_to[i]
        self.inner_graph = shared_ptr[Graph](new Graph(_num_nodes,_num_edges,&cint_edges_from[0],&cint_edges_to[0]))
        free(cint_edges_from)
        free(cint_edges_to)

    def reshape(self,int _num_nodes, int _num_edges, int[:] edges_from, int[:] edges_to):
        self.reshape_Graph(_num_nodes, _num_edges, edges_from, edges_to)


cdef class py_GSet:
    cdef shared_ptr[GSet] inner_gset
    def __cinit__(self):
        self.inner_gset = shared_ptr[GSet](new GSet())
    # def __dealloc__(self):
    #     if self.inner_gset != NULL:
    #         self.inner_gset.reset()
    #         gc.collect()
    def InsertGraph(self,int gid,py_Graph graph):
        deref(self.inner_gset).InsertGraph(gid,graph.inner_graph)
        #self.InsertGraph(gid,graph.inner_graph)

        # deref(self.inner_gset).InsertGraph(gid,graph.inner_graph)
         #self.Inner_InsertGraph(gid,graph.inner_graph)

    def Sample(self):
        temp_innerGraph=deref(deref(self.inner_gset).Sample())   # Get the Graph object.
        return self.G2P(temp_innerGraph)

    def Get(self,int gid):
        temp_innerGraph=deref(deref(self.inner_gset).Get(gid))   # Get the Graph object.
        return self.G2P(temp_innerGraph)

    def Clear(self):
        deref(self.inner_gset).Clear()

    cdef G2P(self,Graph graph):
        num_nodes = graph.num_nodes     # Number of nodes in the Graph object.
        num_edges = graph.num_edges     # Number of edges in the Graph object.
        edge_list = graph.edge_list
        cint_edges_from = np.zeros([num_edges],dtype=np.int)
        cint_edges_to = np.zeros([num_edges],dtype=np.int)
        for i in range(num_edges):
            cint_edges_from[i]=edge_list[i].first
            cint_edges_to[i] =edge_list[i].second
        return py_Graph(num_nodes,num_edges,cint_edges_from,cint_edges_to)


