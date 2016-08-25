# GraSP
Distributed Streaming Graph Partitioning (C/MPI)

=============================================================================
## Summary
GraSP should build (using make) given a stable installation of MPI. 
METIS must be installed to use the ParMETIS benchmarking code.

The MPI implementation of GraSP will create a number of partitions equal to the number of MPI processes passed to it.
By default GraSP will generate a power-law graph using the Graph500 generator.

A 64-partition of a Kronecker graph with 2^18 vertices (and 16*2^18 edges) is called via the following:

mpirun -np 64 ./grasp_test 18 16

### Third-Party Code
This code includes scaffolding and a graph generator from the Graph 500 reference code by Lumsdaine and Willcock. 
These are used and included under the Boost Software License:

Boost Software License - Version 1.0 - August 17th, 2003

Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:

The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
