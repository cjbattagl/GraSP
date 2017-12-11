CFLAGS = -g -O3 -Wchar-subscripts -Drestrict=__restrict__ -Wall -ffast-math -DGRAPH_GENERATOR_MPI # -g -pg
LDFLAGS = -g # -g -pg
MPICC = mpicc

all: grasp_test

GENERATOR_SOURCES = ../generator/graph_generator.c ../generator/make_graph.c ../generator/splittable_mrg.c ../generator/utils.c
SOURCES = main.c oned_csr.c utils.c

HEADERS = common.h oned_csr.h mpi_workarounds.h

parmetis_test: grasp_parmetis.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC)  $(CFLAGS) $(LDFLAGS) -o parmetis_test  grasp_parmetis.c $(SOURCES) $(GENERATOR_SOURCES) -lm

grasp_test: grasp.c $(SOURCES) $(HEADERS) $(GENERATOR_SOURCES)
	$(MPICC) $(CFLAGS) $(LDFLAGS) -o grasp_test grasp.c $(SOURCES) $(GENERATOR_SOURCES) -lm

clean:
	-rm -f *.o *.a
