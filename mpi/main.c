/* Copyright (C) 2010 The Trustees of Indiana University.                  */
/*                                                                         */
/* Use, modification and distribution is subject to the Boost Software     */
/* License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at */
/* http://www.boost.org/LICENSE_1_0.txt)                                   */
/*                                                                         */
/*  Authors: Jeremiah Willcock                                             */
/*           Andrew Lumsdaine                                              */

/* These need to be before any possible inclusions of stdint.h or inttypes.h.
 * */
#ifndef __STDC_LIMIT_MACROS
#define __STDC_LIMIT_MACROS
#endif
#ifndef __STDC_FORMAT_MACROS
#define __STDC_FORMAT_MACROS
#endif

#include "../generator/make_graph.h"
#include "../generator/utils.h"
#include "common.h"
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include <inttypes.h>

enum {s_minimum, s_firstquartile, s_median, s_thirdquartile, s_maximum, s_mean, s_std, s_LAST};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  setup_globals();

  /* Parse arguments. */
  int SCALE = 16;
  int edgefactor = 16; /* nedges / nvertices, i.e., 2*avg. degree */
  if (argc >= 2) SCALE = atoi(argv[1]);
  if (argc >= 3) edgefactor = atoi(argv[2]);
  if (argc <= 1 || argc >= 4 || SCALE == 0 || edgefactor == 0) {
    if (rank == 0) {
      fprintf(stderr, "Usage: %s SCALE edgefactor\n  SCALE = log_2(# vertices) [integer, required]\n  edgefactor = (# edges) / (# vertices) = .5 * (average vertex degree) [integer, defaults to 16]\n(Random number seed and Kronecker initiator are in main.c)\n", argv[0]);
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  uint64_t seed1 = 2, seed2 = 3;

  const char* filename = getenv("TMPFILE");
  /* If filename is NULL, store data in memory */

  tuple_graph tg;
  tg.nglobaledges = (int64_t)(edgefactor) << SCALE;
  int64_t nglobalverts = (int64_t)(1) << SCALE;

  tg.data_in_file = (filename != NULL);

  if (tg.data_in_file) {
    MPI_File_set_errhandler(MPI_FILE_NULL, MPI_ERRORS_ARE_FATAL);
    MPI_File_open(MPI_COMM_WORLD, (char*)filename, MPI_MODE_RDWR | MPI_MODE_CREATE | MPI_MODE_EXCL | MPI_MODE_DELETE_ON_CLOSE | MPI_MODE_UNIQUE_OPEN, MPI_INFO_NULL, &tg.edgefile);
    MPI_File_set_size(tg.edgefile, tg.nglobaledges * sizeof(packed_edge));
    MPI_File_set_view(tg.edgefile, 0, packed_edge_mpi_type, packed_edge_mpi_type, "native", MPI_INFO_NULL);
    MPI_File_set_atomicity(tg.edgefile, 0);
  }

  /* Make the raw graph edges. */
  /* Get roots for BFS runs, plus maximum vertex with non-zero degree (used by
   * validator). */
  int num_bfs_roots = NUM_ROOTS;
  int64_t* bfs_roots = (int64_t*)xmalloc(num_bfs_roots * sizeof(int64_t));
  int64_t max_used_vertex = 0;

  double make_graph_start = MPI_Wtime();
  {
    /* Spread the two 64-bit numbers into five nonzero values in the correct
     * range. */
    uint_fast32_t seed[5];
    make_mrg_seed(seed1, seed2, seed);

  ////////// 1. CREATE GRAPH  /////////////////

    /* As the graph is being generated, also keep a bitmap of vertices with
     * incident edges.  We keep a grid of processes, each row of which has a
     * separate copy of the bitmap (distributed among the processes in the
     * row), and then do an allreduce at the end.  This scheme is used to avoid
     * non-local communication and reading the file separately just to find BFS
     * roots. */
    MPI_Offset nchunks_in_file = (tg.nglobaledges + FILE_CHUNKSIZE - 1) / FILE_CHUNKSIZE;
    int64_t bitmap_size_in_bytes = int64_min(BITMAPSIZE, (nglobalverts + CHAR_BIT - 1) / CHAR_BIT);
    if (bitmap_size_in_bytes * size * CHAR_BIT < nglobalverts) {
      bitmap_size_in_bytes = (nglobalverts + size * CHAR_BIT - 1) / (size * CHAR_BIT);
    }
    int ranks_per_row = ((nglobalverts + CHAR_BIT - 1) / CHAR_BIT + bitmap_size_in_bytes - 1) / bitmap_size_in_bytes;
    int nrows = size / ranks_per_row;
    int my_row = -1, my_col = -1;
    unsigned char* restrict has_edge = NULL;
    MPI_Comm cart_comm;
    {
      int dims[2] = {size / ranks_per_row, ranks_per_row};
      int periods[2] = {0, 0};
      MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);
    }
    int in_generating_rectangle = 0;
    if (cart_comm != MPI_COMM_NULL) {
      in_generating_rectangle = 1;
      {
        int dims[2], periods[2], coords[2];
        MPI_Cart_get(cart_comm, 2, dims, periods, coords);
        my_row = coords[0];
        my_col = coords[1];
      }
      MPI_Comm this_col;
      MPI_Comm_split(cart_comm, my_col, my_row, &this_col);
      MPI_Comm_free(&cart_comm);
      has_edge = (unsigned char*)xMPI_Alloc_mem(bitmap_size_in_bytes);
      memset(has_edge, 0, bitmap_size_in_bytes);
      /* Every rank in a given row creates the same vertices (for updating the
       * bitmap); only one writes them to the file (or final memory buffer). */
      packed_edge* buf = (packed_edge*)xmalloc(FILE_CHUNKSIZE * sizeof(packed_edge));
      MPI_Offset block_limit = (nchunks_in_file + nrows - 1) / nrows;
      /* fprintf(stderr, "%d: nchunks_in_file = %" PRId64 ", block_limit = %" PRId64 " in grid of %d rows, %d cols\n", rank, (int64_t)nchunks_in_file, (int64_t)block_limit, nrows, ranks_per_row); */
      if (tg.data_in_file) {
        tg.edgememory_size = 0;
        tg.edgememory = NULL;
      } else {
        int my_pos = my_row + my_col * nrows;
        int last_pos = (tg.nglobaledges % ((int64_t)FILE_CHUNKSIZE * nrows * ranks_per_row) != 0) ?
                       (tg.nglobaledges / FILE_CHUNKSIZE) % (nrows * ranks_per_row) :
                       -1;
        int64_t edges_left = tg.nglobaledges % FILE_CHUNKSIZE;
        int64_t nedges = FILE_CHUNKSIZE * (tg.nglobaledges / ((int64_t)FILE_CHUNKSIZE * nrows * ranks_per_row)) +
                         FILE_CHUNKSIZE * (my_pos < (tg.nglobaledges / FILE_CHUNKSIZE) % (nrows * ranks_per_row)) +
                         (my_pos == last_pos ? edges_left : 0);
        /* fprintf(stderr, "%d: nedges = %" PRId64 " of %" PRId64 "\n", rank, (int64_t)nedges, (int64_t)tg.nglobaledges); */
        tg.edgememory_size = nedges;
        tg.edgememory = (packed_edge*)xmalloc(nedges * sizeof(packed_edge));
      }
      MPI_Offset block_idx;
      for (block_idx = 0; block_idx < block_limit; ++block_idx) {
        /* fprintf(stderr, "%d: On block %d of %d\n", rank, (int)block_idx, (int)block_limit); */
        MPI_Offset start_edge_index = int64_min(FILE_CHUNKSIZE * (block_idx * nrows + my_row), tg.nglobaledges);
        MPI_Offset edge_count = int64_min(tg.nglobaledges - start_edge_index, FILE_CHUNKSIZE);
        packed_edge* actual_buf = (!tg.data_in_file && block_idx % ranks_per_row == my_col) ?
                                  tg.edgememory + FILE_CHUNKSIZE * (block_idx / ranks_per_row) :
                                  buf;
        /* fprintf(stderr, "%d: My range is [%" PRId64 ", %" PRId64 ") %swriting into index %" PRId64 "\n", rank, (int64_t)start_edge_index, (int64_t)(start_edge_index + edge_count), (my_col == (block_idx % ranks_per_row)) ? "" : "not ", (int64_t)(FILE_CHUNKSIZE * (block_idx / ranks_per_row))); */
        if (!tg.data_in_file && block_idx % ranks_per_row == my_col) {
          assert (FILE_CHUNKSIZE * (block_idx / ranks_per_row) + edge_count <= tg.edgememory_size);
        }
        generate_kronecker_range(seed, SCALE, start_edge_index, start_edge_index + edge_count, actual_buf);
        if (tg.data_in_file && my_col == (block_idx % ranks_per_row)) { /* Try to spread writes among ranks */
          MPI_File_write_at(tg.edgefile, start_edge_index, actual_buf, edge_count, packed_edge_mpi_type, MPI_STATUS_IGNORE);
        }
        ptrdiff_t i;
#ifdef _OPENMP
#pragma omp parallel for
#endif
        for (i = 0; i < edge_count; ++i) {
          int64_t src = get_v0_from_edge(&actual_buf[i]);
          int64_t tgt = get_v1_from_edge(&actual_buf[i]);
          if (src == tgt) continue;
          if (src / bitmap_size_in_bytes / CHAR_BIT == my_col) {
#ifdef _OPENMP
#pragma omp atomic
#endif
            has_edge[(src / CHAR_BIT) % bitmap_size_in_bytes] |= (1 << (src % CHAR_BIT));
          }
          if (tgt / bitmap_size_in_bytes / CHAR_BIT == my_col) {
#ifdef _OPENMP
#pragma omp atomic
#endif
            has_edge[(tgt / CHAR_BIT) % bitmap_size_in_bytes] |= (1 << (tgt % CHAR_BIT));
          }
        }
      }
      free(buf);
#if 0
      /* The allreduce for each root acts like we did this: */
      MPI_Allreduce(MPI_IN_PLACE, has_edge, bitmap_size_in_bytes, MPI_UNSIGNED_CHAR, MPI_BOR, this_col);
#endif
      MPI_Comm_free(&this_col);
    } else {
      tg.edgememory = NULL;
      tg.edgememory_size = 0;
    }
    MPI_Allreduce(&tg.edgememory_size, &tg.max_edgememory_size, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
    /* Find roots and max used vertex */
    {
      uint64_t counter = 0;
      int bfs_root_idx;
      for (bfs_root_idx = 0; bfs_root_idx < num_bfs_roots; ++bfs_root_idx) {
        int64_t root;
        while (1) {
          double d[2];
          make_random_numbers(2, seed1, seed2, counter, d);
          root = (int64_t)((d[0] + d[1]) * nglobalverts) % nglobalverts;
          counter += 2;
          if (counter > 2 * nglobalverts) break;
          int is_duplicate = 0;
          int i;
          for (i = 0; i < bfs_root_idx; ++i) {
            if (root == bfs_roots[i]) {
              is_duplicate = 1;
              break;
            }
          }
          if (is_duplicate) continue; /* Everyone takes the same path here */
          int root_ok = 0;
          if (in_generating_rectangle && (root / CHAR_BIT / bitmap_size_in_bytes) == my_col) {
            root_ok = (has_edge[(root / CHAR_BIT) % bitmap_size_in_bytes] & (1 << (root % CHAR_BIT))) != 0;
          }
          MPI_Allreduce(MPI_IN_PLACE, &root_ok, 1, MPI_INT, MPI_LOR, MPI_COMM_WORLD);
          if (root_ok) break;
        }
        bfs_roots[bfs_root_idx] = root;
      }
      num_bfs_roots = bfs_root_idx;

      /* Find maximum non-zero-degree vertex. */
      {
        int64_t i;
        max_used_vertex = 0;
        if (in_generating_rectangle) {
          for (i = bitmap_size_in_bytes * CHAR_BIT; i > 0; --i) {
            if (i > nglobalverts) continue;
            if (has_edge[(i - 1) / CHAR_BIT] & (1 << ((i - 1) % CHAR_BIT))) {
              max_used_vertex = (i - 1) + my_col * CHAR_BIT * bitmap_size_in_bytes;
              break;
            }
          }
        }
        MPI_Allreduce(MPI_IN_PLACE, &max_used_vertex, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
        //Because we permute:
        max_used_vertex = nglobalverts;
      }
    }
    if (in_generating_rectangle) {
      MPI_Free_mem(has_edge);
    }
    if (tg.data_in_file) {
      MPI_File_sync(tg.edgefile);
    }
  }
  double make_graph_stop = MPI_Wtime();
  double make_graph_time = make_graph_stop - make_graph_start;
  if (rank == 0) { /* Not an official part of the results */
    fprintf(stderr, "graph_generation:               %f s\n", make_graph_time);
  }

  //fprintf(stderr, "%d: nedges = %" PRId64 " of %" PRId64 "   ", rank, (int64_t)tg.edgememory_size, (int64_t)tg.nglobaledges);
  //fprintf(stderr, "max = %" PRId64 "\n", (int64_t)tg.max_edgememory_size);

  ////////// 2. CONVERT TO CSR  /////////////////
  /* Make user's graph data structure. */
  double data_struct_start = MPI_Wtime();
  make_graph_data_structure(&tg);
  double data_struct_stop = MPI_Wtime();
  double data_struct_time = data_struct_stop - data_struct_start;
  if (rank == 0) { /* Not an official part of the results */
    fprintf(stderr, "construction_time:              %f s\n", data_struct_time);
  }

  FILE *GraphFile;
  if(MAT_OUT){
    char filename1[256];
    sprintf(filename1, "out_tup%02d.mat", rank);
    GraphFile = fopen(filename1, "w");
    assert(GraphFile != NULL);
    print_graph_tuple(GraphFile, &tg, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    fclose(GraphFile);
  }

  ////////// 3. RUN FENNEL ON CSR  /////////////////
  double graph_part_start = MPI_Wtime();
  partition_graph_data_structure(); 
  double graph_part_stop = MPI_Wtime();
  if (rank == 0) { fprintf(stderr, "graph part time:               %f s\n", graph_part_stop - graph_part_start); }

  if (rank == 0) { 
    fprintf(stdout, "SCALE:%d E_F:%d, np:%d \n", SCALE,edgefactor,size);
    fprintf(stdout, "edgefactor:                     %d\n", edgefactor);
    fprintf(stdout, "NBFS:                           %d\n", num_bfs_roots); 
    fprintf(stdout, "num_mpi_processes:              %d\n", size);
  }

  cleanup_globals();
  MPI_Finalize();
  return 0;
}

