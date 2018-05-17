
#include "common.h"
#include "oned_csr.h"
#include <mpi.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <assert.h>

#include <time.h>
#include <math.h>

static oned_csr_graph g;

static PART_TYPE* parts;

static int64_t* g_perm;
static int64_t* g_oldq;
static int64_t* g_newq;

static unsigned long* g_visited;

// Used to coarsen communication
static const int coalescing_size = 256;

// Collective communication variables
static int64_t* g_outgoing;
static size_t* g_outgoing_counts /* 2x actual count */;
static MPI_Request* g_outgoing_reqs;
static int* g_outgoing_reqs_active;
static int64_t* g_recvbuf;

static int64_t num_hi_deg_verts;

void partition_graph_data_structure() { 
  size_t n = g.nglobalverts;
  size_t n_local = g.nlocalverts;
  size_t offset = g.nlocalverts * rank; //!//Does this work?

  int nparts = size;

  int64_t tot_nnz = 0;
  parts = (PART_TYPE*)malloc(n * sizeof(PART_TYPE));
  int64_t *partsize_update = (int64_t*)malloc(nparts * sizeof(int64_t));
  int64_t *old_partsize = (int64_t*)malloc(nparts * sizeof(int64_t));
  int64_t *partsize = (int64_t*)malloc(nparts * sizeof(int64_t));

  int64_t *partnnz_update = (int64_t*)malloc(nparts * sizeof(int64_t));
  int64_t *old_partnnz = (int64_t*)malloc(nparts * sizeof(int64_t));
  int64_t *partnnz = (int64_t*)malloc(nparts * sizeof(int64_t));

  int64_t *partscore = (int64_t*)malloc(nparts * sizeof(int64_t));
  int64_t *partcost = (int64_t*)malloc(nparts * sizeof(int64_t));
  int64_t *vorder = (int64_t*)malloc(n_local * sizeof(int64_t)); 

  PART_TYPE oldpart;
  PART_TYPE randidx;

  int64_t emptyverts = 0;
  num_hi_deg_verts = 0;
  size_t *row;
  size_t vert;
  size_t k,  mydegree, best_part;
  int64_t *colidx = g.column;
  size_t *rowptr = g.rowstarts;
  float curr_score, best_score;
  float gamma = F_GAMMA;
  size_t i, s, l;

  //g_perm = (int64_t*)malloc(n * sizeof(int64_t));
 
  if(MAT_OUT) { // Print graph
    char filename[256];
    sprintf(filename, "out_csr%02d.mat", rank);
    FILE *GraphFile;
    GraphFile = fopen(filename, "w");
    assert(GraphFile != NULL);
    print_graph_csr(GraphFile, rowptr, colidx, n_local);
    MPI_Barrier(MPI_COMM_WORLD);
    fclose(GraphFile);
  }

  //memset(parts, -1, n * sizeof(PART_TYPE));

  // Initialize partition stats
  for (l=0; l<nparts; ++l) {
    partsize[l] = 0;
    old_partsize[l] = 0;
    partsize_update[l] = 0;
    partnnz[l] = 0;
    old_partnnz[l] = 0;
    partnnz_update[l] = 0;
  }

  int64_t localedges = (int64_t)g.nlocaledges;
  MPI_Allreduce(&localedges, &tot_nnz, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  float alpha = sqrt(2) * (tot_nnz/pow(n,gamma));
  
  //fprintf(stdout,"n = %d, n_local = %d, local nnz = %d, total nnz = %d\n",n,n_local,localedges,tot_nnz);
  int64_t repeat_run;
  int64_t parts_idx;
  int64_t local_idx;
  double allgpartstime = 0;
  double streamtime = 0;
  genRandPerm(vorder, n_local);

  double streamstart = MPI_Wtime();
  for (repeat_run = 0; repeat_run < NUM_STREAMS; repeat_run++) {

    //First run: initialize with hashed (random) partition
    if (repeat_run == 0) {
      for (i = 0; i < n_local; ++i) {
        vert = (size_t)vorder[i];
        row = &rowptr[vert];
        mydegree = *(row+1) - *row;
        local_idx = offset + vert; 
        randidx = (PART_TYPE)irand(nparts);
        parts[local_idx] = randidx;
        partsize[randidx]++;
        partnnz[randidx] += mydegree;
      }
    }
    else {
      alpha *= ALPHA_EXP_RATE;
      for (i = 0; i < n_local; ++i) {
        memset(partscore, 0, nparts * sizeof(int64_t));
        memset(partcost, 0, nparts * sizeof(int64_t));
        vert = (size_t)vorder[i];
        local_idx = offset + vert; //VERTEX_LOCAL(global_vert_idx);
        row = &rowptr[vert];
        mydegree = *(row+1) - *row;
        oldpart = -1;
        if(mydegree > 0) {
          for (k = *row; k < ((*row)+mydegree); ++k) {
            parts_idx = VERTEX_OWNER(colidx[k])*g.nlocalverts + VERTEX_LOCAL(colidx[k]);
            PART_TYPE node_part = parts[parts_idx]; /////
            if (parts[parts_idx] >= 0) { partscore[node_part]++; }
          }
          for (s = 0; s < nparts; ++s) { partcost[s] = partscore[s] - alpha*(gamma/2)*pow(partsize[s],gamma-1); }
          best_part = 0;
          best_score = partcost[0];
          for (s = 1; s < nparts; ++s) { 
            curr_score = partcost[s]; 
            if (curr_score > best_score) { best_score = curr_score;  best_part = s; }
          }
          oldpart = parts[local_idx];
          parts[local_idx] = best_part;
          partsize[best_part]++; partnnz[best_part]+=mydegree;
          if (oldpart >= 0 && oldpart < nparts) { partsize[oldpart]--; partnnz[oldpart]-=mydegree; }
        } 
#if 0
        else { // empty vertex, assign randomly
          if (parts[local_idx]==-1) {
            emptyverts++;
            randidx = (PART_TYPE)irand(nparts);
            oldpart = parts[local_idx];
            parts[local_idx] = randidx; partsize[randidx]++;
            if (oldpart >= 0 && oldpart < nparts) { partsize[oldpart]--; partnnz[oldpart]-=mydegree; }
          }
        }
#endif
        if (i % (n_local / 2000) == 0) {
          for (l=0; l<nparts; ++l) { partsize_update[l] = partsize[l] - old_partsize[l]; }            
          MPI_Allreduce(MPI_IN_PLACE, partsize_update, nparts, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
          for (l=0; l<nparts; ++l) { old_partsize[l] += partsize_update[l]; partsize[l] = old_partsize[l]; }
        }
      }
    }
    for (l=0; l<nparts; ++l) { 
      partsize_update[l] = partsize[l] - old_partsize[l];
      partnnz_update[l] = partnnz[l] - old_partnnz[l];
      //fprintf(stdout,"partsize[%d] on rank %d is %d. partsizeupdate is %d. oldsize is %d\n", l, rank, partsize[l], partsize_update[l], old_partsize[l]);
    }
    MPI_Allreduce(MPI_IN_PLACE, partsize_update, nparts, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, partnnz_update, nparts, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

    for (l=0; l<nparts; ++l) { 
      old_partsize[l] += partsize_update[l]; 
      old_partnnz[l] += partnnz_update[l]; 
      partsize[l] = old_partsize[l];
      partnnz[l] = old_partnnz[l];
    }

    double allgstart= MPI_Wtime();
    MPI_Allgather(MPI_IN_PLACE, n_local, MPI_PART_TYPE, parts, n_local, MPI_PART_TYPE, MPI_COMM_WORLD);
    double allgstop = MPI_Wtime();
    allgpartstime += allgstop - allgstart;
    //if (rank == 0) { fprintf(stderr, "allgather time:               %f s\n", allgstop - allgstart); }

    //sanity check: manually compute partsizes
    if (1){//(0 || SANITY) {
      int64_t max_partsize = 0;
      int64_t min_partsize = n;
      int64_t *check_partsize = (int64_t*)malloc((nparts+1)*sizeof(int64_t));
      int64_t max_partnnz = 0;
      int64_t min_partnnz = tot_nnz;
      for (l=0; l<=nparts; ++l) { 
        check_partsize[l] = 0; 
      }
      for (i=0; i<n; ++i) {
        PART_TYPE mypart = parts[i];
        if (mypart == -1) { fprintf(stderr, "-1 :("); }
        assert(mypart>=0 && mypart<=nparts);
        check_partsize[mypart]++;
      }
      for (l=0; l<nparts; ++l) { 
        if (rank==l && VERBY) { fprintf(stdout,"partsize[%d] on rank %d is %" PRId64 ". check_partsize is %" PRId64 "\n", (int)l, (int)rank, partsize[l], check_partsize[l]); }
        assert(check_partsize[l] == partsize[l]); 
        if (check_partsize[l] > max_partsize) { max_partsize = check_partsize[l]; }
        if (check_partsize[l] < min_partsize) { min_partsize = check_partsize[l]; }
        if (partnnz[l] > max_partnnz) { max_partnnz = partnnz[l]; }
        if (partnnz[l] < min_partnnz) { min_partnnz = partnnz[l]; }
        //if (rank==0) { fprintf(stdout,"%d / %d\n",max_partsize, min_partsize); }
        //if (rank==0) { fprintf(stdout,"%d / %d\n",max_partnnz, min_partnnz); }
      }
      //if (rank==0) { fprintf(stdout,"max partsize = %d, min partsize = %d max/min partnnz = %d, %d ", max_partsize, min_partsize, max_partnnz, min_partnnz); }
      if (rank==0) { fprintf(stdout,"n balance: %f, nnz balance: %f\t", (float)max_partsize / min_partsize, (float)max_partnnz / min_partnnz); }
    }

    if (1) {//(repeat_run % 2 == 0 && repeat_run > 0 || SANITY) {
      mpi_compute_cut(rowptr, colidx, parts, nparts, n_local, offset);
    }
  }
  double streamstop= MPI_Wtime();
  streamtime += streamstop - streamstart;
  if (rank == 0) { fprintf(stderr, "stream time: %f,  per-stream time: %f \n", streamtime, streamtime/NUM_STREAMS); }

#if 0
  for (i=offset; i<offset+n_local; ++i) {
    if (parts[i] == nparts) {
      if (HI_RAND) { parts[i] = irand(nparts); }
      else { parts[i] = nparts; }
    }
  }
#endif
  MPI_Allgather(MPI_IN_PLACE, n_local, MPI_PART_TYPE, parts, n_local, MPI_PART_TYPE, MPI_COMM_WORLD);
  //MPI_Allgather(parts+offset, n_local, MPI_INT, parts, n_local, MPI_INT, MPI_COMM_WORLD);
  //MPI_Barrier(MPI_COMM_WORLD);
#if 0
  if (MAT_OUT &&(rank == 0)) {  // Print Parts
    FILE *PartFile;
    PartFile = fopen("parts.mat", "w");
    assert(PartFile != NULL);
    print_parts(PartFile, parts, n, n_local);
    fclose(PartFile);
  }

  if (SANITY) {
    // Sanity Checks
    for (i=0; i < g.nglobalverts; ++i) {
      assert(parts[i]>=0);
      assert(parts[i]<nparts);
    }
  }
#endif
  if (rank == 0) { fprintf(stderr, "allgather parts time:               %f s\n", allgpartstime); }
  //if (rank == 0) { fprintf(stderr, "comp  time:               %f s\n", comptime); }

}

// Random permutation generator. Move to another file.
int64_t* genRandPerm(int64_t* orderList, int64_t size) {
  assert(orderList);
  srand(time(NULL));
  // Generate 'identity' permutation
  int i;
  for (i = 0; i < size; i++) { orderList[i] = i; }
  shuffle_int(orderList, size);
  return orderList;
}

void shuffle_int(int64_t *list, int len) {
  int j;
  int64_t tmp;
  while(len) {
      j = irand(len);
      if (j != len - 1) {
        tmp = list[j];
        list[j] = list[len - 1];
        list[len - 1] = tmp;
      }
    len--;
  }
}

int irand(int n) {
  int r, rand_max = RAND_MAX - (RAND_MAX % n);
  while ((r = rand()) >= rand_max);
  return r / (rand_max / n);
}

float calc_dc(float alpha, float gamma, int64_t len) {
  return (alpha*pow(len + F_DELTA ,gamma)) - (alpha*pow(len,gamma));
}

int64_t mpi_compute_cut(size_t *rowptr, int64_t *colidx, PART_TYPE* parts, int nparts, int64_t n_local, int64_t offset) {
  size_t vert;
  int64_t mydegree;
  int64_t v_part;
  int64_t cutedges = 0;
  int64_t mytotedges = 0;
  int64_t mytotlodegedges = 0;

  size_t *cuts_per_part = (size_t*)malloc(nparts * sizeof(size_t));
  for (int i=0; i<nparts; i++) { cuts_per_part[i] = 0; }

  size_t *row;
  int64_t i;
  size_t k;
  int64_t emptyparts = 0;
  mytotedges = rowptr[n_local];
  for (i = 0; i < n_local; i++) {
    vert = i;
    row = &rowptr[vert];
    mydegree = (int64_t)(*(row+1) - *(row)); //nnz in row
    v_part = parts[vert+offset];
    if (v_part == -1) {
      v_part = 0;
      emptyparts++;
    }
    // count edges to other partitions
    for (k = *row; k < ((*row)+mydegree); ++k) {
      int64_t node = colidx[k];
      int64_t node_owner = VERTEX_OWNER(node);
      int64_t node_local_idx = VERTEX_LOCAL(node);
      int64_t parts_idx = node_owner*g.nlocalverts + node_local_idx;
      if (parts[parts_idx] < nparts) { mytotlodegedges++; } //count low degree edges
      if (parts[parts_idx] != v_part && parts[parts_idx] < nparts) { cutedges++; cuts_per_part[v_part]++; } 
    }
  }
  int64_t tot_cutedges;
  int64_t tot_lodegedges;
  MPI_Allreduce(&cutedges, &tot_cutedges, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&mytotlodegedges, &tot_lodegedges, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
  // fprintf(stdout,"offset: %d emptyparts = %d cutedges = %d totcutedges = %d tot edges=%d mylodegedges=%d totlodegedges=%d\n",offset, emptyparts,cutedges,tot_cutedges,mytotedges,mytotlodegedges,tot_lodegedges);
  if (rank == 0) {   fprintf(stdout,"total cutedges = %" PRId64 ", pct of total:%f pct of worstcase:%f \n", tot_cutedges, (float)tot_cutedges/tot_lodegedges, ((float)tot_cutedges/tot_lodegedges)/((float)(nparts-1)/nparts)); }

  MPI_Allreduce(MPI_IN_PLACE, cuts_per_part, nparts, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);

  if(rank==0) {
    fprintf (stdout, "Cuts/part: ");
    for (int i=0; i<nparts; i++) {
        fprintf (stdout, " [%d: %d] ",i,cuts_per_part[i]);
    }
    fprintf (stdout, "\n");
  }

  return tot_cutedges;
}

void free_graph_data_structure(void) { free_oned_csr_graph(&g); }

void make_graph_data_structure(const tuple_graph* const tg) { convert_graph_to_oned_csr(tg, &g); }

void remake_graph_data_structure(const tuple_graph* const tg) {
  convert_graph_to_oned_csr(tg, &g);
  const size_t nlocalverts = g.nlocalverts;
  g_oldq = (int64_t*)xmalloc(nlocalverts * sizeof(int64_t));
  g_newq = (int64_t*)xmalloc(nlocalverts * sizeof(int64_t));
  const int ulong_bits = sizeof(unsigned long) * CHAR_BIT;
  int64_t visited_size = (nlocalverts + ulong_bits - 1) / ulong_bits;
  g_visited = (unsigned long*)xmalloc(visited_size * sizeof(unsigned long));
  g_outgoing = (int64_t*)xMPI_Alloc_mem(coalescing_size * size * 2 * sizeof(int64_t));
  g_outgoing_counts = (size_t*)xmalloc(size * sizeof(size_t)) /* 2x actual count */;
  g_outgoing_reqs = (MPI_Request*)xmalloc(size * sizeof(MPI_Request));
  g_outgoing_reqs_active = (int*)xmalloc(size * sizeof(int));
  g_recvbuf = (int64_t*)xMPI_Alloc_mem(coalescing_size * 2 * sizeof(int64_t));
}

int64_t get_permed_vertex(int64_t id) { 
  return ((g_perm[id] % g.nlocalverts) * size + floor(g_perm[id]/(g.nglobalverts/size))); 
}

void print_graph() {
    char filename[256];
    sprintf(filename, "out_pcsr%02d.mat", rank);
    FILE *GraphFile;
    GraphFile = fopen(filename, "w");
    assert(GraphFile != NULL);
    print_graph_csr(GraphFile, g.rowstarts, g.column, g.nlocalverts);
    MPI_Barrier(MPI_COMM_WORLD);
    fclose(GraphFile);
}

// Print tuple graph as tuples
int print_graph_tuple(FILE* out, tuple_graph* tg, int rank) {
  int64_t v0, v1;
  packed_edge* result = tg->edgememory;
  packed_edge* edge;
  int i;
  for (i=0; i < tg->edgememory_size; ++i) {
    edge = &result[i];
    v0 = get_v0_from_edge(edge);
    v1 = get_v1_from_edge(edge);
    fprintf (out, "%d %d %d\n", (int)v0+1, (int)v1+1, 1);
  }
  return 1;
}

// Print CSR graph as tuples
int print_graph_csr(FILE* out, size_t *rowptr, int64_t *colidx, int n_local) {
  int i, k;
  int64_t src, dst;
  for (i=0; i<n_local; ++i) {
    for (k = rowptr[i]; k < rowptr[i+1]; ++k) {
      src = n_local*rank + i;
      dst = colidx[k]; 
      dst = ((dst-VERTEX_OWNER(dst))/size) + n_local*VERTEX_OWNER(dst);
      fprintf (out, "%d %d %d\n",(int)src+1,(int)dst+1,1);
    }
  }
  return 1;
}

// Print out the partition that each node belongs to
int print_parts(FILE* out, PART_TYPE* parts, int n, int n_local) {
  int i;
  for (i=0; i<n; ++i) {
    int node = i;
    int node_owner = VERTEX_OWNER(node);
    int node_local_idx = VERTEX_LOCAL(node);
    int parts_idx = node_owner*n_local + node_local_idx;
    int v_part = parts[parts_idx];
    fprintf (out, "%d %d\n",node+1,v_part+1);
  }
  return 1;
}
