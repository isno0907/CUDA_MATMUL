#define TS    16
#define WPT   4 
#define RTS   4

// __kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
//   int i = get_global_id(0); // row index of C
//   int j = get_global_id(1); // column index of C
//   if (i >= M || j >= N) return; // boundary check

//   C[i * N + j] = 0;
//   for (int k = 0; k < K; k++) {
//     C[i * N + j] += A[i * K + k] * B[k * N + j];
//   }
// }

// __kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
//   const int row = get_local_id(0); 
//   const int col = get_local_id(1);

//   const int global_row = TS * get_group_id(0) + row;
//   const int global_col = TS * get_group_id(1) + col;

//   __local float Asub[TS][TS];
//   __local float Bsub[TS][TS];


//   float intermediate_val = 0.0f;
//   const int num_tiles = K / TS;

//   for (int t = 0; t < num_tiles; ++t) {
//     const int t_row = TS * t + row;
//     const int t_col = TS * t + col;
//     Asub[row][col] = A[global_row * K + t_col];
//     Bsub[row][col] = B[t_row * N + global_col];

//     barrier(CLK_LOCAL_MEM_FENCE);

//     for (int k = 0; k < TS; ++k) {
//       intermediate_val += Asub[row][k] * Bsub[k][col];
//     }
//     barrier(CLK_LOCAL_MEM_FENCE);
//   }
    
  
//   C[global_row * N + global_col] = intermediate_val;
// }

__kernel void sgemm(__global float *A, __global float *B, __global float *C, int M, int N, int K) {
  const int row = get_local_id(0);
  const int col = get_local_id(1);

  const int global_row = TS * get_group_id(0) + row;
  const int global_col = TS * get_group_id(1) + col;

  __local float Asub[TS][TS];
  __local float Bsub[TS][TS];


  float intermediate_val[WPT];
  for (int w = 0; w < WPT; ++w) {
    intermediate_val[w] = 0.0f;
  }
  const int num_tiles = (K-1) / TS + 1;

  for (int t = 0; t < num_tiles; ++t) {
    for (int w = 0; w < WPT; ++w) {
      const int t_row = TS * t + row;
      const int t_col = TS * t + col;
      // printf("global_row = %d\n", global_row);
      if (t_col >= K || (global_row + w*RTS) >= M) Asub[row + w*RTS][col] = 0.0f;
      else Asub[row + w*RTS][col] = A[(global_row + w*RTS) * K + t_col];
      if (global_col >= N || (t_row + w*RTS) >= K) Bsub[row + w*RTS][col] = 0.0f;
      else Bsub[row + w*RTS][col] = B[(t_row + w*RTS) * N + global_col];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k = 0; k < TS; ++k) {
      for (int w = 0; w < WPT; ++w) {
        intermediate_val[w] += Asub[row + w*RTS][k] * Bsub[k][col];
      }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

  }
  
  for (int w = 0; w < WPT; ++w) {
    if ((global_row + w*RTS) < M && global_col < N) C[(global_row + w*RTS) * N + global_col] = intermediate_val[w];
  }
}