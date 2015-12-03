#include <cusparse_v2.h>
#include <stdio.h>
#include <stdbool.h>

#define N 18

// error check macros
#define CUSPARSE_CHECK(x) {cusparseStatus_t _c=x; if (_c != CUSPARSE_STATUS_SUCCESS) {printf("cusparse fail: %d, line: %d\n", (int)_c, __LINE__); exit(-1);}}

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


int main(int argc, char *argv[]) {

    cusparseHandle_t hndl;
    cusparseStatus_t stat;
    cusparseMatDescr_t descrA;
    cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;

    int *nnzPerRow;
    int *csrRowPtrA, *csrColIndA;
    float *csrValA, *d_sparse;
    int nnzA;
    int * nnzTotalDevHostPtr = &nnzA;


    cudaMalloc((void**)&nnzPerRow, N*sizeof(int));
    cudaMalloc((void**)&csrRowPtrA, N*sizeof(int));
    cudaMalloc((void**)&d_sparse, N*N*sizeof(float));
    cudaCheckErrors("cudaMalloc fail");

    float h_sparse[N][N]; FILE * f;
    f = fopen("H_matrix", "r");
    int i=0; char x;
    while((x=fgetc(f)) != EOF) {
        if ((x == '1') | (x == '0')) {
           h_sparse[i/N][i%N] = (float)(x-'0');
           i++;
           }
    }
    fclose(f);

//    for (i=0; i<N*N; i++) {
//    if (i%N == 0) printf("\n");
//    printf("%f\t", h_sparse[i/N][i%N]);
//}

    cudaMemcpy(d_sparse, h_sparse, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy fail");

    CUSPARSE_CHECK(cusparseCreate(&hndl));
    stat = cusparseCreateMatDescr(&descrA);
    CUSPARSE_CHECK(stat);
    stat = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    CUSPARSE_CHECK(stat);
    stat = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    CUSPARSE_CHECK(stat);
    cusparseSnnz(hndl, dir, N, N,
                descrA, d_sparse, N,
                nnzPerRow, nnzTotalDevHostPtr);
    if (nnzTotalDevHostPtr != NULL) {
        nnzA = * nnzTotalDevHostPtr;
    }

    int h_nnzPerRow[N];
    cudaMemcpy(h_nnzPerRow, nnzPerRow, sizeof(int)*N, cudaMemcpyDeviceToHost);
    for (int i=0; i<N; i++) {
        printf("%d\n", h_nnzPerRow[i]);
    }

    cudaMalloc((void**)&csrValA, sizeof(float)*nnzA);
    cudaMalloc((void**)&csrColIndA, sizeof(int)*nnzA);
    cudaCheckErrors("cudaMalloc fail");
    cusparseSdense2csr(hndl, N, N,
                       descrA, d_sparse,
                       N, nnzPerRow,
                       csrValA, csrRowPtrA, csrColIndA);

    float * h_csrValA; int * h_csrRowPtrA, * h_csrColIndA;
    h_csrValA = (float *)malloc(nnzA*sizeof(float));
    h_csrColIndA = (int *)malloc(nnzA*sizeof(int));
    h_csrRowPtrA = (int *)malloc(N*sizeof(int));
    cudaMemcpy(h_csrValA, csrValA, nnzA*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrColIndA, csrColIndA, nnzA*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_csrRowPtrA, csrRowPtrA, N*sizeof(float), cudaMemcpyDeviceToHost);    
    for (int i=0;i<nnzA;i++) {
        printf("\n%f\t%d", h_csrValA[i], h_csrColIndA[i]); }
    for (int i =0; i<N; i++) {
        printf("\n%d", h_csrRowPtrA[i]); }


    float h_X[N][N], *h_Y, *X, *Y;
    h_Y = (float *)malloc(N*N*sizeof(float));
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            h_X[i][j] = (i==j) ? 1 : 0;
        }
    }

//   for (int i=0; i<N; i++) {
//       printf("\n");
//       for (int j=0; j<N; j++) {
//            printf("%f\t", h_X[i][j]);
//        }
//    }

    cudaMalloc((void**)&X, N*N*sizeof(float));
    cudaMalloc((void**)&Y, N*N*sizeof(float));
    cudaCheckErrors("cudaMalloc fail");
    cudaMemcpy(X, h_X, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy fail");

    cusparseOperation_t operationA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSolveAnalysisInfo_t info;
    stat = cusparseCreateSolveAnalysisInfo(&info);
    CUSPARSE_CHECK(stat);
    stat = cusparseScsrsm_analysis(hndl, operationA, N, nnzA,
                            descrA, csrValA, csrRowPtrA, csrColIndA,
                            info);
    CUSPARSE_CHECK(stat);

    float p = 1;
    const float *alpha = &p;
    stat = cusparseScsrsm_solve(hndl, operationA, N, N,
                                alpha,
                                descrA,
                                csrValA, csrRowPtrA, csrColIndA,
                                info, X, N, Y, N);
    CUSPARSE_CHECK(stat);

    cudaMemcpy(h_Y, Y, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy fail");

//    for(int i=0; i<N*N; i++) {
//        if (i%N==0) printf("\n");
//        printf("%f\t", h_Y[i]);
//}
}
