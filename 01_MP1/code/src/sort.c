#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <omp.h>
#include <math.h>

#include "sort.h"
#include "edgelist.h"
#include "timer.h"
int numThreads = 2;

// Order edges by id of a source vertex, 
// using the Counting Sort
// Complexity: O(E + V)
int check_correctness(long long int numE, struct Edge *edges_sorted){
    long long int CORRECT = 1;
    long long int g;
    for(g=1;g<numE;g++){
        if(edges_sorted[g].src < edges_sorted[g-1].src){
            CORRECT = 0;
            break;
        }
    }
    return CORRECT;
}

void countSortEdgesBySource(struct Edge *edges_sorted, struct Edge *edges, int numVertices, int numEdges) {
                                                                          //     k                n
    long long int i;
    long long int t;
    long long int key;
    long long int pos;
    long long int numT = numThreads;
    long long int numV = numVertices; // k: the max vertice nubmer
    long long int numE = numEdges;    // n: how many edge relations
    omp_set_nested(1); // enable nested loop for omp
    omp_set_num_threads(numT);

    // auxiliary arrays, allocated at the start up of the program
    long long int *vertex_cnt = (long long int*)malloc(numT*numV*sizeof(long long int)); // needed for Counting Sort
    long long int base = 0;
#pragma omp parallel private(i, key)
{
    long long int tid = omp_get_thread_num();
    for(i = 0; i < numV; i++) {
        vertex_cnt[tid*numV+i] = 0; //simluate 2D array
    }

    // count occurrence of key: id of a source vertex
    long long int tid_start = tid*(numE/numT);
    long long int tid_end   = (tid != numT-1)?tid_start+(numE/numT):tid_start+(numE/numT)+(numE%numT);

    for(i = tid_start; i < tid_end; i++){
        key = (long long int)edges[i].src;
        vertex_cnt[tid*numV+key]++;
    }    
} 
    //transform to cumulative sum
    for(i = 0; i < numV; ++i) {
        for(t = 0 ; t < numT ; t++){
            base += vertex_cnt[t*numV+i];
            vertex_cnt[t*numV+i] = base;
        }
    }
#pragma omp parallel private(i, key, pos)
{
    
    long long int tid = omp_get_thread_num();
    long long int tid_start = tid*(numE/numT);
    long long int tid_end   = (tid != numT-1)?tid_start+(numE/numT):tid_start+(numE/numT)+(numE%numT);
   
    for(i = tid_end-1; i >= tid_start; i--){
        key = (long long int)edges[i].src;
        pos = vertex_cnt[tid*numV+key]-1;
        edges_sorted[pos] = edges[i];
        vertex_cnt[tid*numV+key]--;
    }
}    
    printf("result's correctness: %d\n", check_correctness(numE, edges_sorted));
    free(vertex_cnt);
}

int GetDigit(int num, int digit, int d){
    int i;
    int *num_array = (int*)malloc(d*sizeof(int));
    // printf("num=%d,(tid=%d)", num, omp_get_thread_num());
    for(i = 0 ; i < d ; i++){
        num_array[i] = num%10;
        num/=10;
    }
    // printf("digit=%d, ans=%d(tid=%d)\n", digit, num_array[digit], omp_get_thread_num());
    return num_array[digit];
}

void radixSortEdgesBySource(struct Edge *edges_sorted, struct Edge *edges, int numVertices, int numEdges){
    int d = 1;// how many piecewise digits of maximun number
    int digit;
    long long int i;
    long long int t;
    long long int key;
    long long int pos;
    long long int numT = numThreads;
    long long int numE = numEdges;    // n: how many edge relations
    long long int base;
    int bitewise = 8;
    int bitewise_length = pow(2, bitewise);
    // printf("numVertices = %d\n", numVertices);
    while(numVertices != 0){ // how many digits in binary of the maximun number
        numVertices = numVertices >> bitewise;
        d++;
        // printf("numVertices=%d, d=%d\n", numVertices, d);
    }
    // printf("d=%d\n", d);
    omp_set_nested(1); // enable nested loop for omp
    omp_set_num_threads(numT);
    long long int *vertex_cnt = (long long int*)malloc(numT*bitewise_length*sizeof(long long int)); // needed for Counting Sort
    struct Timer* timer = (struct Timer*) malloc(sizeof(struct Timer));

    for(digit = 0 ; digit < d ; digit++){
        base = 0;                        // init for each digit
        for(i = 0; i < numT*bitewise_length; i++){
            vertex_cnt[i] = 0;
        }
        Start(timer);
#pragma omp parallel private(i, key)
{
        long long int tid = omp_get_thread_num();
        long long int tid_start = tid*(numE/numT);
        long long int tid_end   = (tid != numT-1)?tid_start+(numE/numT):tid_start+(numE/numT)+(numE%numT);
        for(i=tid_start; i < tid_end; i++){
            key = (edges[i].src >> bitewise*digit)%bitewise_length;    // get value of digits
            vertex_cnt[tid*bitewise_length+key]++; 
        }
}
        Stop(timer);
        // printf("loop2 %-51f\n", Seconds(timer));
        for(i = 0; i < bitewise_length ; i++){
            for(t = 0 ; t < numT ; t++){
                base += vertex_cnt[t*bitewise_length+i];
                vertex_cnt[t*bitewise_length+i] = base;
            }
        }
        Start(timer);
#pragma omp parallel private(i, key, pos)
{
        long long int tid = omp_get_thread_num();
        long long int tid_start = tid*(numE/numT);
        long long int tid_end   = (tid != numT-1)?tid_start+(numE/numT):tid_start+(numE/numT)+(numE%numT);
        for(i = tid_end-1; i >= tid_start; i--){
            key = (edges[i].src >> bitewise*digit)%bitewise_length;
            pos = vertex_cnt[tid*bitewise_length+key]-1;
            edges_sorted[pos] = edges[i];
            vertex_cnt[tid*bitewise_length+key]--;
        }
}
        Stop(timer);
        // printf("loop4 %-51f\n", Seconds(timer));
        # pragma omp for 
        for(i = 0 ; i < numE; i++){
            edges[i] = edges_sorted[i];
        }
    }
    printf("result's correctness: %d\n", check_correctness(numE, edges_sorted));
    free(vertex_cnt);

// ==== Below is Decimal version, which is slow ===
//     int d = 0; // how many digits of max number
//     int digit;
//     long long int i;
//     long long int t;
//     long long int numT = numThreads;
//     long long int base;
//     while(numVertices != 0){          // k: numVertices
//         numVertices/=10;
//         d++;
//     }
//     long long int numE = numEdges;    // n: how many edge relations
//     omp_set_nested(1); // enable nested loop for omp
//     omp_set_num_threads(numT);
//     long long int *vertex_cnt = (long long int*)malloc(numT*10*sizeof(long long int)); // needed for Counting Sort
//     for(digit = 0;  digit < d; digit++){
//         base = 0;                        // init for each digit
//         for(i = 0; i < numT*10; i++){
//             vertex_cnt[i] = 0;
//         }
// #pragma omp parallel private(i)
// {
//         long long int tid = omp_get_thread_num();
//         long long int tid_start = tid*(numE/numT);
//         long long int tid_end   = (tid != numT-1)?tid_start+(numE/numT):tid_start+(numE/numT)+(numE%numT);
//         for(i=tid_start; i < tid_end; i++){
//             vertex_cnt[tid*10+GetDigit(edges[i].src, digit, d)]++; 
//         }
// }
//         for(i = 0; i < 10 ; i++){
//             for(t = 0 ; t < numT ; t++){
//                 base += vertex_cnt[t*10+i];
//                 vertex_cnt[t*10+i] = base;
//             }
//         }
// #pragma omp parallel private(i)
// {
//         long long int tid = omp_get_thread_num();
//         long long int tid_start = tid*(numE/numT);
//         long long int tid_end   = (tid != numT-1)?tid_start+(numE/numT):tid_start+(numE/numT)+(numE%numT);
//         for(i = tid_end-1; i >= tid_start; i--){
//             // printf("d=%d, tid=%lli, i=%lli, v=%lli\n", GetDigit(edges[i].src, digit, d), tid, i, vertex_cnt[GetDigit(edges[i].src, digit, d)]);
//             edges_sorted[vertex_cnt[tid*10+GetDigit(edges[i].src, digit, d)]-1] = edges[i];
//             vertex_cnt[tid*10+GetDigit(edges[i].src, digit, d)]--;
//         }
// }
//         # pragma omp for 
//         for(i = 0 ; i < numE; i++){
//             edges[i] = edges_sorted[i];
//         }
//     }
//     printf("result's correctness: %d\n", check_correctness(numE, edges_sorted));
//     free(vertex_cnt);
}
