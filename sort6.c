#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include <limits.h>
#include <stdint.h>
 
static inline int cmp(const void *p1, const void *p2)
{
    return *(int*)p1 > *(int*)p2 ? 1 : (*(int*)p1 == *(int*)p2 ? 0 : -1);
    //return *(int*)p1-*(int*)p2;
}
 
static inline void sort6_libqsort(int * d)
{
    qsort(d, 6, sizeof(int), cmp);
}
 
static inline void sort6_insertion_sort_v1(int * d){
 
    int j, i, imin;
    int tmp;
    for (j = 0 ; j < 5 ; j++){
        imin = j;
        for (i = j + 1; i < 6 ; i++){
            if (d[i] < d[imin]){
                imin = i;
            }
        }
        tmp = d[j];
        d[j] = d[imin];
        d[imin] = tmp;
    }
}
 
static inline void sort6_insertion_sort_v2(int *d){
    int i, j;
    for (i = 1; i < 6; i++) {
        int tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j-1]; j--)
            d[j] = d[j-1];
        d[j] = tmp;
    }
}
 
static inline void sort6_insertion_sort_unrolled(int *d){
    int j1;
    int tmp1 = d[1];
    for (j1 = 1; j1 >= 1 && tmp1 < d[j1-1]; j1--)
        d[j1] = d[j1-1];
    d[j1] = tmp1;
    int j2;
    int tmp2 = d[2];
    for (j2 = 2; j2 >= 1 && tmp2 < d[j2-1]; j2--)
        d[j2] = d[j2-1];
    d[j2] = tmp2;
    int j3;
    int tmp3 = d[3];
    for (j3 = 3; j3 >= 1 && tmp3 < d[j3-1]; j3--)
        d[j3] = d[j3-1];
    d[j3] = tmp3;
    int j4;
    int tmp4 = d[4];
    for (j4 = 4; j4 >= 1 && tmp4 < d[j4-1]; j4--)
        d[j4] = d[j4-1];
    d[j4] = tmp4;
    int j5;
    int tmp5 = d[5];
    for (j5 = 5; j5 >= 1 && tmp5 < d[j5-1]; j5--)
        d[j5] = d[j5-1];
    d[j5] = tmp5;
}

static inline void sort6_insertion_sort_avx(int* d) {
	__m256i src = _mm256_setr_epi32(d[0], d[1], d[2], d[3], d[4], d[5], 0, 0);
	__m256i index = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	__m256i shlpermute = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
	__m256i sorted = _mm256_setr_epi32(d[0], INT_MAX, INT_MAX, INT_MAX,
			INT_MAX, INT_MAX, INT_MAX, INT_MAX);
	__m256i val, gt, permute;
	unsigned j;
	 // 8 / 32 = 2^-2
#define ITER(I) \
		val = _mm256_permutevar8x32_epi32(src, _mm256_set1_epi32(I));\
		gt =  _mm256_cmpgt_epi32(sorted, val);\
		permute =  _mm256_blendv_epi8(index, shlpermute, gt);\
		j = ffs( _mm256_movemask_epi8(gt)) >> 2;\
		sorted = _mm256_blendv_epi8(_mm256_permutevar8x32_epi32(sorted, permute),\
				val, _mm256_cmpeq_epi32(index, _mm256_set1_epi32(j)))
	ITER(1);
	ITER(2);
	ITER(3);
	ITER(4);
	ITER(5);
	int x[8];
	_mm256_storeu_si256((__m256i*)x, sorted);
	d[0] = x[0]; d[1] = x[1]; d[2] = x[2]; d[3] = x[3]; d[4] = x[4]; d[5] = x[5];
#undef ITER
}

static inline void sort6_sorting_network_v1(int * d){
#define SWAP(x,y) if (d[y] < d[x]) { int tmp = d[x]; d[x] = d[y]; d[y] = tmp; }
    SWAP(1, 2);
    SWAP(0, 2);
    SWAP(0, 1);
    SWAP(4, 5);
    SWAP(3, 5);
    SWAP(3, 4);
    SWAP(0, 3);
    SWAP(1, 4);
    SWAP(2, 5);
    SWAP(2, 4);
    SWAP(1, 3);
    SWAP(2, 3);
#undef SWAP
}
 
#define min(x, y) (y ^ ((x ^ y) & -(x < y)))
#define max(x, y) (x ^ ((x ^ y) & -(x < y)))
 
static inline void sort2_sorting_network_v2(int *p0, int *p1)
{
    const int temp = min(*p0, *p1);
    *p1 = max(*p0, *p1);
    *p0 = temp;
}
 
static inline void sort3_sorting_network_v2(int *p0, int *p1, int *p2)
{
    sort2_sorting_network_v2(p0, p1);
    sort2_sorting_network_v2(p1, p2);
    sort2_sorting_network_v2(p0, p1);
}
 
static inline void sort4_sorting_network_v2(int *p0, int *p1, int *p2, int *p3)
{
    sort2_sorting_network_v2(p0, p1);
    sort2_sorting_network_v2(p2, p3);
    sort2_sorting_network_v2(p0, p2);  
    sort2_sorting_network_v2(p1, p3);  
    sort2_sorting_network_v2(p1, p2);  
}
 
static inline void sort6_sorting_network_v2(int *d)
{
    sort3_sorting_network_v2(d+0, d+1, d+2);
    sort3_sorting_network_v2(d+3, d+4, d+5);
    sort2_sorting_network_v2(d+0, d+3);  
    sort2_sorting_network_v2(d+2, d+5);  
    sort4_sorting_network_v2(d+1, d+2, d+3, d+4);  
}
#undef min
#undef max
 
 
static inline void sort6_sorting_network_v3(int * d){
#define min(x, y) (y ^ ((x ^ y) & -(x < y)))
#define max(x, y) (x ^ ((x ^ y) & -(x < y)))
#define SWAP(x,y) { int tmp = min(d[x], d[y]); d[y] = max(d[x], d[y]); d[x] = tmp; }
    SWAP(1, 2);
    SWAP(0, 2);
    SWAP(0, 1);
    SWAP(4, 5);
    SWAP(3, 5);
    SWAP(3, 4);
    SWAP(0, 3);
    SWAP(1, 4);
    SWAP(2, 5);
    SWAP(2, 4);
    SWAP(1, 3);
    SWAP(2, 3);
#undef SWAP
#undef min
#undef max
}
 
static inline void sort6_sorting_network_v4(int * d){
#define min(x, y) (y ^ ((x ^ y) & -(x < y)))
#define max(x, y) (x ^ ((x ^ y) & -(x < y)))
#define SWAP(x,y) { int tmp = min(d[x], d[y]); d[y] = max(d[x], d[y]); d[x] = tmp; }
    SWAP(1, 2);
    SWAP(4, 5);
    SWAP(0, 2);
    SWAP(3, 5);
    SWAP(0, 1);
    SWAP(3, 4);
    SWAP(1, 4);
    SWAP(0, 3);
    SWAP(2, 5);
    SWAP(1, 3);
    SWAP(2, 4);
    SWAP(2, 3);
#undef SWAP
#undef min
#undef max
}

static inline void sort6_sorting_net_simple_swap(int * d){
#define min(x, y) (y > x ? x : y)
#define max(x, y) (y > x ? y : x)
#define SWAP(x,y) { int a = min(d[x], d[y]); int b = max(d[x], d[y]); d[x] = a; d[y] = b;}
    SWAP(1, 2);
    SWAP(4, 5);
    SWAP(0, 2);
    SWAP(3, 5);
    SWAP(0, 1);
    SWAP(3, 4);
    SWAP(1, 4);
    SWAP(0, 3);
    SWAP(2, 5);
    SWAP(1, 3);
    SWAP(2, 4);
    SWAP(2, 3);
#undef SWAP
#undef min
#undef max
}

static inline void sort6_rank_order(int *d) {
    int e[6];
    memcpy(e,d,6*sizeof(int));
    int o0 = (d[0]>d[1])+(d[0]>d[2])+(d[0]>d[3])+(d[0]>d[4])+(d[0]>d[5]);
    int o1 = (d[1]>=d[0])+(d[1]>d[2])+(d[1]>d[3])+(d[1]>d[4])+(d[1]>d[5]);
    int o2 = (d[2]>=d[0])+(d[2]>=d[1])+(d[2]>d[3])+(d[2]>d[4])+(d[2]>d[5]);
    int o3 = (d[3]>=d[0])+(d[3]>=d[1])+(d[3]>=d[2])+(d[3]>d[4])+(d[3]>d[5]);
    int o4 = (d[4]>=d[0])+(d[4]>=d[1])+(d[4]>=d[2])+(d[4]>=d[3])+(d[4]>d[5]);
    int o5 = 15-(o0+o1+o2+o3+o4);
    d[o0]=e[0]; d[o1]=e[1]; d[o2]=e[2]; d[o3]=e[3]; d[o4]=e[4]; d[o5]=e[5];
}
 
static inline void sort6_rank_order_reg(int *d) {
    register int x0,x1,x2,x3,x4,x5;
    x0 = d[0];
    x1 = d[1];
    x2 = d[2];
    x3 = d[3];
    x4 = d[4];
    x5 = d[5];
    int o0 = (x0>x1)+(x0>x2)+(x0>x3)+(x0>x4)+(x0>x5);
    int o1 = (x1>=x0)+(x1>x2)+(x1>x3)+(x1>x4)+(x1>x5);
    int o2 = (x2>=x0)+(x2>=x1)+(x2>x3)+(x2>x4)+(x2>x5);
    int o3 = (x3>=x0)+(x3>=x1)+(x3>=x2)+(x3>x4)+(x3>x5);
    int o4 = (x4>=x0)+(x4>=x1)+(x4>=x2)+(x4>=x3)+(x4>x5);
    int o5 = 15-(o0+o1+o2+o3+o4);
    d[o0]=x0; d[o1]=x1; d[o2]=x2; d[o3]=x3; d[o4]=x4; d[o5]=x5;
}

static inline void sort6_rank_order_reuse(int *d) {
    register int x0,x1,x2,x3,x4,x5;
    x0 = d[0];
    x1 = d[1];
    x2 = d[2];
    x3 = d[3];
    x4 = d[4];
    x5 = d[5];
    int o0, o1, o2, o3, o4, o5;
    o0 = o1 = o2 = o3 = o4 = o5 = 0;
    int ci1, ci2, ci3, ci4;
    ci1 = x1 >= x0; ci2 = x2 >= x0; ci3 = x3 >= x0; ci4 = x4 >= x0;
    o0 += (1-ci1) + (1-ci2) + (1-ci3) + (1-ci4) + (x0 > x5);
    o1 += ci1; o2 += ci2; o3 += ci3; o4 += ci4;
    ci2 = x2 >= x1; ci3 = x3 >= x1; ci4 = x4 >= x1;
    o1 += (1-ci2) + (1-ci3) + (1-ci4) + (x1>x5);
    o2 += ci2; o3 += ci3; o4 += ci4;
    ci3 = x3 >= x2; ci4 = x4 >= x2;
    o2 += (1-ci3) + (1-ci4) + (x2>x5);
    o3 += ci3; o4 += ci4;
    ci4 = x4 >= x3;
    o3 += (1-ci4) + (x3>x5);
    o4 += ci4;
    o4 += (x4>x5);
    o5 = 15-o0-o1-o2-o3-o4;
    d[o0]=x0; d[o1]=x1; d[o2]=x2; d[o3]=x3; d[o4]=x4; d[o5]=x5;
}

static inline void sort6_rank_order_loop(int *d) {
	int e[6];
	int p[6];
	p[0] = 0; p[1] = 1; p[2] = 2; p[3] = 3; p[4] = 4; p[5] = 5;
	int o5 = 15;
	for (int i = 0; i < 5; i++) {
		int o = 0;
		for (int j=0;j<6;j++) o += d[i] > d[j];
		o5 -= p[o];
		e[p[o]++] = d[i];
	}
	e[o5] = d[5];
	for (int i=0; i<6; i++) d[i] = e[i];
}

static inline void sort6_rank_order_avx(int* d) {
	__m256i ror = _mm256_setr_epi32(5, 0, 1, 2, 3, 4, 6, 7);
	__m256i one = _mm256_set1_epi32(1);
	__m256i src = _mm256_setr_epi32(d[0], d[1], d[2], d[3], d[4], d[5], INT_MAX, INT_MAX);
	__m256i rot = src;
    __m256i index = _mm256_setzero_si256();
	__m256i gt, permute;
    __m256i shl = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 6, 6);
    __m256i dstIx = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
	__m256i srcIx = dstIx;
    __m256i eq = src;
    __m256i rotIx = _mm256_setzero_si256();
#define INC(I)\
	rot = _mm256_permutevar8x32_epi32(rot, ror);\
	gt = _mm256_cmpgt_epi32(src, rot);\
	index = _mm256_add_epi32(index, _mm256_and_si256(gt, one));\
    eq = _mm256_permutevar8x32_epi32(eq, shl);\
    index = _mm256_add_epi32(index, _mm256_and_si256(\
                _mm256_cmpeq_epi32(src, eq), one))
	INC(0);
	INC(1);
	INC(2);
	INC(3);
	INC(4);
    int e[6];
    e[0] = d[0]; e[1] = d[1]; e[2] = d[2]; e[3] = d[3]; e[4] = d[4]; e[5] = d[5];
    int i[8];
    _mm256_storeu_si256((__m256i*)i, index);
    d[i[0]] = e[0]; d[i[1]] = e[1]; d[i[2]] = e[2]; d[i[3]] = e[3]; d[i[4]] = e[4]; d[i[5]] = e[5];
}

static inline void sort6_inlined_bubble(int * d){
#define SWAP(x,y) { int dx = d[x], dy = d[y], tmp; tmp = d[x] = dx < dy ? dx : dy; d[y] ^= dx ^ tmp; }
    SWAP(0,1); SWAP(1,2); SWAP(2,3); SWAP(3,4); SWAP(4,5);
    SWAP(0,1); SWAP(1,2); SWAP(2,3); SWAP(3,4);
    SWAP(0,1); SWAP(1,2); SWAP(2,3);
    SWAP(0,1); SWAP(1,2);
    SWAP(0,1);
#undef SWAP
}


static inline void sort6_insertion_sort_unrolled_v2(int * d){
    //#define ITER(x) { if (t < d[x]) { d[x+1] = d[x]; d[x] = t; } }
    //Faster on x86, probably slower on ARM or similar:
    #define ITER(x) { d[x+1] ^= t < d[x] ? d[x] ^ d[x+1] : 0; d[x] = t < d[x] ? t : d[x]; }
        int t;
        t = d[1]; ITER(0);
        t = d[2]; ITER(1); ITER(0);
        t = d[3]; ITER(2); ITER(1); ITER(0);
        t = d[4]; ITER(3); ITER(2); ITER(1); ITER(0);
        t = d[5]; ITER(4); ITER(3); ITER(2); ITER(1); ITER(0);
    #undef ITER
}

// to be debugged
static inline void sort6_shellsort(int * d) {
    char j, i, inc;
    int tmp;
    for (inc = 4; inc > 0; inc -= 3) {
        for (i = inc; i < 5; i++) {
            tmp = d[i];
            j = i;
            while (j >= inc && d[j - inc] > tmp) {
                d[j] = d[j - inc];
                j -= inc;
            }
            d[j] = tmp;
        }
    }
}

static inline void sort6_fast_network(int * d) {
//#define SWAP(x,y) asm("mov %0, %2 ; cmp %1, %0 ; cmovg %1, %0 ; cmovg %2, %1" : "=r" (x), "=r" (y), "=r" (tmp) : "0" (x), "1" (y) : "cc");
//    register int x0,x1,x2,x3,x4,x5,tmp;
#define SWAP(x,y) { int dx = x, dy = y, tmp; tmp = x = dx < dy ? dx : dy; y ^= dx ^ tmp; }
    register int x0,x1,x2,x3,x4,x5;
    x1 = d[1];
    x2 = d[2];
    SWAP(x1, x2);
    x4 = d[4];
    x5 = d[5];
    SWAP(x4, x5);
    x0 = d[0];
    SWAP(x0, x2);
    x3 = d[3];
    SWAP(x3, x5);
    SWAP(x0, x1);
    SWAP(x3, x4);
    SWAP(x1, x4);
    SWAP(x0, x3);
    d[0] = x0;
    SWAP(x2, x5);
    d[5] = x5;
    SWAP(x1, x3);
    d[1] = x1;
    SWAP(x2, x4);
    d[4] = x4;
    SWAP(x2, x3);
    d[2] = x2;
    d[3] = x3;
 
#undef SWAP
#undef min
#undef max
}
 

static inline void sort6_fast_network_simplified(int * d) {
//#define SWAP(x,y) asm("mov %0, %2 ; cmp %1, %0 ; cmovg %1, %0 ; cmovg %2, %1" : "=r" (x), "=r" (y), "=r" (tmp) : "0" (x), "1" (y) : "cc");
#define SWAP(x,y) { int dx = d[x]; int dy = d[y]; int tmp = d[x] = dx < dy ? dx : dy; d[y] ^= dx ^ tmp; }
    SWAP(1, 2);
    SWAP(4, 5);
    SWAP(0, 2);
    SWAP(3, 5);
    SWAP(0, 1);
    SWAP(3, 4);
    SWAP(1, 4);
    SWAP(0, 3);
    SWAP(2, 5);
    SWAP(1, 3);
    SWAP(2, 4);
    SWAP(2, 3);
#undef SWAP
#undef min
#undef max
}

static inline unsigned long long rdtsc(void)
{
    unsigned long long int x;
    asm volatile ("rdtsc; shlq $32, %%rdx; orq %%rdx, %0" : "=a" (x) : : "rdx");
    return x;
}
 
void ran_fill(int n, int *a) {
    static int seed = 76521;
    while (n--) *a++ = (seed = seed *1812433253 + 12345);
}

int order6(int *d, int* check) {
#define ORDER_PAIR(I) (d[I] <= d[I+1])
    int order = ORDER_PAIR(0) && ORDER_PAIR(1) && ORDER_PAIR(2) && ORDER_PAIR(3) &&
        ORDER_PAIR(4);
    int bijective = 1;
    // all([x in d for x in check])
    // all([x in check for x in d])
    for (int i = 0; bijective && i < 6; i++) {
        int present = 0;
        for (int j = 0; !present && j < 6; j++)
            present = d[j] == check[i] ? 1 : 0;
        bijective = !present ? 0 : 1;
    }
    for (int i = 0; bijective && i < 6; i++) {
        int extras = 1;
        for (int j = 0; extras && j < 6; j++)
            extras = d[i] == check[j] ? 0 : 1;
        bijective = extras ? 0 : 1;
    }
    return bijective && order;
#undef ORDER_PAIR
}
 
#define NTESTS 16384
int main(){
#define TEST(variant, description) {\
    int i;\
    int d[6*NTESTS];\
    int check[6*NTESTS];\
    sort6_##variant(d);\
    ran_fill(6*NTESTS, d);\
    memcpy(check, d, sizeof(int) * 6 * NTESTS);\
    unsigned long long cycles = rdtsc();\
    for (i = 0; i < 6*NTESTS ; i+=6){\
        sort6_##variant(d+i);\
    }\
    cycles = rdtsc() - cycles;\
    printf(description " : %.2lf\n", (double)cycles/(double)NTESTS);\
    {\
    int passed = 1;\
    for (i = 0; i < 6*NTESTS ; i+=6) { \
        if (!order6(d + i, check + i)) { \
            printf("d%d : %d %d %d %d %d %d\n", i, \
                    d[i+0], d[i+1], d[i+2], \
                    d[i+3], d[i+4], d[i+5]); \
            passed = 0;\
        }\
    } \
    int eq[] = { 5, 1, 2, 4, 5, 0 };\
    int check2[] = { 5, 1, 2, 4, 5, 0 };\
    sort6_##variant(eq);\
    if (!order6(eq, check2)) {\
        printf("eq : %d %d %d %d %d %d\n",\
                eq[0], eq[1], eq[2], \
                eq[3], eq[4], eq[5]); \
        passed = 0;\
    }\
    if (!passed) {\
            printf(" FAILED\n");\
    }\
    }\
}
 
TEST(libqsort,                "Direct call to qsort library function     ");
TEST(insertion_sort_v1,       "Naive implementation (insertion sort)     ");
TEST(insertion_sort_v2,       "Insertion Sort (Daniel Stutzbach)         ");
TEST(insertion_sort_unrolled, "Insertion Sort Unrolled                   ");
TEST(insertion_sort_avx,      "Insertion Sort by AVX                     ");
TEST(rank_order,              "Rank Order                                ");
TEST(rank_order_reg,          "Rank Order with registers                 ");
TEST(rank_order_reuse,        "Rank Order with reuse                     ");
TEST(rank_order_loop,         "Rank Order in loop                        ");
TEST(rank_order_avx,          "Rank Order with avx                       ");
TEST(sorting_network_v1,      "Sorting Networks (Daniel Stutzbach)       ");
TEST(sorting_network_v2,      "Sorting Networks (Paul R)                 ");
TEST(sorting_network_v3,      "Sorting Networks 12 with Fast Swap        ");
TEST(sorting_network_v4,      "Sorting Networks 12 reordered Swap        ");
TEST(sorting_net_simple_swap, "Sorting Networks 12 reordered Simple Swap ");
TEST(fast_network,            "Reordered Sorting Network w/ fast swap    ");
TEST(fast_network_simplified, "Reordered Sorting Network w/ fast swap V2 ");
TEST(inlined_bubble,          "Inlined Bubble Sort (Paolo Bonzini)       ");
TEST(insertion_sort_unrolled_v2, "Unrolled Insertion Sort (Paolo Bonzini)   ");
 
return 0;
 
}
