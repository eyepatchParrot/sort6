	rot = _mm256_permutevar8x32_epi32(rot, ror);\
	gt = _mm256_cmpgt_epi32(src, rot);\
	index = _mm256_add_epi32(index, _mm256_and_si256(gt, one));\
    eq = _mm256_permutevar8x32_epi32(eq, shl);\
    index = _mm256_add_epi32(index, _mm256_and_si256(\
                _mm256_cmpeq_epi32(src, eq), one))

#define INC(I)\
    rot = _mm256_permutevar8x32_epi32(src, rotIx);\
    rotIx = _mm256_add_epi32(rotIx, one);\
	gt = _mm256_and_si256(one, _mm256_cmpgt_epi32(src, rot));\
    eq = _mm256_and_si256(one,\
            _mm256_and_si256(_mm256_cmpeq_epi32(src, rot),\
                _mm256_cmpgt_epi32(dstIx, rotIx)));\
    index = _mm256_add_epi32(_mm256_add_epi32(index, gt), eq)
