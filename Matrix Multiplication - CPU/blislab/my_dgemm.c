/*
 * --------------------------------------------------------------------------
 * BLISLAB 
 * --------------------------------------------------------------------------
 * Copyright (C) 2016, The University of Texas at Austin
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 * bl_dgemm.c
 *
 *
 * Purpose:
 * this is the main file of blislab dgemm.
 *
 * Todo:
 *
 *
 * Modification:
 *      bryan chin - ucsd
 *      changed to row-major order  
 *      handle arbitrary  size C
 * */

#include <stdio.h>

#include "bl_dgemm_kernel.h"
#include "bl_dgemm.h"

const char* dgemm_desc = "my blislab ";

/* 
 * pack one subpanel of A
 *
 * pack like this 
 * if A is row major order
 *
 *     a c e g
 *     b d f h
 *     i k m o
 *     j l n p
 *     q r s t
 *     
 * then pack into a sub panel
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses). 
 * - down each column
 * - then next column in sub panel
 * - then next sub panel down (on subseqent call)
 
 *     a c e g  < each call packs one
 *     b d f h  < subpanel
 *     -------
 *     i k m o
 *     j l n p
 *     -------
 *     q r s t
 *     0 0 0 0
 */
static inline void packA_mcxkc_d(
        int m,
        int k,
        double *XA,
        int ldXA,
        double *packA
        )
{
  int i, p;
  double *a_pntr[DGEMM_MR];
  for (i = 0; i < m; i++)
    a_pntr[i] = XA + (i * ldXA);

  for (p = 0; p < k; p++)
  {
    for (i = 0; i < m; i++)
    {
      *packA = *a_pntr[i];
      packA++;
      a_pntr[i]++;
    }
  }
}


/*
 * --------------------------------------------------------------------------
 */

/* 
 * pack one subpanel of B
 * 
 * pack like this 
 * if B is 
 *
 * row major order matrix
 *     a b c j k l s t
 *     d e f m n o u v
 *     g h i p q r w x
 *
 * each letter represents sequantial
 * addresses in the packed result
 * (e.g. a, b, c, d are sequential
 * addresses). 
 *
 * Then pack 
 *   - across each row in the subpanel
 *   - then next row in each subpanel
 *   - then next subpanel (on subsequent call)
 *
 *     a b c |  j k l |  s t 0
 *     d e f |  m n o |  u v 0
 *     g h i |  p q r |  w x 0
 *
 *     ^^^^^
 *     each call packs one subpanel
 */
static inline void packB_kcxnc_d(
        int n,
        int k,
        double *XB,
        int ldXB,
        double *packB
        )
{
  int j, p;
  double *b_pntr[DGEMM_NR];
  for (j = 0; j < n; j++)
    b_pntr[j] = XB + j;

  for (p = 0; p < k; p++)
  {
    for (j = 0; j < n; j++)
    {
      *packB = *b_pntr[j];
      packB++;
      b_pntr[j] += ldXB;
    }
  }
}


/*
 * --------------------------------------------------------------------------
 */

static inline void bl_macro_kernel(
        int m,
        int n,
        int k,
        const double *packA,
        const double *packB,
        double *C,
        int ldc
        )
{
    int i, j;
    for (i = 0; i < m; i += DGEMM_MR)
    {
      for (j = 0; j < n; j += DGEMM_NR)
      {
        (*bl_micro_kernel) (
            k,
            min(m - i, DGEMM_MR),
            min(n - j, DGEMM_NR),
            &packA[i * k],
            &packB[j * k],
            &C[i * ldc + j],
            (unsigned long long) ldc
            );
      }
    }
}

void bl_dgemm (
        int m,
        int n,
        int k,
        double *XA,
        int lda,
        double *XB,
        int ldb,
        double *C,       
        int ldc       
        )
{
  int i, j, ic, ib, jc, jb, pc, pb;
  double *packA, *packB;
  
  packA  = bl_malloc_aligned(DGEMM_MC, DGEMM_KC, sizeof(double));
  packB  = bl_malloc_aligned(DGEMM_KC, DGEMM_NC, sizeof(double));

  for (ic = 0; ic < m; ic += DGEMM_MC)
  {
    ib = min(m - ic, DGEMM_MC);
    for (pc = 0; pc < k; pc += DGEMM_KC)
    {
      pb = min(k - pc, DGEMM_KC);
      for (i = 0; i < ib; i += DGEMM_MR)
      {
        packA_mcxkc_d(
          min(ib - i, DGEMM_MR),
          pb,
          &XA[lda * (ic + i) + pc],
          k,
          &packA[i * pb]
        );
      }
          
      for (jc = 0; jc < n; jc += DGEMM_NC)
      {
        jb = min(n - jc, DGEMM_NC);
        for (j = 0; j < jb; j += DGEMM_NR)
        {
          packB_kcxnc_d(
            min(jb - j, DGEMM_NR),
            pb,
            &XB[ldb * pc + (jc + j)],
            n,
            &packB[j * pb]
          );
        }

        bl_macro_kernel (
          ib,
          jb,
          pb,
          packA,
          packB,
          &C[ic * ldc + jc],
          ldc
        );
      }
    }
  }
  
  free(packA);
  free(packB);
}

void square_dgemm(int lda, double *A, double *B, double *C)
{
  bl_dgemm(lda, lda, lda, A, lda, B, lda, C,  lda);
}