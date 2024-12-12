#include "bl_config.h"
#include "bl_dgemm_kernel.h"

//
// C-based micorkernel
//
void bl_dgemm_ukr ( 
        int k,
        int m,
        int n,
        const double *a,
        const double *b,
        double *c,
        unsigned long long ldc
        )
{
    register int i, j, p;
    for (p = 0; p < k; p++)
    {
        for (i = 0; i < m; i++)
        {
            for (j = 0; j < n; j++)
                c[i * ldc + j] += a[p * m + i] * b[p * n + j];
        }
    }
}

void bl_dgemm_2x4 (
        int k,
        int m,
        int n,
        const double *a,
        const double *b,
        double *c,
        unsigned long long ldc
        )
{   
    register int p;
    register svfloat64_t a0x, a1x;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x;

    svbool_t npred = svwhilelt_b64(0, n);
    if (m > 0)
        c0x = svld1_f64(npred, c);
    if (m > 1)
        c1x = svld1_f64(npred, c + ldc);

    for (p = 0; p < k; p++)
    {
        bx = svld1_f64(npred, b);
        if (m > 0)
        {
            a0x = svdup_f64(*a);
            a++;
            c0x = svmla_f64_m(npred, c0x, bx, a0x);
        }    
        if (m > 1)
        {
            a1x = svdup_f64(*a);
            a++;
            c1x = svmla_f64_m(npred, c1x, bx, a1x);
        }
        b += n;
    }

    if (m > 0)
        svst1_f64(npred, c, c0x);
    if (m > 1)
        svst1_f64(npred, c + ldc, c1x);
}

void bl_dgemm_4x4 (
        int k,
        int m,
        int n,
        const double *a,
        const double *b,
        double *c,
        unsigned long long ldc
        )
{   
    register int p;
    register svfloat64_t a0x, a1x, a2x, a3x;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x;

    svbool_t npred = svwhilelt_b64(0, n);
    if (m > 0)
        c0x = svld1_f64(npred, c);
    if (m > 1)
        c1x = svld1_f64(npred, c + ldc);
    if (m > 2)
        c2x = svld1_f64(npred, c + ldc * 2);
    if (m > 3)
        c3x = svld1_f64(npred, c + ldc * 3);

    for (p = 0; p < k; p++)
    {
        bx = svld1_f64(npred, b);
        if (m > 0)
        {
            a0x = svdup_f64(*a);
            a++;
            c0x = svmla_f64_m(npred, c0x, bx, a0x);
        }    
        if (m > 1)
        {
            a1x = svdup_f64(*a);
            a++;
            c1x = svmla_f64_m(npred, c1x, bx, a1x);
        }
        if (m > 2)
        {
            a2x = svdup_f64(*a);
            a++;
            c2x = svmla_f64_m(npred, c2x, bx, a2x);
        }
        if (m > 3)
        {
            a3x = svdup_f64(*a);
            a++;
            c3x = svmla_f64_m(npred, c3x, bx, a3x);
        }
        b += n;
    }

    if (m > 0)
        svst1_f64(npred, c, c0x);
    if (m > 1)
        svst1_f64(npred, c + ldc, c1x);
    if (m > 2)
        svst1_f64(npred, c + ldc * 2, c2x);
    if (m > 3)
        svst1_f64(npred, c + ldc * 3, c3x);
}

void bl_dgemm_8x4 (
        int k,
        int m,
        int n,
        const double *a,
        const double *b,
        double *c,
        unsigned long long ldc
        )
{   
    register int p;
    register svfloat64_t a0x, a1x, a2x, a3x, a4x, a5x, a6x, a7x;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x;

    svbool_t npred = svwhilelt_b64(0, n);
    if (m > 0)
        c0x = svld1_f64(npred, c);
    if (m > 1)
        c1x = svld1_f64(npred, c + ldc);
    if (m > 2)
        c2x = svld1_f64(npred, c + ldc * 2);
    if (m > 3)
        c3x = svld1_f64(npred, c + ldc * 3);
    if (m > 4)
        c4x = svld1_f64(npred, c + ldc * 4);   
    if (m > 5)
        c5x = svld1_f64(npred, c + ldc * 5);
    if (m > 6)
        c6x = svld1_f64(npred, c + ldc * 6);
    if (m > 7)
        c7x = svld1_f64(npred, c + ldc * 7);

    for (p = 0; p < k; p++)
    {
        bx = svld1_f64(npred, b);
        if (m > 0)
        {
            a0x = svdup_f64(*a);
            a++;
            c0x = svmla_f64_m(npred, c0x, bx, a0x);
        }
        if (m > 1)
        {
            a1x = svdup_f64(*a);
            a++;
            c1x = svmla_f64_m(npred, c1x, bx, a1x);
        }
        if (m > 2)
        {
            a2x = svdup_f64(*a);
            a++;
            c2x = svmla_f64_m(npred, c2x, bx, a2x);
        }
        if (m > 3)
        {
            a3x = svdup_f64(*a);
            a++;
            c3x = svmla_f64_m(npred, c3x, bx, a3x);
        }
        if (m > 4)
        {
            a4x = svdup_f64(*a);
            a++;
            c4x = svmla_f64_m(npred, c4x, bx, a4x);
        }
        if (m > 5)
        {
            a5x = svdup_f64(*a);
            a++;
            c5x = svmla_f64_m(npred, c5x, bx, a5x);
        }
        if (m > 6)
        {
            a6x = svdup_f64(*a);
            a++;
            c6x = svmla_f64_m(npred, c6x, bx, a6x);
        }
        if (m > 7)
        {
            a7x = svdup_f64(*a);
            a++;
            c7x = svmla_f64_m(npred, c7x, bx, a7x);
        }
        b += n;
    }

    if (m > 0)
        svst1_f64(npred, c, c0x);
    if (m > 1)
        svst1_f64(npred, c + ldc, c1x);
    if (m > 2)
        svst1_f64(npred, c + ldc * 2, c2x);
    if (m > 3)
        svst1_f64(npred, c + ldc * 3, c3x);
    if (m > 4)
        svst1_f64(npred, c + ldc * 4, c4x);
    if (m > 5)
        svst1_f64(npred, c + ldc * 5, c5x);
    if (m > 6)
        svst1_f64(npred, c + ldc * 6, c6x);
    if (m > 7)
        svst1_f64(npred, c + ldc * 7, c7x);
}

void bl_dgemm_16x4 (
        int k,
        int m,
        int n,
        const double *a,
        const double *b,
        double *c,
        unsigned long long ldc
        )
{   
    register int p;
    register svfloat64_t a0x, a1x, a2x, a3x, a4x, a5x, a6x, a7x, a8x, a9x, a10x, a11x, a12x, a13x, a14x, a15x;
    register svfloat64_t bx;
    register svfloat64_t c0x, c1x, c2x, c3x, c4x, c5x, c6x, c7x, c8x, c9x, c10x, c11x, c12x, c13x, c14x, c15x;

    svbool_t npred = svwhilelt_b64(0, n);
    if (m > 0)
        c0x = svld1_f64(npred, c);
    if (m > 1)
        c1x = svld1_f64(npred, c + ldc);
    if (m > 2)
        c2x = svld1_f64(npred, c + ldc * 2);
    if (m > 3)
        c3x = svld1_f64(npred, c + ldc * 3);
    if (m > 4)
        c4x = svld1_f64(npred, c + ldc * 4);   
    if (m > 5)
        c5x = svld1_f64(npred, c + ldc * 5);
    if (m > 6)
        c6x = svld1_f64(npred, c + ldc * 6);
    if (m > 7)
        c7x = svld1_f64(npred, c + ldc * 7);
    if (m > 8)
        c8x = svld1_f64(npred, c + ldc * 8);
    if (m > 9)
        c9x = svld1_f64(npred, c + ldc * 9);
    if (m > 10)
        c10x = svld1_f64(npred, c + ldc * 10);
    if (m > 11)
        c11x = svld1_f64(npred, c + ldc * 11);
    if (m > 12)
        c12x = svld1_f64(npred, c + ldc * 12);   
    if (m > 13)
        c13x = svld1_f64(npred, c + ldc * 13);
    if (m > 14)
        c14x = svld1_f64(npred, c + ldc * 14);
    if (m > 15)
        c15x = svld1_f64(npred, c + ldc * 15);

    for (p = 0; p < k; p++)
    {
        bx = svld1_f64(npred, b);
        if (m > 0)
        {
            a0x = svdup_f64(*a);
            a++;
            c0x = svmla_f64_m(npred, c0x, bx, a0x);
        }
        if (m > 1)
        {
            a1x = svdup_f64(*a);
            a++;
            c1x = svmla_f64_m(npred, c1x, bx, a1x);
        }
        if (m > 2)
        {
            a2x = svdup_f64(*a);
            a++;
            c2x = svmla_f64_m(npred, c2x, bx, a2x);
        }
        if (m > 3)
        {
            a3x = svdup_f64(*a);
            a++;
            c3x = svmla_f64_m(npred, c3x, bx, a3x);
        }
        if (m > 4)
        {
            a4x = svdup_f64(*a);
            a++;
            c4x = svmla_f64_m(npred, c4x, bx, a4x);
        }
        if (m > 5)
        {
            a5x = svdup_f64(*a);
            a++;
            c5x = svmla_f64_m(npred, c5x, bx, a5x);
        }
        if (m > 6)
        {
            a6x = svdup_f64(*a);
            a++;
            c6x = svmla_f64_m(npred, c6x, bx, a6x);
        }
        if (m > 7)
        {
            a7x = svdup_f64(*a);
            a++;
            c7x = svmla_f64_m(npred, c7x, bx, a7x);
        }
        if (m > 8)
        {
            a8x = svdup_f64(*a);
            a++;
            c8x = svmla_f64_m(npred, c8x, bx, a8x);
        }
        if (m > 9)
        {
            a9x = svdup_f64(*a);
            a++;
            c9x = svmla_f64_m(npred, c9x, bx, a9x);
        }
        if (m > 10)
        {
            a10x = svdup_f64(*a);
            a++;
            c10x = svmla_f64_m(npred, c10x, bx, a10x);
        }
        if (m > 11)
        {
            a11x = svdup_f64(*a);
            a++;
            c11x = svmla_f64_m(npred, c11x, bx, a11x);
        }
        if (m > 12)
        {
            a12x = svdup_f64(*a);
            a++;
            c12x = svmla_f64_m(npred, c12x, bx, a12x);
        }
        if (m > 13)
        {
            a13x = svdup_f64(*a);
            a++;
            c13x = svmla_f64_m(npred, c13x, bx, a13x);
        }
        if (m > 14)
        {
            a14x = svdup_f64(*a);
            a++;
            c14x = svmla_f64_m(npred, c14x, bx, a14x);
        }
        if (m > 15)
        {
            a15x = svdup_f64(*a);
            a++;
            c15x = svmla_f64_m(npred, c15x, bx, a15x);
        }
        b += n;
    }

    if (m > 0)
        svst1_f64(npred, c, c0x);
    if (m > 1)
        svst1_f64(npred, c + ldc, c1x);
    if (m > 2)
        svst1_f64(npred, c + ldc * 2, c2x);
    if (m > 3)
        svst1_f64(npred, c + ldc * 3, c3x);
    if (m > 4)
        svst1_f64(npred, c + ldc * 4, c4x);
    if (m > 5)
        svst1_f64(npred, c + ldc * 5, c5x);
    if (m > 6)
        svst1_f64(npred, c + ldc * 6, c6x);
    if (m > 7)
        svst1_f64(npred, c + ldc * 7, c7x);
    if (m > 8)
        svst1_f64(npred, c + ldc * 8, c8x);
    if (m > 9)
        svst1_f64(npred, c + ldc * 9, c9x);
    if (m > 10)
        svst1_f64(npred, c + ldc * 10, c10x);
    if (m > 11)
        svst1_f64(npred, c + ldc * 11, c11x);
    if (m > 12)
        svst1_f64(npred, c + ldc * 12, c12x);
    if (m > 13)
        svst1_f64(npred, c + ldc * 13, c13x);
    if (m > 14)
        svst1_f64(npred, c + ldc * 14, c14x);
    if (m > 15)
        svst1_f64(npred, c + ldc * 15, c15x);
}