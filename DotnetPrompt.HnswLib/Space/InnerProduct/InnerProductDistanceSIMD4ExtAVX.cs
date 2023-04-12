using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotnetPrompt.HnswLib.Space.InnerProduct;

/// <summary>
/// Compute inner product using AVX (Advanced Vector Extensions) SIMD (Single Instruction, Multiple Data)
/// </summary>
public class InnerProductDistanceSIMD4ExtAVX : IVectorSpace<float[]>, IDotProduct
{
    public static unsafe float InnerProductSIMD4ExtAVX(ReadOnlySpan<float> pVect1, ReadOnlySpan<float> pVect2, int qty)
    {
        float[] TmpRes = new float[8];

        // Calculate number of iterations for processing 16 and 4 elements at a time
        int qty16 = qty / 16;
        int qty4 = qty / 4;

        // Set pointers to the end of data processed in each loop
        int pEnd1 = 16 * qty16;
        int pEnd2 = 4 * qty4;

        // Initialize sum256 to store the sum of products
        Vector256<float> sum256 = Vector256<float>.Zero;
        Vector128<float> sum_prod = Vector128<float>.Zero;

        fixed (float* pVect1Ptr = pVect1, pVect2Ptr = pVect2)
        {
            // Process 16 elements at a time
            for (int i = 0; i < pEnd1; i += 16)
            {
                // Load 8 elements from each vector
                Vector256<float> v1 = Avx.LoadVector256(pVect1Ptr + i);
                Vector256<float> v2 = Avx.LoadVector256(pVect2Ptr + i);

                // Multiply the vectors and accumulate the result in sum256
                sum256 = Avx.Add(sum256, Avx.Multiply(v1, v2));

                // Repeat the process for the next 8 elements
                v1 = Avx.LoadVector256(pVect1Ptr + i + 8);
                v2 = Avx.LoadVector256(pVect2Ptr + i + 8);
                sum256 = Avx.Add(sum256, Avx.Multiply(v1, v2));
            }

            // Combine the two 128-bit halves of sum256
            sum_prod = Sse.Add(Avx.ExtractVector128(sum256, 0), Avx.ExtractVector128(sum256, 1));

            // Process remaining 4 elements at a time
            for (int i = pEnd1; i < pEnd2; i += 4)
            {
                Vector128<float> v1 = Sse.LoadVector128(pVect1Ptr + i);
                Vector128<float> v2 = Sse.LoadVector128(pVect2Ptr + i);
                sum_prod = Sse.Add(sum_prod, Sse.Multiply(v1, v2));
            }
        }

        // Store the result and sum the elements of sum_prod
        fixed (float* tmpResPtr = TmpRes)
        {
            Sse.Store(tmpResPtr, sum_prod);
        }

        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
        return sum;
    }

    // Compute inner product distance using AVX SIMD
    public static float InnerProductDistanceSIMD4ExtAVXImpl(ReadOnlySpan<float> pVect1, ReadOnlySpan<float> pVect2, int qty)
    {
        return 1.0f - InnerProductSIMD4ExtAVX(pVect1, pVect2, qty);
    }

    public float Distance(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length)
            throw new ArgumentException("Vectors must be of the same length");

        return InnerProductDistanceSIMD4ExtAVXImpl(vector1, vector2, vector1.Length);
    }

    public float DotProduct(float[] vector1, float[] vector2, int qty)
    {
        return InnerProductSIMD4ExtAVX(vector1, vector2, qty);
    }
}