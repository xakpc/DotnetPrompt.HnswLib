using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotnetPrompt.HnswLib.Space.InnerProduct;

public class InnerProductDistanceSIMD16ExtSSE : IVectorSpace<float[]>, IDotProduct
{
    /// <summary>
    /// Calculate inner product using SIMD16 SSE
    /// </summary>
    private static unsafe float InnerProductSIMD16ExtSSE(ReadOnlySpan<float> pVect1, ReadOnlySpan<float> pVect2, int qty)
    {
        // Temporary result array to store the sum of products
        float[] TmpRes = new float[8];

        // Calculate the number of iterations to process 16 elements at a time
        int qty16 = qty / 16;
        int pEnd1 = 16 * qty16;

        // Initialize the sum_prod vector to zero
        Vector128<float> sum_prod = Vector128<float>.Zero;

        // Pin the input arrays in memory and obtain pointers to their first elements
        fixed (float* pVect1Ptr = pVect1, pVect2Ptr = pVect2)
        {
            // Process 16 elements at a time
            for (int i = 0; i < pEnd1; i += 16)
            {
                // Load 4 elements from each input array and multiply, accumulating the result in sum_prod
                Vector128<float> v1 = Sse.LoadVector128(pVect1Ptr + i);
                Vector128<float> v2 = Sse.LoadVector128(pVect2Ptr + i);
                sum_prod = Sse.Add(sum_prod, Sse.Multiply(v1, v2));

                // Repeat the process for the next 4 elements
                v1 = Sse.LoadVector128(pVect1Ptr + i + 4);
                v2 = Sse.LoadVector128(pVect2Ptr + i + 4);
                sum_prod = Sse.Add(sum_prod, Sse.Multiply(v1, v2));

                // Repeat the process for the next 4 elements
                v1 = Sse.LoadVector128(pVect1Ptr + i + 8);
                v2 = Sse.LoadVector128(pVect2Ptr + i + 8);
                sum_prod = Sse.Add(sum_prod, Sse.Multiply(v1, v2));

                // Repeat the process for the next 4 elements
                v1 = Sse.LoadVector128(pVect1Ptr + i + 12);
                v2 = Sse.LoadVector128(pVect2Ptr + i + 12);
                sum_prod = Sse.Add(sum_prod, Sse.Multiply(v1, v2));
            }
        }

        // Store the accumulated sum_prod in the TmpRes array
        fixed (float* tmpResPtr = TmpRes)
        {
            Sse.Store(tmpResPtr, sum_prod);
        }

        // Calculate the final sum of products by adding the elements of TmpRes
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

        return sum;
    }

    /// <summary>
    /// Calculate inner product distance using SIMD16 SSE
    /// </summary>
    private static float InnerProductDistanceSIMD16ExtSSEImpl(ReadOnlySpan<float> pVect1, ReadOnlySpan<float> pVect2, int qty)
    {
        // Compute the inner product distance by subtracting the inner product from 1.0f
        return 1.0f - InnerProductSIMD16ExtSSE(pVect1, pVect2, qty);
    }

    public float Distance(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length)
            throw new ArgumentException("Vectors must be of the same length");

        return InnerProductDistanceSIMD16ExtSSEImpl(vector1, vector2, vector1.Length);
    }

    public float DotProduct(float[] vector1, float[] vector2, int qty)
    {
        return InnerProductSIMD16ExtSSE(vector1, vector2, qty);
    }
}