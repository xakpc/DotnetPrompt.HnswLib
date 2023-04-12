using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotnetPrompt.HnswLib.Space.InnerProduct;

public class InnerProductDistanceSIMD4ExtSSE : IVectorSpace<float[]>, IDotProduct
{
    // Calculate inner product using SIMD4 SSE
    private static unsafe float InnerProductSIMD4ExtSSE(ReadOnlySpan<float> pVect1, ReadOnlySpan<float> pVect2, int qty)
    {
        float[] TmpRes = new float[8];

        int qty16 = qty / 16;
        int qty4 = qty / 4;

        int pEnd1 = 16 * qty16;
        int pEnd2 = 4 * qty4;

        Vector128<float> sum_prod = Vector128<float>.Zero;

        fixed (float* pVect1Ptr = pVect1, pVect2Ptr = pVect2)
        {
            int index1 = 0;
            // Process 16 elements at a time
            while (index1 < pEnd1)
            {
                for (int i = 0; i < 4; i++)
                {
                    Vector128<float> v1 = Sse.LoadVector128(pVect1Ptr + index1);
                    Vector128<float> v2 = Sse.LoadVector128(pVect2Ptr + index1);
                    sum_prod = Sse.Add(sum_prod, Sse.Multiply(v1, v2));
                    index1 += 4;
                }
            }

            // Process remaining elements (less than 16)
            while (index1 < pEnd2)
            {
                Vector128<float> v1 = Sse.LoadVector128(pVect1Ptr + index1);
                Vector128<float> v2 = Sse.LoadVector128(pVect2Ptr + index1);
                sum_prod = Sse.Add(sum_prod, Sse.Multiply(v1, v2));
                index1 += 4;
            }

            fixed (float* TmpResPtr = TmpRes)
            {
                Sse.Store(TmpResPtr, sum_prod);
            }
        }

        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
        return sum;
    }
    
    // Calculate inner product distance using SIMD4 SSE
    private static float InnerProductDistanceSIMD4ExtSSEImpl(ReadOnlySpan<float> pVect1, ReadOnlySpan<float> pVect2, int qty)
    {
        return 1.0f - InnerProductSIMD4ExtSSE(pVect1, pVect2, qty);
    }

    public float Distance(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length)
            throw new ArgumentException("Vectors must be of the same length");

        return InnerProductDistanceSIMD4ExtSSEImpl(vector1, vector2, vector1.Length);
    }

    public float DotProduct(float[] vector1, float[] vector2, int qty)
    {
        return InnerProductSIMD4ExtSSE(vector1, vector2, qty);
    }
}