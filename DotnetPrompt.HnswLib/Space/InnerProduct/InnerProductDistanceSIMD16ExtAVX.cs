using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotnetPrompt.HnswLib.Space.InnerProduct;

public class InnerProductDistanceSIMD16ExtAVX : IVectorSpace<float[]>, IDotProduct
{
    /// <summary>
    /// Calculate inner product using SIMD16 AVX
    /// </summary>
    private static unsafe float InnerProductSIMD16ExtAVX(ReadOnlySpan<float> pVect1, ReadOnlySpan<float> pVect2, int qty)
    {
        // Temporary result array to store the sum of products
        float[] TmpRes = new float[8];

        // Calculate the number of iterations to process 16 elements at a time
        int qty16 = qty / 16;
        int pEnd1 = 16 * qty16;

        // Initialize the sum256 vector to zero
        Vector256<float> sum256 = Vector256<float>.Zero;

        // Pin the input arrays in memory and obtain pointers to their first elements
        fixed (float* pVect1Ptr = pVect1, pVect2Ptr = pVect2)
        {
            // Process 16 elements at a time
            for (int i = 0; i < pEnd1; i += 16)
            {
                // Load 8 elements from each input array
                Vector256<float> v1 = Avx.LoadVector256(pVect1Ptr + i);
                Vector256<float> v2 = Avx.LoadVector256(pVect2Ptr + i);

                // Multiply the vectors and accumulate the result in sum256
                sum256 = Avx.Add(sum256, Avx.Multiply(v1, v2));

                // Repeat the process for the next 8 elements
                v1 = Avx.LoadVector256(pVect1Ptr + i + 8);
                v2 = Avx.LoadVector256(pVect2Ptr + i + 8);
                sum256 = Avx.Add(sum256, Avx.Multiply(v1, v2));
            }
        }

        // Store the accumulated sum256 in the TmpRes array
        fixed (float* tmpResPtr = TmpRes)
        {
            Avx.Store(tmpResPtr, sum256);
        }

        // Calculate the final sum of products by adding the elements of TmpRes
        float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

        return sum;
    }

    /// <summary>
    /// Calculate inner product distance using SIMD16 AVX
    /// </summary>
    private static float InnerProductDistanceSIMD16ExtAVXImpl(ReadOnlySpan<float> pVect1, ReadOnlySpan<float> pVect2, int qty)
    {
        return 1.0f - InnerProductSIMD16ExtAVX(pVect1, pVect2, qty);
    }

    public float Distance(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length)
            throw new ArgumentException("Vectors must be of the same length");

        return InnerProductDistanceSIMD16ExtAVXImpl(vector1, vector2, vector1.Length);
    }

    public float DotProduct(float[] vector1, float[] vector2, int qty)
    {
        return InnerProductSIMD16ExtAVX(vector1, vector2, qty);
    }
}