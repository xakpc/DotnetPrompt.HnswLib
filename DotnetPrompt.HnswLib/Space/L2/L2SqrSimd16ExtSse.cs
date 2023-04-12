using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotnetPrompt.HnswLib.Space.L2;

public class L2SqrSimd16ExtSse : IVectorSpace<float[]>, IVectorPartSpace<float[]>
{
    private static unsafe float L2SqrSIMD16ExtSSEImpl(ReadOnlySpan<float> pVect1, ReadOnlySpan<float> pVect2, int qty)
    {
        float[] TmpRes = new float[8];
        int qty16 = qty >> 4;

        // Pointer to the end of the first vector
        int pEnd1 = qty16 << 4;

        Vector128<float> diff, v1, v2;
        Vector128<float> sum = Vector128<float>.Zero;

        fixed (float* pVect1Ptr = pVect1, pVect2Ptr = pVect2, TmpResPtr = TmpRes)
        {
            float* p1 = pVect1Ptr;
            float* p2 = pVect2Ptr;

            while (p1 < pVect1Ptr + pEnd1)
            {
                // Load the first 4 elements from the vectors
                v1 = Sse.LoadVector128(p1);
                p1 += 4;
                v2 = Sse.LoadVector128(p2);
                p2 += 4;
                diff = Sse.Subtract(v1, v2);
                sum = Sse.Add(sum, Sse.Multiply(diff, diff));

                // Repeat the process for the next sets of 4 elements in the vectors
                v1 = Sse.LoadVector128(p1);
                p1 += 4;
                v2 = Sse.LoadVector128(p2);
                p2 += 4;
                diff = Sse.Subtract(v1, v2);
                sum = Sse.Add(sum, Sse.Multiply(diff, diff));

                v1 = Sse.LoadVector128(p1);
                p1 += 4;
                v2 = Sse.LoadVector128(p2);
                p2 += 4;
                diff = Sse.Subtract(v1, v2);
                sum = Sse.Add(sum, Sse.Multiply(diff, diff));

                v1 = Sse.LoadVector128(p1);
                p1 += 4;
                v2 = Sse.LoadVector128(p2);
                p2 += 4;
                diff = Sse.Subtract(v1, v2);
                sum = Sse.Add(sum, Sse.Multiply(diff, diff));
            }

            Sse.Store(TmpResPtr, sum);
        }

        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }

    public float Distance(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length)
            throw new ArgumentException("Vectors must be of the same length", nameof(vector2));

        return L2SqrSIMD16ExtSSEImpl(vector1, vector2, vector1.Length);
    }

    public float Distance(float[] vector1, float[] vector2, int qty)
    {
        return L2SqrSIMD16ExtSSEImpl(vector1, vector2, qty);
    }
}