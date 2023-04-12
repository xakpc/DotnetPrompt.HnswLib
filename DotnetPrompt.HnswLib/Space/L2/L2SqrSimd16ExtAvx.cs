using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotnetPrompt.HnswLib.Space.L2;

public class L2SqrSimd16ExtAvx : IVectorSpace<float[]>, IVectorPartSpace<float[]>
{
    private static unsafe float L2SqrSIMD16ExtAVXImpl(ReadOnlySpan<float> pVect1, ReadOnlySpan<float> pVect2, int qty)
    {
        int qty16 = qty >> 4;
        var TmpRes = new float[8];

        // Pointer to the end of the first vector
        int pEnd1 = qty16 << 4;

        Vector256<float> diff, v1, v2;
        Vector256<float> sum = Vector256<float>.Zero;

        fixed (float* pVect1Ptr = pVect1, pVect2Ptr = pVect2, TmpResPtr = TmpRes)
        {
            float* p1 = pVect1Ptr;
            float* p2 = pVect2Ptr;

            while (p1 < pVect1Ptr + pEnd1)
            {
                v1 = Avx.LoadVector256(p1);
                p1 += 8;
                v2 = Avx.LoadVector256(p2);
                p2 += 8;

                diff = Avx.Subtract(v1, v2);
                sum = Avx.Add(sum, Avx.Multiply(diff, diff));

                v1 = Avx.LoadVector256(p1);
                p1 += 8;
                v2 = Avx.LoadVector256(p2);
                p2 += 8;

                diff = Avx.Subtract(v1, v2);
                sum = Avx.Add(sum, Avx.Multiply(diff, diff));
            }

            Avx.Store(TmpResPtr, sum);
        }

        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }

    /// <inheritdoc />
    public float Distance(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length)
            throw new ArgumentException("Vectors must be of the same length", nameof(vector2));

        return L2SqrSIMD16ExtAVXImpl(vector1, vector2, vector1.Length);
    }

    public float Distance(float[] vector1, float[] vector2, int qty)
    {
        return L2SqrSIMD16ExtAVXImpl(vector1, vector2, qty);
    }
}