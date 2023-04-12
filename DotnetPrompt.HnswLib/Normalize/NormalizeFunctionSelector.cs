using System.Runtime.Intrinsics.X86;
using System.Runtime.Intrinsics;

namespace DotnetPrompt.HnswLib.Normalize;

public delegate float[] NormalizeFunction(ReadOnlySpan<float> vector);

public static class NormalizeFunctionSelector
{
    public static NormalizeFunction SelectNormalizeFunction()
    {
        if (Avx.IsSupported)
        {
            return NormalizeAvx;
        }

        if (Sse.IsSupported)
        {
            return NormalizeSse;
        }

        return Normalize;
    }
    
    internal static unsafe float[] NormalizeAvx(ReadOnlySpan<float> vector)
    {
        int length = vector.Length;
        int simdLength = length / 8 * 8;

        Vector256<float> lengthSquaredVec = Vector256<float>.Zero;

        fixed (float* vectorPtr = vector)
        {
            for (int i = 0; i < simdLength; i += 8)
            {
                Vector256<float> vec = Avx.LoadVector256(vectorPtr + i);
                lengthSquaredVec = Avx.Add(lengthSquaredVec, Avx.Multiply(vec, vec));
            }
        }

        Vector128<float> partialSum = Avx.Add(lengthSquaredVec.GetLower(), lengthSquaredVec.GetUpper());
        partialSum = Sse.Add(partialSum, Sse.Shuffle(partialSum, partialSum, 0b10110001));
        partialSum = Sse.Add(partialSum, Sse.Shuffle(partialSum, partialSum, 0b01001110));
        float lengthSquared = partialSum.ToScalar();

        for (int i = simdLength; i < length; i++)
        {
            lengthSquared += vector[i] * vector[i];
        }

        float lengthValue = (float)Math.Sqrt(lengthSquared);
        float[] normalizedVector = new float[length];

        Vector256<float> lengthVec = Vector256.Create(lengthValue);

        fixed (float* srcPtr = vector, dstPtr = normalizedVector)
        {
            for (int i = 0; i < simdLength; i += 8)
            {
                Vector256<float> vec = Avx.LoadVector256(srcPtr + i);
                Avx.Store(dstPtr + i, Avx.Divide(vec, lengthVec));
            }
        }

        for (int i = simdLength; i < length; i++)
        {
            normalizedVector[i] = vector[i] / lengthValue;
        }

        return normalizedVector;
    }

    internal static unsafe float[] NormalizeSse(ReadOnlySpan<float> vector)
    {
        int length = vector.Length;
        int simdLength = length / 4 * 4;
        Vector128<float> lengthSquaredVec = Vector128<float>.Zero;

        fixed (float* vectorPtr = vector)
        {
            for (int i = 0; i < simdLength; i += 4)
            {
                Vector128<float> vec = Sse.LoadVector128(vectorPtr + i);
                lengthSquaredVec = Sse.Add(lengthSquaredVec, Sse.Multiply(vec, vec));
            }
        }

        lengthSquaredVec = Sse.Add(lengthSquaredVec, Sse.MoveHighToLow(lengthSquaredVec, lengthSquaredVec));
        lengthSquaredVec = Sse.Add(lengthSquaredVec, Sse.Shuffle(lengthSquaredVec, lengthSquaredVec, 1));
        float lengthSquared = lengthSquaredVec.ToScalar();

        for (int i = simdLength; i < length; i++)
        {
            lengthSquared += vector[i] * vector[i];
        }

        float lengthValue = (float)Math.Sqrt(lengthSquared);
        float[] normalizedVector = new float[length];

        Vector128<float> lengthVec = Vector128.Create(lengthValue);

        fixed (float* srcPtr = vector, dstPtr = normalizedVector)
        {
            for (int i = 0; i < simdLength; i += 4)
            {
                Vector128<float> vec = Sse.LoadVector128(srcPtr + i);
                Sse.Store(dstPtr + i, Sse.Divide(vec, lengthVec));
            }
        }

        for (int i = simdLength; i < length; i++)
        {
            normalizedVector[i] = vector[i] / lengthValue;
        }

        return normalizedVector;
    }

    internal static float[] Normalize(ReadOnlySpan<float> vector)
    {
        var length = 0f;
        for (int i = 0; i < vector.Length; i++)
        {
            length += vector[i] * vector[i];
        }
        length = (float)Math.Sqrt(length);

        var normalizedVector = new float[vector.Length];
        for (int i = 0; i < vector.Length; i++)
        {
            normalizedVector[i] = vector[i] / length;
        }

        return normalizedVector;
    }
}