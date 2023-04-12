namespace DotnetPrompt.HnswLib.Space.InnerProduct;

/// <summary>
/// Internal interface to publish DotProduct method to Residuals
/// </summary>
internal interface IDotProduct
{
    float DotProduct(float[] vector1, float[] vector2, int qty);
}

public class InnerProductDistance : IVectorSpace<float[]>, IDotProduct
{
    private static float InnerProduct(ReadOnlySpan<float> vect1, ReadOnlySpan<float> vect2, int qty)
    {
        float res = 0;
        for (int i = 0; i < qty; i++)
        {
            res += vect1[i] * vect2[i];
        }
        return res;
    }

    private static float InnerProductDistanceImpl(ReadOnlySpan<float> vect1, ReadOnlySpan<float> vect2, int qty)
    {
        return 1.0f - InnerProduct(vect1, vect2, qty);
    }

    public float Distance(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length)
            throw new ArgumentException("Vectors must be of the same length");

        return InnerProductDistanceImpl(vector1, vector2, vector1.Length);
    }

    public float DotProduct(float[] vector1, float[] vector2, int qty)
    {
        return InnerProduct(vector1, vector2, qty);
    }
}