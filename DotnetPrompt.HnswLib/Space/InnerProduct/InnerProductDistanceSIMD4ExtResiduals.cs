namespace DotnetPrompt.HnswLib.Space.InnerProduct;

public class InnerProductDistanceSIMD4ExtResiduals : IVectorSpace<float[]>
{
    private readonly IDotProduct _space;
    private readonly IDotProduct _spaceFallback;

    /// <summary>
    /// 
    /// </summary>
    /// <param name="space">Should be any SIMD4 strategy</param>
    internal InnerProductDistanceSIMD4ExtResiduals(IDotProduct space)
    {
        _space = space;
        _spaceFallback = new InnerProductDistance();
    }

    /// <summary>
    /// Calculate inner product using SIMD4 with residuals
    /// </summary>
    public float InnerProductDistanceSIMD4ExtResidualsImpl(float[] pVect1, float[] pVect2, int qty)
    {
        int qty4 = qty >> 2 << 2;
        float res = _space.DotProduct(pVect1, pVect2, qty4);

        int qtyLeft = qty - qty4;
        float resTail = _spaceFallback.DotProduct(pVect1[qty4..], pVect2[qty4..], qtyLeft);

        return 1 - (res + resTail);
    }

    public float Distance(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length)
            throw new ArgumentException("Vectors must be of the same length");

        return InnerProductDistanceSIMD4ExtResidualsImpl(vector1, vector2, vector1.Length);
    }
}