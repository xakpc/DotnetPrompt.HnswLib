namespace DotnetPrompt.HnswLib.Space.InnerProduct;

public class InnerProductDistanceSIMD16ExtResiduals : IVectorSpace<float[]>
{
    private readonly IDotProduct _space;
    private readonly IDotProduct _spaceFallback;

    internal InnerProductDistanceSIMD16ExtResiduals(IDotProduct space)
    {
        _space = space;
        _spaceFallback = new InnerProductDistance();
    }

    /// <summary>
    /// Calculate inner product using SIMD16 AVX with residuals
    /// </summary>
    public float InnerProductDistanceSIMD16ExtResidualsImpl(float[] pVect1, float[] pVect2, int qty)
    {
        int qty16 = qty >> 4 << 4;
        float res = _space.DotProduct(pVect1, pVect2, qty16);

        int qtyLeft = qty - qty16;
        float resTail = _spaceFallback.DotProduct(pVect1[qty16..], pVect2[qty16..], qtyLeft);

        return 1 - (res + resTail);
    }

    public float Distance(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length) 
            throw new ArgumentException("Vectors must be of the same length");

        return InnerProductDistanceSIMD16ExtResidualsImpl(vector1, vector2, vector1.Length);
    }
}