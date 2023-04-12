namespace DotnetPrompt.HnswLib.Space.L2;

public class L2SqrSimd4ExtResiduals : IVectorSpace<float[]>
{
    private readonly L2SquareDistance _spaceResiduals;
    private readonly L2SqrSimd4Ext _space;

    public L2SqrSimd4ExtResiduals()
    {
        _space = new L2SqrSimd4Ext();
        _spaceResiduals = new L2SquareDistance();
    }

    private float L2SqrSIMD4ExtResidualsImpl(float[] pVect1, float[] pVect2, int qty)
    {
        // Calculate qty divided by 4, and multiplied by 4
        int qty4 = qty >> 2 << 2;

        // Calculate the L2 square distance for the first qty4 elements using SIMD4Ext function
        float res = _space.Distance(pVect1, pVect2, qty4);

        // Calculate the remaining elements (residuals) in the arrays
        int qty_left = qty - qty4;
        float[] pVect1Residual = new float[qty_left];
        float[] pVect2Residual = new float[qty_left];
        Array.Copy(pVect1, qty4, pVect1Residual, 0, qty_left);
        Array.Copy(pVect2, qty4, pVect2Residual, 0, qty_left);

        // Calculate the L2 square distance for the remaining elements
        float res_tail = _spaceResiduals.Distance(pVect1Residual, pVect2Residual, qty_left);

        // Return the sum of L2 square distances for both parts
        return res + res_tail;
    }

    public float Distance(float[] vector1, float[] vector2)
    {
        if(vector1.Length != vector2.Length)
            throw new ArgumentException("Vectors must be of the same length", nameof(vector2));

        return L2SqrSIMD4ExtResidualsImpl(vector1, vector2, vector1.Length);
    }
}