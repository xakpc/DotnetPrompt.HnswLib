using System.Runtime.Intrinsics.X86;
using DotnetPrompt.HnswLib.Space.InnerProduct;
using DotnetPrompt.HnswLib.Space.L2;

namespace DotnetPrompt.HnswLib.Space;

public class InnerProductStrategySelector
{
    public IVectorSpace<float[]> SelectSpaceStrategy(Spaces space, int dimensions)
    {
        switch (space)
        {
            case Spaces.L2:
                return SelectL2Space(dimensions);
            case Spaces.InnerProduct:
                return SelectInnerProductSpace(dimensions);
            case Spaces.Cosine:
                throw new NotImplementedException("Cosine still not implemented");
            default:
                throw new ArgumentException("Unknown space");
        }
    }

    private IVectorSpace<float[]> SelectInnerProductSpace(int dim)
    {
        if (Avx.IsSupported)
        {
            if (dim % 16 == 0)
                return new InnerProductDistanceSIMD16ExtAVX();
            if (dim % 4 == 0)
                return new InnerProductDistanceSIMD4ExtAVX();

            if (dim > 16)
                return new InnerProductDistanceSIMD16ExtResiduals(new InnerProductDistanceSIMD16ExtAVX());
            if (dim > 4)
                return new InnerProductDistanceSIMD4ExtResiduals(new InnerProductDistanceSIMD4ExtAVX());
        }

        if (Sse.IsSupported)
        {
            if (dim % 16 == 0)
                return new InnerProductDistanceSIMD16ExtSSE();
            if (dim % 4 == 0)
                return new InnerProductDistanceSIMD4ExtSSE();

            if (dim > 16)
                return new InnerProductDistanceSIMD16ExtResiduals(new InnerProductDistanceSIMD16ExtSSE());
            if (dim > 4)
                return new InnerProductDistanceSIMD4ExtResiduals(new InnerProductDistanceSIMD4ExtSSE());
        }

        return new InnerProductDistance();
    }

    private IVectorSpace<float[]> SelectL2Space(int dim)
    {
        if (Avx.IsSupported)
        {
            if (dim % 16 == 0)
                return new L2SqrSimd16ExtAvx();
            if (dim % 4 == 0)
                return Sse.IsSupported ? new L2SqrSimd4Ext() : new L2SquareDistance();
            if (dim > 16)
                return new L2SqrSimd16ExtResiduals(new L2SqrSimd16ExtAvx());
            if (dim > 4)
                return new L2SqrSimd4ExtResiduals();
        }

        if (Sse.IsSupported)
        {
            if (dim % 16 == 0)
                return new L2SqrSimd16ExtSse();
            if (dim % 4 == 0)
                return new L2SqrSimd4Ext();
            if (dim > 16)
                return new L2SqrSimd16ExtResiduals(new L2SqrSimd16ExtSse());
            if (dim > 4)
                return new L2SqrSimd4ExtResiduals();
            return new L2SqrSimd4Ext();
        }

        return new L2SquareDistance();
    }
}