using DotnetPrompt.HnswLib;
using DotnetPrompt.HnswLib.Space.L2;
using NUnit.Framework;

namespace DotnetPrompt.HnswLb.Test;

[TestFixture]
public class L2SquareDistanceStrategyTests
{
    float[] vect1b = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
    float[] vect2b = new float[] { 1.0f, 0.5f, 1.5f, 0.25f, 1.25f, 0.75f, 1.75f, 0.125f, 1.125f, 0.625f, 1.625f, 0.375f, 1.375f, 0.875f, 1.875f, 0.0625f };

    [Test]
    public void TestL2SquareDistance()
    {
        var space = new L2SquareDistance();
        float result = space.Distance(vect1b, vect2b);
        float expected = 1256.37890625f;

        Assert.AreEqual(expected, result, 1e-6);
    }

    [Test]
    public void TestL2SqrSIMD16ExtAVX()
    {
        var space = new L2SqrSimd16ExtAvx();
        float result = space.Distance(vect1b, vect2b);
        float expected = 1256.37890625f;

        Assert.AreEqual(expected, result, 1e-6);
    }

    [Test]
    public void TestL2SqrSIMD16ExtSSE()
    {
        var space = new L2SqrSimd16ExtSse();
        float result = space.Distance(vect1b, vect2b);
        float expected = 1256.37890625f;

        Assert.AreEqual(expected, result, 1e-6);
    }

    [Test]
    public void TestL2SqrSIMD16ExtResiduals()
    {
        var space1 = new L2SqrSimd16ExtResiduals(new L2SqrSimd16ExtSse());
        var space2 = new L2SqrSimd16ExtResiduals(new L2SqrSimd16ExtAvx());

        float result = space1.Distance(vect1b, vect2b);
        float expected = space2.Distance(vect1b, vect2b);

        Assert.AreEqual(expected, result, 1e-6);
    }

    [Test]
    public void TestL2SqrSIMD4Ext()
    {
        var space = new L2SqrSimd4Ext();
        float result = space.Distance(vect1b, vect2b);
        float expected = 1256.37890625f;

        Assert.AreEqual(expected, result, 1e-6);
    }

    [Test]
    public void TestL2SqrSIMD4ExtResiduals()
    {
        var space = new L2SqrSimd4ExtResiduals();
        float result = space.Distance(vect1b, vect2b);
        float expected = 1256.37890625f;

        Assert.AreEqual(expected, result, 1e-6);
    }
}