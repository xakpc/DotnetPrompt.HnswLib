using DotnetPrompt.HnswLib.Normalize;
using NUnit.Framework;

namespace DotnetPrompt.HnswLib.Test;

[TestFixture]
public class NormalizeVectorTests
{
    [Test]
    public void TestNormalizeVectorSize16()
    {
        float[] vector = Enumerable.Range(1, 16).Select(x => (float)x).ToArray();
        float[] expected = NormalizeFunctionSelector.Normalize(vector);
        float[] actual = NormalizeFunctionSelector.NormalizeAvx(vector);

        Assert.That(actual, Is.EqualTo(expected).Within(1e-6));
    }

    [Test]
    public void TestNormalizeVectorSize16Sse()
    {
        float[] vector = Enumerable.Range(1, 16).Select(x => (float)x).ToArray();
        float[] expected = NormalizeFunctionSelector.Normalize(vector);
        float[] actual = NormalizeFunctionSelector.NormalizeSse(vector);

        Assert.That(actual, Is.EqualTo(expected).Within(1e-6));
    }

    [Test]
    public void TestNormalizeVectorSize4()
    {
        float[] vector = Enumerable.Range(1, 4).Select(x => (float)x).ToArray();
        float[] expected = NormalizeFunctionSelector.Normalize(vector);
        float[] actual = NormalizeFunctionSelector.NormalizeSse(vector);

        Assert.That(actual, Is.EqualTo(expected).Within(1e-6));
    }

    [Test]
    public void TestNormalizeVectorSize4Avx()
    {
        float[] vector = Enumerable.Range(1, 4).Select(x => (float)x).ToArray();
        float[] expected = NormalizeFunctionSelector.Normalize(vector);
        float[] actual = NormalizeFunctionSelector.NormalizeAvx(vector);

        Assert.That(actual, Is.EqualTo(expected).Within(1e-6));
    }

    [Test]
    public void TestNormalizeVectorSize21()
    {
        float[] vector = Enumerable.Range(1, 21).Select(x => (float)x).ToArray();
        float[] expected = NormalizeFunctionSelector.Normalize(vector);
        float[] actual = NormalizeFunctionSelector.NormalizeAvx(vector);

        Assert.That(actual, Is.EqualTo(expected).Within(1e-6));
    }

    [Test]
    public void TestNormalizeVectorSize6()
    {
        float[] vector = Enumerable.Range(1, 6).Select(x => (float)x).ToArray();
        float[] expected = NormalizeFunctionSelector.Normalize(vector);
        float[] actual = NormalizeFunctionSelector.NormalizeSse(vector);

        Assert.That(actual, Is.EqualTo(expected).Within(1e-6));
    }
}