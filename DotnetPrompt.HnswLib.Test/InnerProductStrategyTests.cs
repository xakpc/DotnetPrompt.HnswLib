using System.Runtime.Intrinsics.X86;
using DotnetPrompt.HnswLib.Space.InnerProduct;

namespace DotnetPrompt.HnswLb.Test
{
    [TestFixture]
    public class InnerProductStrategyTests
    {
        [Test]
        public void TestWhatAvailible()
        {
            Assert.True(Avx.IsSupported);
            Assert.True(Avx2.IsSupported);
            Assert.True(Sse.IsSupported);
        }

        float[] pVect1 = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f };
        float[] pVect2 = new float[] { 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f };

        [Test]
        public void TestInnerProductDistance()
        {
            var space = new InnerProductDistance();
            float result = space.Distance(pVect1, pVect2);
            float expected = 1.0f - 36.0f;

            Assert.AreEqual(expected, result, 1e-6);
        }

        [Test]
        public void TestInnerProductDistanceSIMD4ExtAVX()
        {
            var space = new InnerProductDistanceSIMD4ExtAVX();
            float result = space.Distance(pVect1, pVect2);
            float expected = 1.0f - 36.0f;

            Assert.AreEqual(expected, result, 1e-6);
        }

        [Test]
        public void TestInnerProductDistanceSIMD4ExtSSE()
        {
            var space = new InnerProductDistanceSIMD4ExtSSE();
            float result = space.Distance(pVect1, pVect2);
            float expected = 1.0f - 36.0f;

            Assert.AreEqual(expected, result, 1e-6);
        }

        float[] vect1b = new float[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f };
        float[] vect2b = new float[] { 1.0f, 0.5f, 1.5f, 0.25f, 1.25f, 0.75f, 1.75f, 0.125f, 1.125f, 0.625f, 1.625f, 0.375f, 1.375f, 0.875f, 1.875f, 0.0625f };


        [Test]
        public void TestInnerProductDistanceSIMD16ExtAVX()
        {
            var space = new InnerProductDistanceSIMD16ExtAVX();
            float result = space.Distance(vect1b, vect2b);
            float expected = 1.0f - 129.5f;

            Assert.AreEqual(expected, result, 1e-6);
        }

        [Test]
        public void TestInnerProductDistanceSIMD16ExtSSE()
        {
            var space = new InnerProductDistanceSIMD16ExtSSE();
            float result = space.Distance(vect1b, vect2b);
            float expected = 1.0f - 129.5f;

            Assert.AreEqual(expected, result, 1e-6);
        }

        [Test]
        public void TestInnerProductDistanceSIMD16ExtResiduals()
        {
            var vect1bl = new[] { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f };
            var vect2bl = new[] { 1.0f, 0.5f, 1.5f, 0.25f, 1.25f, 0.75f, 1.75f, 0.125f, 1.125f, 0.625f, 1.625f, 0.375f, 1.375f, 0.875f, 1.875f, 0.0625f, 1.0f, 0.5f };

            var space = new InnerProductDistanceSIMD16ExtResiduals(new InnerProductDistanceSIMD16ExtAVX());
            var spaceExpected = new InnerProductDistance();

            var result = space.Distance(vect1bl, vect2bl);
            var expected = spaceExpected.Distance(vect1bl, vect2bl);

            Assert.AreEqual(expected, result, 1e-6);
        }
    }
}