using DotnetPrompt.HnswLib.Space;

namespace DotnetPrompt.HnswLib.Test
{
    [TestFixture]
    public class BruteforceTest
    {
        [Test]
        public void TestBruteforce()
        {
            const int dimension = 4;
            const int n = 100;
            const int numQuery = 10;
            const int k = 10;

            var data = new float[n * dimension];
            var query = new float[numQuery * dimension];

            var rng = new Random(47);

            for (var i = 0; i < n * dimension; ++i)
            {
                data[i] = (float)rng.NextDouble();
            }
            for (var i = 0; i < numQuery * dimension; ++i)
            {
                query[i] = (float)rng.NextDouble();
            }

            // Initialize the brute force algorithm
            // Replace 'BruteforceSearch' and 'L2Space' with their C# equivalents
            var selector = new InnerProductStrategySelector();
            var space = selector.SelectSpaceStrategy(Spaces.L2, dimension);
            IIndexBasedNearestNeighborSearch algBrute = new BruteforceSearch(space);

            for (var i = 0; i < n; ++i)
            {
                algBrute.AddPoint(data.AsSpan(dimension * i, dimension), i);
            }

            // Test searchKnnCloserFirst of BruteforceSearch
            // Replace 'searchKnn' and 'searchKnnCloserFirst' with their C# equivalents
            for (var j = 0; j < numQuery; ++j)
            {
                var querySpan = query.AsSpan(j * dimension, dimension);

                var gd = algBrute.SearchKnn(querySpan.ToArray(), k).ToList();
                var res = algBrute.SearchKnnCloserFirst(querySpan.ToArray(), k).ToList();
                
                Assert.AreEqual(gd.Count, res.Count);

                for (int i = 0; i < gd.Count; i++)
                {
                    Assert.AreEqual(gd[i].distance, res[^(i+1)].distance);
                }
            }
        }
    }
}
