using DotnetPrompt.HnswLib.Space;

namespace DotnetPrompt.HnswLib.Test;

[TestFixture]
public class HnswIndexTests
{
    [Test]
    public void TestInsertionAndSearch()
    {
        var config = new HnswConfig
        {
            M = 10,
            MaxLevel = (int)Math.Floor(Math.Log(1000)/Math.Log(10)),
            VectorDimension = 50,
            EfConstruction = 50,
            EfSearch = 50,
            Space = Spaces.L2
        };

        var index = new HnswIndex(config);
        var rnd = new Random(42);
        int numPoints = 1000;
        int dimensions = 50;

        for (int i = 0; i < numPoints; i++)
        {
            float[] point = new float[dimensions];
            for (int j = 0; j < dimensions; j++)
            {
                point[j] = (float)rnd.NextDouble();
            }
            index.Add(point);
        }

        float[] query = new float[dimensions];
        for (int j = 0; j < dimensions; j++)
        {
            query[j] = (float)rnd.NextDouble();
        }

        int k = 10;
        var neighbors = index.Search(query, k);

        Assert.AreEqual(k, neighbors.Count);
    }

    [Test]
    public void TestSerialization()
    {
        var config = new HnswConfig
        {
            M = 10,
            MaxLevel = (int)Math.Floor(Math.Log(1000) / Math.Log(10)),
            EfConstruction = 50,
            EfSearch = 50,
            Space = Spaces.L2
        };

        var index = new HnswIndex(config);
        var rnd = new Random(42);
        int numPoints = 1000;
        int dimensions = 50;

        for (int i = 0; i < numPoints; i++)
        {
            float[] point = new float[dimensions];
            for (int j = 0; j < dimensions; j++)
            {
                point[j] = (float)rnd.NextDouble();
            }
            index.Add(point);
        }

        string filePath = "test_index.bin";
        index.Save(filePath);

        var loadedIndex = HnswIndex.Load(filePath);
        File.Delete(filePath);

        float[] query = new float[dimensions];
        for (int j = 0; j < dimensions; j++)
        {
            query[j] = (float)rnd.NextDouble();
        }

        int k = 10;
        var neighbors = loadedIndex.Search(query, k);

        Assert.AreEqual(k, neighbors.Count);
    }

}