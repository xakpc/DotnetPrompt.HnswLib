using System.Diagnostics;
using System.Runtime.InteropServices;
using DotnetPrompt.HnswLib.Space;

namespace DotnetPrompt.HnswLib;

public class BruteforceSearch : IIndexBasedNearestNeighborSearch
{
    private Dictionary<int, float[]> _dataDictionary = new();
    private readonly IVectorSpace<float[]> _vectorSpace;

    public BruteforceSearch(IVectorSpace<float[]> vectorSpace)
    {
        _vectorSpace = vectorSpace;
    }

    /// <inheritdoc />
    public void AddPoint(ReadOnlySpan<float> datapoint, int label)
    {
        _dataDictionary.Add(label, datapoint.ToArray());
    }

    /// <inheritdoc />
    public IEnumerable<(float distance, int label)> SearchKnnCloserFirst(float[] queryData, int k, Func<int, bool>? isIdAllowed = null)
    {
        var result = SearchKnn(queryData.ToArray(), k, isIdAllowed);

        for (var i = result.Count - 1; i >= 0; i--)
        {
            yield return result.Dequeue();
        }
    }

    private PriorityQueue<(float, int), float> SearchKnn(float[] queryData, int k, Func<int, bool>? isIdAllowed = null)
    {
        if (k >= _dataDictionary.Count)
            throw new ArgumentException("k must be smaller than the number of data points");

        var topResults = new PriorityQueue<(float, int), float>();

        if (_dataDictionary.Count == 0)
            return topResults;

        // Calculate the distances to the first k data points
        for (var i = 0; i < k; i++)
        {
            var dist = _vectorSpace.Distance(queryData, _dataDictionary.Values.ElementAt(i));
            var label = _dataDictionary.Keys.ElementAt(i);

            if (isIdAllowed == null || isIdAllowed(label))
            {
                topResults.Enqueue((dist, label), dist);
            }
        }

        var lastDist = topResults.Count == 0 ? default : topResults.Peek();

        // Iterate over the rest of the data points and update the top k results
        for (var i = k; i < _dataDictionary.Count; i++)
        {
            var dist = _vectorSpace.Distance(queryData, _dataDictionary.Values.ElementAt(i));

            if (dist <= lastDist.Item1)
            {
                var label = _dataDictionary.Keys.ElementAt(i);

                if (isIdAllowed == null || isIdAllowed(label))
                {
                    topResults.Enqueue((dist, label), dist);
                }

                if (topResults.Count > k)
                    topResults.Dequeue();

                if (topResults.Count > 0)
                {
                    lastDist = topResults.Peek();
                }
            }
        }

        return topResults;
    }
}