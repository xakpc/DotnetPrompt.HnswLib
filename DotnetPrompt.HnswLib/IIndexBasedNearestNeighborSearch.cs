namespace DotnetPrompt.HnswLib;

public interface IIndexBasedNearestNeighborSearch
{
    void AddPoint(ReadOnlySpan<float> datapoint, int label);

    IEnumerable<(float distance, int label)> SearchKnn(float[] queryData, int k, Func<int, bool>? isIdAllowed = null)
    {
        var ret = SearchKnnCloserFirst(queryData, k, isIdAllowed); // Here "searchKnn" returns the result in the order of closest first
        return ret.Reverse();
    }

    IEnumerable<(float distance, int label)> SearchKnnCloserFirst(float[] queryData, int k, Func<int, bool>? isIdAllowed = null);
}