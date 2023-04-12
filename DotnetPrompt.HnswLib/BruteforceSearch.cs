using System.Diagnostics;
using System.Runtime.InteropServices;
using DotnetPrompt.HnswLib.Space;

namespace DotnetPrompt.HnswLib;

public interface IBruteforceSearch<TDist>
{
    void AddPoint(IntPtr datapoint, int label, bool replaceDeleted = false);
    void RemovePoint(int label);
    PriorityQueue<TDist, int> SearchKnn(IntPtr queryData, int k, Func<int, bool>? isIdAllowed = null);
    void SaveIndex(string location);
    void LoadIndex(string location, Class2.ISpace<TDist> space);
}

public class BruteforceSearch : IIndexBasedNearestNeighborSearch
{
    private readonly IVectorSpace<float[]> _vectorSpace;
    private byte[] data_;
    private int maxelements_;
    private int cur_element_count_;
    private uint size_per_element_;

    private uint data_size_;
    private Func<float[], float[], int, float> fstdistfunc_;
    private IntPtr dist_func_param_;
    private object index_lock_;

    private Dictionary<int, float[]> _dataDictionary = new();

    private Dictionary<int, int> dict_external_to_internal_;

    public BruteforceSearch(Class2.ISpace<float> space)
    {
        data_ = null;
        maxelements_ = 0;
        cur_element_count_ = 0;
        size_per_element_ = 0;
        data_size_ = 0;
        dist_func_param_ = IntPtr.Zero;
        index_lock_ = new object();
    }

    public BruteforceSearch(Class2.ISpace<float> space, string location)
    {
        data_ = null;
        maxelements_ = 0;
        cur_element_count_ = 0;
        size_per_element_ = 0;
        data_size_ = 0;
        dist_func_param_ = IntPtr.Zero;
        index_lock_ = new object();
        LoadIndex(location, space);
    }

    public BruteforceSearch(Class2.ISpace<float> space, int maxElements)
    {
        maxelements_ = maxElements;
        data_size_ = space.DataSize;
        //fstdistfunc_ = space.DistFunc;
        dist_func_param_ = space.DistFuncParam;
        size_per_element_ = data_size_ + sizeof(int);
        data_ = new byte[maxElements * size_per_element_];
        if (data_ == null)
            throw new OutOfMemoryException("Not enough memory: BruteforceSearch failed to allocate data");
        cur_element_count_ = 0;
    }

    public BruteforceSearch(IVectorSpace<float[]> vectorSpace)
    {
        _vectorSpace = vectorSpace;
    }

    #region Interface IAlgorithm

    //public void AddPoint(float[] dataPoint, int label, bool replaceDeleted = false)
    //{
    //    Array.Copy(dataPoint, _dataDictionary[label], dataPoint.Length);
    //}

    /// <summary>
    /// 
    /// </summary>
    /// <param name="queryData"></param>
    /// <param name="k"></param>
    /// <param name="isIdAllowed"></param>
    /// <returns>result is sorted in ascending order</returns>
    /// <exception cref="ArgumentException"></exception>
    private PriorityQueue<(float, int), float> SearchKnn(float[] queryData, int k, Func<int, bool>? isIdAllowed = null)
    {
        if (k >= _dataDictionary.Count)
            throw new ArgumentException("k must be smaller than the number of data points");

        var topResults = new PriorityQueue<(float, int), float>();
        
        if (_dataDictionary.Count == 0)
            return topResults;

        // Calculate the distances to the first k data points
        for (int i = 0; i < k; i++)
        {
            float dist = _vectorSpace.Distance(queryData, _dataDictionary.Values.ElementAt(i));
            int label = _dataDictionary.Keys.ElementAt(i);

            if (isIdAllowed == null || isIdAllowed(label))
            {
                topResults.Enqueue((dist, label), dist);
            }
        }

        var lastDist = topResults.Count == 0 ? default : topResults.Peek();

        // Iterate over the rest of the data points and update the top k results
        for (int i = k; i < cur_element_count_; i++)
        {
            float dist = _vectorSpace.Distance(queryData, _dataDictionary.Values.ElementAt(i));

            if (dist <= lastDist.Item1)
            {
                int label = _dataDictionary.Keys.ElementAt(i);

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

    public void SaveIndex(string location)
    {
        //using (var output = new BinaryWriter(File.Open(location, FileMode.Create)))
        //{
        //    output.Write(maxelements_);
        //    output.Write(size_per_element_);
        //    output.Write(cur_element_count_);

        //    output.Write(data_, 0, maxelements_ * size_per_element_);
        //}
    }

    public void LoadIndex(string location, Class2.ISpace<float> s)
    {
        //using var input = new BinaryReader(File.Open(location, FileMode.Open));
        //maxelements_ = input.ReadInt32();
        //size_per_element_ = input.ReadInt32();
        //cur_element_count_ = input.ReadInt32();

        //data_size_ = s.DataSize;
        //fstdistfunc_ = s.DistFunc;
        //dist_func_param_ = s.DistFuncParam();
        //data_ = new byte[maxelements_ * size_per_element_];

        //input.Read(data_, 0, maxelements_ * size_per_element_);

        //// Update the label dictionary
        //dict_external_to_internal_.Clear();
        //for (int i = 0; i < cur_element_count_; i++)
        //{
        //    int label = BitConverter.ToInt32(data_, size_per_element_ * i + data_size_);
        //    dict_external_to_internal_[label] = i;
        //}
    }

    public void AddPoint(ReadOnlySpan<float> datapoint, int label)
    {
        _dataDictionary.Add(label, datapoint.ToArray());
    }

    public void RemovePoint(int label)
    {
        _dataDictionary.Remove(label);
    }

    public IEnumerable<(float distance, int label)> SearchKnnCloserFirst(float[] queryData, int k, Func<int, bool>? isIdAllowed = null)
    {
        var result = SearchKnn(queryData.ToArray(), k, isIdAllowed);

        for (var i = result.Count - 1; i >= 0; i--)
        {
            yield return result.Dequeue();
        }
    }

    #endregion
}