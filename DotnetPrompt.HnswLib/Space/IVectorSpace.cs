namespace DotnetPrompt.HnswLib.Space;

public interface IVectorSpace<T>
{
    /// <summary>
    /// Calculate distance between two vectors
    /// </summary>
    /// <param name="vector1"></param>
    /// <param name="vector2"></param>
    /// <returns></returns>
    float Distance(T vector1, T vector2);
}