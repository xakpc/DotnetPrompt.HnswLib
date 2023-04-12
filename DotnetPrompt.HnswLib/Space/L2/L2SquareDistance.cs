namespace DotnetPrompt.HnswLib.Space.L2;

/// <summary>
/// Internal interface to publish Distance method to Residuals
/// </summary>
/// <typeparam name="T"></typeparam>
internal interface IVectorPartSpace<T>
{
    float Distance(T vector1, T vector2, int qty);
}

/// <summary>
/// Calculate the L2 square distance between two vectors
/// </summary>
public class L2SquareDistance : IVectorSpace<float[]>, IVectorPartSpace<float[]>
{
    private static float L2SquareDistanceImp(ReadOnlySpan<float> pointA, ReadOnlySpan<float> pointB, int qty)
    {
        float res = 0; // Initialize the result variable to store the L2 square distance

        // Iterate through the vectors and calculate the L2 square distance
        for (var i = 0; i < qty; i++)
        {
            var t = pointA[i] - pointB[i]; // Compute the difference between corresponding elements of the vectors
            res += t * t; // Add the square of the difference to the result
        }

        return res; // Return the L2 square distance
    }

    /// <inheritdoc />
    public float Distance(float[] vector1, float[] vector2)
    {
        if (vector1.Length != vector2.Length)
            throw new ArgumentException("Vectors must be of the same length", nameof(vector2));

        return L2SquareDistanceImp(vector1, vector2, vector1.Length);
    }

    public float Distance(float[] vector1, float[] vector2, int qty)
    {
        return L2SquareDistanceImp(vector1, vector2, qty);
    }
}