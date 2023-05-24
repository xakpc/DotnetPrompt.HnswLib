# DotnetPrompt.HnswLib (WIP)
.NET-based Hierarchical Navigable Small World (HNSW) search library. 

HNSW is an efficient approximate nearest neighbor search algorithm that aims to find nearest neighbors in high-dimensional spaces.

DotnetPrompt.HnswLib actively utilizes AVX and SSE to calculate distances in a most efficient way.

## Components
1. HnswIndex - Manages the HNSW graph and provides APIs for insertion, search, and serialization
2. HnswNode - Represents a single data point in the HNSW graph
3. Space - Space property represents the similarity space in which the algorithm operates to calculate the distances between vectors when building the HNSW index and during search operations

Different spaces can be used depending on the nature of the data and the desired outcome. The most commonly used spaces in HNSW are:

* **Euclidean Space (L2)**: This space uses the Euclidean distance metric, which is the straight-line distance between two points in the multi-dimensional space. It is the default similarity space for HNSW and works well for many types of data. Euclidean distance is calculated as the square root of the sum of the squared differences between corresponding coordinates of the two points.
* **Cosine Space**: This space uses the cosine similarity metric, which measures the cosine of the angle between two non-zero vectors. It is often used in cases where the direction of the vectors is more important than their magnitudes, such as text data represented as word vectors. Cosine similarity ranges between -1 and 1, with 1 indicating that the vectors are pointing in the same direction, 0 meaning they are orthogonal, and -1 indicating they point in opposite directions. When using cosine similarity as a distance metric, it is common to normalize the input vectors before indexing and searching.
* **Inner Product Space**: another similarity space that can be used in the HNSW algorithm. In this space, the algorithm calculates the similarity between two vectors using their inner (dot) product. The inner product of two vectors is the sum of the product of their corresponding components. When working with normalized vectors, the inner product can provide results similar to cosine similarity. However, unlike cosine similarity, which ranges from -1 to 1 and only considers the angle between vectors, the inner product takes both the direction and magnitude of the vectors into account. In some cases, such as when using word embeddings or other types of dense vector representations, the inner product space can yield more meaningful results than Euclidean or cosine spaces.

4. HnswConfig - Represents configuration parameters for the HNSW index

## Algorithms

1. Insertion
 * Calculate the level for the new node
 * Traverse the HNSW graph from the entry point to find closest neighbors
 * Update neighbors for the new node and its neighbors
 * Update the entry point if the new node has a higher level
2. Search
 * Traverse the HNSW graph from the entry point to find the nearest neighbors
 * Return the nearest neighbors
3. Serialization
 * Serialize the HNSW graph to a binary format
 * Deserialize the HNSW graph from a binary format

## Real-world use cases where the HNSW algorithm can be applied

Nearest neighbor search in high-dimensional spaces: HNSW is particularly useful for finding nearest neighbors in high-dimensional spaces, which is a common requirement in many applications such as recommendation systems, image retrieval, and natural language processing.

1. Similarity search: You can use the HNSW algorithm to perform similarity search in various domains like text, images, and audio. By defining an appropriate distance metric, you can find items that are most similar to a given query item.
1. Clustering: HNSW can be employed as a building block for clustering algorithms, helping to identify dense regions in the data space or group similar items together.
1. Anomaly detection: By identifying points that are distant from their nearest neighbors, HNSW can be used to detect outliers or anomalies in the data, which could be useful in fraud detection or monitoring applications.
1. Feature extraction and dimensionality reduction: HNSW can be combined with other machine learning techniques to perform feature extraction or dimensionality reduction, which can help improve the efficiency and effectiveness of various data analysis tasks.
1. Text classification and document retrieval: By representing text documents as high-dimensional vectors, HNSW can be used to perform tasks like document retrieval, topic modeling, or text classification, helping to find relevant documents or classify them into predefined categories.
1. Content-based recommendation systems: HNSW can be applied to content-based recommendation systems to find items similar to a user's preferences, enabling personalized recommendations for users.

## Additional information 
If you seek additional information  or guidance on HNSW or related topics, here are some resources that you may find helpful:

1. HNSW Algorithm Paper: The original paper by Yu. A. Malkov and D. A. Yashunin, titled "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs," provides a detailed explanation of the algorithm, its properties, and experimental results. The paper can be found here: https://arxiv.org/abs/1603.09320
1. HNSW Libraries: There are several open-source libraries available that implement the HNSW algorithm. Some popular libraries include:
  * Faiss (C++ with Python bindings): https://github.com/facebookresearch/faiss
  * Annoy (C++ with Python bindings): https://github.com/spotify/annoy
  * hnswlib (C++ with Python bindings): https://github.com/nmslib/hnswlib
