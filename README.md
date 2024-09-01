


# Amazon User Clustering using K-Means

This project applies K-Means clustering to segment Amazon users based on their annual income and purchase rating. The optimal number of clusters is determined using the Elbow Method, and the results are visualized to understand the segmentation.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project demonstrates the use of K-Means clustering to analyze and segment Amazon users based on their annual income and purchase rating. The goal is to group similar users together to identify distinct customer segments.

## Dataset

The dataset used for this project is titled `Amazon.com Clusturing.csv` and contains information about Amazon users. The specific features used for clustering are:

- **Annual Income (in INR)**
- **Purchase Rating**

## Installation

To run this project, you'll need Python and the necessary libraries. Follow these steps to set up your environment:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/amazon-user-clustering.git
   cd amazon-user-clustering
   ```

2. **Install the required libraries:**

   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

## Project Structure

```
amazon-user-clustering/
│
├── Amazon.com Clusturing.csv
├── clustering.py
├── README.md
└── requirements.txt
```

- `Amazon.com Clusturing.csv`: The dataset used for clustering.
- `clustering.py`: The script that performs data preprocessing, K-Means clustering, and visualization.
- `README.md`: Project documentation.
- `requirements.txt`: Lists the dependencies required to run the project.

## Usage

1. **Import the necessary libraries:**

   The script imports libraries such as NumPy, Pandas, Matplotlib, and Scikit-learn.

2. **Load and preprocess the dataset:**

   The dataset is loaded using Pandas, and the relevant features (Annual Income and Purchase Rating) are extracted for clustering.

   ```python
   df = pd.read_csv("Amazon.com Clusturing.csv")
   x = df.iloc[:, 3:5].values
   ```

3. **Determine the optimal number of clusters:**

   The Elbow Method is used to find the optimal number of clusters by plotting the within-cluster sum of squares (WCSS) against the number of clusters.

   ```python
   from sklearn.cluster import KMeans
   wcss = []
   for i in range(1, 11):
       kmeans = KMeans(n_clusters=i, init='k-means++', random_state=21)
       kmeans.fit(x)
       wcss.append(kmeans.inertia_)
   plt.plot(range(1, 11), wcss)
   ```

4. **Train the K-Means model:**

   The K-Means model is trained with the optimal number of clusters (in this case, 4), and the dataset is segmented accordingly.

   ```python
   kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
   y_means = kmeans.fit_predict(x)
   ```

5. **Visualize the clusters:**

   The clusters and centroids are visualized to understand the segmentation of Amazon users.

   ```python
   plt.scatter(x[y_means == 0, 0], x[y_means == 0, 1], s = 100, c = 'magenta', label = 'Cluster 1')
   plt.scatter(x[y_means == 1, 0], x[y_means == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
   plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
   plt.show()
   ```

6. **Run the script:**

   To execute the code and see the results, run:

   ```bash
   python clustering.py
   ```

## Methodology

- **Elbow Method:** Used to determine the optimal number of clusters by plotting WCSS values for different cluster counts.
- **K-Means Clustering:** Segments the dataset into distinct groups based on similarity.
- **Visualization:** Clusters are visualized using scatter plots to understand the grouping.

## Results

The script identifies four distinct clusters of Amazon users based on their annual income and purchase rating. The clusters are visualized using Matplotlib, showing the centroids and the data points in each cluster.

## Future Work

- Experiment with different clustering algorithms like DBSCAN or hierarchical clustering.
- Include additional features for clustering, such as user age or geographic location.
- Apply clustering to other types of user data for broader insights.

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have any improvements or suggestions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

