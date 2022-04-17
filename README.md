# Multimedia retrieval system
 
The practical part of the multimedia retrieval course consisted of creating a multimedia retrieval system. This system analyzes a 3D mesh and finds 3D meshes in its database that visually resemble the input mesh as much as possible (It is similar to a reverse image search on Google where an image is uploaded and Google finds images that are most similar. The difference is that this system uses 3D meshes instead of images). <br>

## Dependencies
The system uses [Vedo](https://vedo.embl.es/) for visualizing the meshes and the [Labeled PSB database](https://people.cs.umass.edu/~kalo/papers/LabelMeshes/) provides the meshes for comparison. The Labeled PSB database contains 380 meshes from 19 different classes. The meshes in the databse are first preprocessed to increase the retrieval speed and to remove certain biases in the data. One such bias is the average number of faces in each class. Certain classes contain more complex meshes (e.g. a mesh from the 'human' class is more complex than a mesh from the 'cup' class) and these tend to have more faces. To prevent the system from relying on the number of faces to find the most similar meshes, each mesh is changed to lower the number of faces. Additionally the meshes are all rotated such that the largest eigenvector aligns with the x-axis (using PCA), the meshes are all scaled and translated such that the barycenter of the mesh is equal to the origin. The preprocessed database can be found here: https://drive.google.com/file/d/131Hb6CQZZdvHval4OoJWs5DmsdwWM527/view?usp=sharing <br>

## Features
As the system is not able to 'see' the meshes like humans, it has to rely on non-visual information. This can be done using features, which are computed using the vertices which make up the mesh. In order to find similar meshes the system uses a combination of simple features, which give a single numerical value (e.g. the diameter), and distribution features. For the distribution features, a value is computed 10.000 - 1.000.000 times (depending on the feature) based on the random vertices (points) that make up the meshes. These 10.000 - 1.000.000 values are then divided between a number of 'bins' to represent the distribution. E.g. if you were to look at the distance between the barycenter and the vertices of a sphere then all these distances will be roughly the same (equal to the radius), but if you do this for a cube you will get a different distribution.
6 simple features:
- Surface area
- Compactness (in comparison to a sphere)
- Bounding box volume (after aligning to the axes)
- Diameter
- Eccentricity (the ratio between the largest and smallest eigenvalues)
These 6 simple features are combined with distribution features:
- Angle between three random vertices (points)
- Distance between the barycenter (origin) and a random vertex (point)
- Distance between two random vertices 
- Square root of area of a triangle made up of three random vertices (points)
- Cube root of volume of tetrahedron formed by four random vertices (points)

## Querying
As there are 380 meshes in the database, finding the most similar meshes requires at least 379 comparisons and thus comparing two meshes has to be done quickly. To prevent repeating the same work, the features above are computed once and stored for when the system needs to compare meshes. For comparing two meshes the system computes a distance between them (the larger the distance, the less similar the meshes) based on the Euclidean distance for the simple features. For the distribution features the Earth mover's distance is used instead as this lends itself better for comparing distribution than the Euclidean distance does.  <br>
Not all the features described above are equally useful and thus the features are weighted to maximize the performance of the system. The diameter and the bounding box volume, for example, have a small weight as they have a relatively high intra-class distance and relatively low inter-class distance. Meanwhile, the surface area is quite informative and thus has a larger weight.
# K-nearest neighbors
Although the database used only has 380 meshes, the system should also be able to use different databases and these can contain many more meshes. Therefore, the system needs to be able to compare two meshes even quicker than just comparing the feature values. For this purpose the system can use K-nearest neighbors to find similar meshes. K-nearest neighbors reduced the number of comparisons that need to be made as it already knows the distances/similarities between the meshes in its database and thus it can use this information.

## Evaluation
As the goal of the system is to find meshes that are **visually** similar, it is not straightforward how to evaluate the performance. This is because the meshes are labeled with a class but the database does not contain ground-truth distance between the meshes. We can use the class labels and determine that if a cup is queried, then the results should ideally consist of cups only. It is not clear which cup should be the first result and which cup the second, for example. <br>
When looking at the performance using the F1 measure, which combines recall and precision, the results are the following:
<img src="https://user-images.githubusercontent.com/45210768/156941105-e53955e2-0256-4101-b6e0-e46082a18487.jpg" width="200" alt="Home screen"/>




