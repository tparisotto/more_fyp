##Final Year Project

###Simultaneous Multi-View 3D Object Recognition and Grasp Pose Estimation

#### Usage
Ensure you are running Python 3

Import the required libraries by running:

```
pip install -r requirements.txt
```

Then run the script with:

```
python generate_views.py filename
```
substitute `filename` with the 3D object file with `.off` extension (other extensions are also compatible).
The script will create a `out` folder with two subfolders `depth` and `image` which will contain respectively the depth images and the greyscale images of the rendered views.

The script will also generate a `.csv` file with the entropy values of the different views alongside their positions. The positions are coded as `x` and `y` values, which stand for the rotations of the object on the x-axis and y-axis from its original position, hence simulating the camera moving on a spherical structure around the object.