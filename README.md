# Simultaneous Multi-View 3D Object Recognition and Grasp Pose Estimation

### Preparing Data
Download and unzip the Modelet10 dataset from the official distribution:
http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip (451MB)

Ensure you are running Python>=3.6 and import the required libraries by running:
```
pip install -r requirements.txt
```

WARNING: The dataset is quite large and generation scripts may take 
hours to complete.

To generate the entropy dataset run:
```
python generate_entropy_dataset.py --modelnet10 <dir>
```
substitute `<dir>` with the root directory of the ModelNet10 dataset.

To generate the view dataset run:
```
python generate_view_dataset.py --modelnet10 <dir> --set <set>
```
substitute `<dir>` with the root directory of the ModelNet10 dataset
and `<set>` with either `train` or `test` to generate the views from 
the respective split.

[***]




```
python entropy_model.py --data <dir> --csv <entropy_dataset>
```