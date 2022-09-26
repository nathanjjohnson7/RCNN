# RCNN
Implementation of Efficient Graph-Based Image Segmentation, Selective Search and RCNN

This basic implementation of the RCNN algorithm does not use 21 class specific SVMs or Bounding Box Regression. It only uses one CNN which is trained on the data that the SVMs would have been trained with. Furthermore, the images shown below used a model that was only trained for 7000 epochs.

Run time is approximately 2.5 to 3 seconds.

![alt text](https://github.com/nathanjjohnson7/RCNN/blob/main/results/baseball_players.png?raw=true)

![alt text](https://github.com/nathanjjohnson7/RCNN/blob/main/results/car.png?raw=true)

![alt text](https://github.com/nathanjjohnson7/RCNN/blob/main/results/plane.png?raw=true)

![alt text](https://github.com/nathanjjohnson7/RCNN/blob/main/results/sheep.png?raw=true)
