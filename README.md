# About-The-Game
This is the simple classic rock paper scissors game along the creativity of computer vision where your computer tries to estimate your hand and detect the gesture of it.
The game has a special part that the computer intends to predict your next moves based on a pattern by ***regression***.Pretty cool, isn't it?

## DL-Algorithms
### Hand-Estimation
For this part, google's open source ***MediaPipe*** is used in the game.
### Gesture-Detection
CNN (*Convolutional Neural Network*) gave the best results and performance compared to ML (*Machine Learning*) algorithms such as SVM, Randomforest and etc and by using 
cnn models I luckily achieved about 100 percent accuracy on the validation set which is quite delightful.One thing to mention is that a couple of data augmentation methods from my
data augmentation repo ![Data-Augmentor](https://github.com/Moeed1mdnzh/Data-Augmentor) is used to perform the best on the training set.
![](https://github.com/Moeed1mdnzh/virtual-rps/blob/main/images/performance.png)
***As you can see the model did a great job on learning using the SGD optimizer***
