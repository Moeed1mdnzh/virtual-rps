# About The Game
This is the simple classic rock paper scissors game along the creativity of computer vision where your computer tries to estimate your hand and detect the gesture of it.
The game has a special part that the computer intends to predict your next moves based on a pattern by ***regression***.Pretty cool, isn't it?
The game has 3 rounds and whoever reaches 5 scores will win the specific round.Computer makes random choices in the first round to collect data about your movements and
will start the predictive choices in the second round and will also continue to collect more and more data.

## Algorithms
### Hand-Estimation
For this part, google's open source ***MediaPipe*** is used in the game.
### Gesture-Detection
CNN (*Convolutional Neural Network*) gave the best results and performance compared to ML (*Machine Learning*) algorithms such as SVM, Randomforest and etc and by using 
cnn models I luckily achieved about 100 percent accuracy on the validation set which is quite delightful.One thing to mention is that a couple of data augmentation methods from my
data augmentation repo <a href="https://github.com/Moeed1mdnzh/Data-Augmentor">Data_Augmentor</a> is used to perform the best on the training set.
![](https://github.com/Moeed1mdnzh/virtual-rps/blob/main/images/performance.png)
***As you can see the model did a great job on learning using the SGD optimizer***

### Move Predictor
I used ensembled regressions to predict the next move of the player. <br /> Ensemble of <br /> *DecisionTreeRegressor* 
<br /> *HuberRegressor* <br /> *LinearRegression* <br /> *RandomForestRegressor* 
<br />*GradientBoostingRegressor*

## Steps
#### Preprocess
clone the repo <br />
`git clone https://github.com/Moeed1mdnzh/virtual-rps.git`
Install the packages 
```python
pip install -r requirements.txt 
``` 
#### Collect data
Use `cd data` to enter the directory of data collection and by typing 
```python
python collector.py
``` 
start collecting your desired data.The file itself will
guid you.Keep your hand in the bounding box and pay attention to the printed messages to change gestures.It will start with the rock gesture and collect
1000 images per class so the total will be about 3000 images.
#### Train
Use `cd ..` to return to the project directory and run 
```python
python trainGesture.py
``` 
to train the cnn model on the data.Remember you can also use google colab to train the model without needing a particular gpu on your own machine which i did too but you'll need
to do a couple of extra things such as uploading the data on google colab.
#### Convert
Convert the model to a tensorflow lite model to run the model on cpu without having to worry about low fps frames by running
```python
python liteConverter.py
``` 
### Play
Eventually you can play the game using the following command.
```python
python main.py
``` 
Just to give a quick guide about the game buttons <br />
![](https://github.com/Moeed1mdnzh/virtual-rps/blob/main/images/help.jpg) <br />
The yellow bounding box is for gesture detection.Once you keep your hand in the box, the program waits 0.5 seconds for you to finalize your hand gesture and then detects.<br />
The top *left* and *right* buttons are obvious that are meant to change the location of the yellow bounding box to the opposite direction for your main hand.You can use them
by shaking your index finger up and down (**Almost like clicking but more dramatic**), if it didn't work do it more until it works properly.<br />
The rounds and scores are also shown
on the top and sides.

