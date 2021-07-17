import numpy as np
from tensorflow import lite
from sklearn.svm import SVR
from scipy.stats import mode
from sklearn.ensemble import VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

###Classify hand gesture

class Classifier:
	def __init__(self, interpreter):
		self.interpreter = interpreter
		interpreter.allocate_tensors()
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()

	def prepare(self, sample):
		self.interpreter.set_tensor(self.input_details[0]['index'], np.float32(sample))
		self.interpreter.invoke()

	def predict(self, sample):
		prepared = self.prepare(sample)
		labels = []
		for _ in range(5):
			predictions = self.interpreter.get_tensor(self.output_details[0]['index'])
			label = np.argmax(predictions)
			labels.append(label)
		prediction = mode(labels).mode[0]
		return prediction


###Model to predict the next movement of the player after the first round

class MovePredictor:
	def __init__(self):
		self.reg = None

	def chooseRandom(self):
		return np.random.randint(0, 3)

	def create(self):
		self.reg = VotingRegressor([
		    ("r1", SVR(kernel="rbf", C=1.0)), 
		    ("r2", DecisionTreeRegressor(max_depth=2)),
		    ("r3", HuberRegressor()),
		    ("r4", LinearRegression()),
		    ("r5", GradientBoostingRegressor(max_depth=2, n_estimators=120, learning_rate=0.1)),
		    ("r6", RandomForestRegressor(max_depth=2)) 
		])

	def train(self, X, y):
		self.reg.fit(X, y)

	def predict(self, sample):
		return self.reg.predict(sample)