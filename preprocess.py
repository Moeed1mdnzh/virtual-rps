import cv2 
import time
import numpy as np 
from mediapipe import solutions

###Anything related to designs and hand estimation is provided here

class Preprocess:
	def __init__(self, handEstimator):
		self.estimator = handEstimator 
		self.boxes = [[(10, 124), (162, 276)], [(504, 124), (360, 276)]]
		self.box = self.boxes[0]
		self.hand_base = solutions.hands
		self.hand = self.hand_base.Hands()
		self.computer_score = 0 
		self.self_score = 0
		self.rounds = 1
		self.thicknesses = [4, 4]
		self.first_attempt = 0
		self.pre = 0
		self.boxHand = self.hand_base.Hands(min_detection_confidence=0.6)
		self.sample = []
		self.box_loc = "left"
		self.texts = [f"Your score : {self.self_score}", f"Computer's score : {self.computer_score}"]
		self.x, self.pre_y = None, None

	def selectBox(self, frame : np.ndarray) -> np.ndarray: #Choose the left or right box based on your main hand
		cv2.rectangle(frame, (120, 0), (256, 64), (100, 100, 255), self.thicknesses[0])
		cv2.rectangle(frame, (256, 0), (384, 64), (255, 100, 100), self.thicknesses[1])
		self.thicknesses = [4, 4]
		cv2.line(frame, (256, 0), (256, 64), (100, 100, 100), 5)
		cv2.putText(frame, "Left", (168 ,40), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (20, 20, 20), 2)
		cv2.putText(frame, "Right", (296 ,40), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (20, 20, 20), 2)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		res = self.hand.process(rgb)
		H, W = frame.shape[:2]
		if res.multi_hand_landmarks:
			for hand_landmarks in res.multi_hand_landmarks:
				finger = hand_landmarks.landmark[self.hand_base.HandLandmark.INDEX_FINGER_TIP]
				X,Y = int(finger.x*W),int(finger.y*H)
				if self.pre_y is not None: 
					dist = Y-self.pre_y
					if dist >= 45: #Box selection based on the previous and current movements of your index finger
						for i, box in enumerate([[(120, 0), (256, 74)], [(256, 0), (384, 74)]]):
							if (self.pre_x >= box[0][0] and self.pre_x <= box[1][0]) and (self.pre_y >= box[0][1] and self.pre_y <= box[1][1]):
								if i == 0: 
									self.box = self.boxes[0]
									self.texts = [f"Your score : {self.self_score}", f"Computer's score : {self.computer_score}"]
									self.thicknesses = [-1, 4]
									self.box_loc = "left"
								else: 
									self.box = self.boxes[1]
									self.texts = [f"Computer's score : {self.computer_score}", f"Your score : {self.self_score}"]
									self.thicknesses = [4, -1]
									self.box_loc = "right"
				self.pre_y, self.pre_x = Y, X
		#print(box_loc)
		return frame, self.box_loc

	def displayDetails(self, frame : np.ndarray) -> np.ndarray: #Displaying scores and rounds
		self.texts = [f"Your score : {self.self_score}", f"Computer's score : {self.computer_score}"]
		cv2.putText(frame, self.texts[0], (10 ,370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 2)
		cv2.putText(frame, self.texts[1], (332 ,370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 255, 180), 2)
		cv2.putText(frame, f"Round : {self.rounds}", (211, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)
		return frame

	def estimate(self, frame : np.ndarray) -> np.ndarray: #Hand estimation in the box for classification
		if self.box == self.boxes[1]:
			cropped = frame[self.box[0][1]:self.box[1][1], self.box[1][0]:self.box[0][0]]
		else:
			cropped = frame[self.box[0][1]:self.box[1][1], self.box[0][0]:self.box[1][0]]
		cropped = cv2.resize(cropped, (64, 64))
		if not self.first_attempt:
			rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
			res = self.boxHand.process(rgb)
			if res.multi_hand_landmarks: 
				self.pre = time.time()
				self.first_attempt = 1
				self.sample = [] 
		if self.first_attempt:
			frame, check = self.countDown(frame, self.pre)
			if check:
				rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
				res = self.boxHand.process(rgb)
				if res.multi_hand_landmarks: 
					self.first_attempt = 0
					self.pre = 0
					self.sample = cropped.copy()
					return frame, self.sample
				else:
					self.first_attempt = 0
					self.pre = 0
		self.sample = []
		cv2.rectangle(frame, self.box[0], self.box[1], (140, 255, 255), 3)
		return frame, self.sample

	def countDown(self, frame : np.ndarray, pre : float) -> np.ndarray : #Whenever a hand comes inside the box, wait 0.5 seconds and then check again
		spent = round(time.time()-pre, 1)
		if spent <= 0.5:
			cv2.putText(frame, f"{spent}", (232, 296), cv2.FONT_HERSHEY_COMPLEX, 1.8, (66, 123, 245), 5)
			cv2.putText(frame, f"{spent}", (233, 295), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
		else:
			return frame, True
		return frame, False

	def preprocess(self, frame : np.ndarray, self_score : int=0, computer_score : int=0, rounds : int=1) -> np.ndarray: #Main function
		self.self_score = self_score
		self.computer_score = computer_score
		self.rounds = rounds
		frame, sample = self.estimate(frame)
		frame, box = self.selectBox(frame)
		frame = self.displayDetails(frame)
		sample = [] if len(sample) == 0 else cv2.cvtColor(sample, cv2.COLOR_BGR2RGB).reshape(-1,64,64,3) / 255.0
		return frame, sample, box
