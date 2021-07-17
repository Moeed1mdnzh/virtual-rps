import cv2
import numpy as np 
from tensorflow import lite
from preprocess import Preprocess
from predictor import Classifier, MovePredictor

cam = cv2.VideoCapture(0)
names = ["rock.png", "paper.png", "scissors.png"]
images = [cv2.resize(cv2.imread(f"assests//{name}") if name != "rock.png" else cv2.flip(cv2.imread(f"assests//{name}"), 1), (128, 128)) for name in names]
X, y = [], []
rounds = 1
player_score = 0 
computer_score = 0
choice_details = {"You Win":[(200, 160), (50, 255, 50), 0.9], "Computer Wins":[(155, 160), (50, 50, 255), 0.8], "Tie":[(230, 160), (50, 255, 255), 1.1]}
move = 1
winners = [0, 0]

###Logic for finding the winner

class GameLogic:
	def __init__(self, player_move, computer_move):
		self.player_move = player_move
		self.computer_move = computer_move

	def computerWin(self):
		conditions = [
					self.computer_move == "rock" and self.player_move == "scissors",
					self.computer_move == "paper" and self.player_move == "rock",
					self.computer_move == "scissors" and self.player_move == "paper"
		]
		for c in conditions:
			if c:
				return "Computer Wins", [0, 1]
		return "", [0, 0]

	def playerWin(self):
		conditions = [
					self.player_move == "rock" and self.computer_move == "scissors",
					self.player_move == "paper" and self.computer_move == "rock",
					self.player_move == "scissors" and self.computer_move == "paper"
		]
		for c in conditions:
			if c:
				return "You Win", [1, 0]
		return "", [0, 0]

	def tie(self):
		if self.player_move == self.computer_move:
			return "Tie", [1, 1]
		return "", [0, 0]

	def findWinner(self):
		for msg, winner in [self.computerWin(), self.playerWin(), self.tie()]:
			if winner[0] or winner[1]:
				return msg, winner

if __name__ == '__main__':
	preprocessor = Preprocess(None)
	interpreter = lite.Interpreter("rps.tflite")
	classifier = Classifier(interpreter)
	move_predictor = MovePredictor()
	###Main code
	while True:
		_, frame = cam.read()
		frame = cv2.flip(frame, 1)
		frame = cv2.resize(frame,(512,384))
		clone = frame.copy()
		frame, sample, box = preprocessor.preprocess(frame.copy(), player_score, computer_score, rounds)
		if len(sample) > 0: #If any hand was found in the bounding box then . . .
			label = classifier.predict(sample)
			if rounds == 1:
				computer_move = move_predictor.chooseRandom()
			else: #Start AI prediction after the first round
				move_predictor.create()
				move_predictor.train(np.array(X), np.array(y))
				computer_move = int(move_predictor.predict([[move]])[0])
			if box == "left":
				img = images[label].copy()
				img[img == 151] = clone[0:128, 50:50+128][img == 151]
				clone[0:128, 50:50+128] = img
				img2 = cv2.flip(images[computer_move].copy(), 1)
				img2[img2 == 151] = clone[0:128, 462-128:462][img2 == 151]
				clone[0:128, 462-128:462] = img2
			else:
				img = cv2.flip(images[label].copy(), 1)
				img[img == 151] = clone[0:128, 462-128:462][img == 151]
				clone[0:128, 462-128:462] = img
				img2 = images[computer_move].copy()
				img2[img2 == 151] = clone[0:128, 50:50+128][img2 == 151]
				clone[0:128, 50:50+128] = img2
			logic = GameLogic(names[label].replace(".png", ""), names[computer_move].replace(".png", ""))
			msg, winner = logic.findWinner()
			if msg != "Tie":
				player_score += winner[0]
				computer_score += winner[1]
			if player_score == 5 or computer_score == 5:
				if player_score == 5:
					winners[0] += 1 
				else:
					winners[1] += 1
				rounds += 1 
				player_score = 0 
				computer_score = 0
			for k, (cord, color, scale) in choice_details.items():
				if k == msg:
					cv2.putText(clone, msg, cord, cv2.FONT_HERSHEY_TRIPLEX, scale, color, 2)

			#Collect numeric data for AI's potential predictions

			X.append([move])
			y.append(label)
			waitTime = 3000
			move += 1

			#Check the conditions of ending the game

			if rounds == 4 or (winners[0] == 2 and winners[1] == 0) or (winners[1] == 2 and winners[0] == 0):
				mask = np.zeros(clone.shape, np.uint8)
				winner = np.argmax(np.array(winners).reshape(-1, 2))
				if winner == 0:
					cv2.putText(mask, "You Won", (170, 192), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
				else:
					cv2.putText(mask, "Computer Won", (120, 192), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
				cv2.imshow("result", mask)
				X, y = [], []
				rounds = 1
				player_score = 0 
				computer_score = 0
				move = 1
				winners = [0, 0]
				waitTime = 5000
			else:
				cv2.imshow("result", clone)
				waitTime = 3000
			cv2.destroyWindow("Frame")
			cv2.waitKey(waitTime)
			cv2.destroyWindow("result")
		cv2.imshow("Frame", frame)
		k = cv2.waitKey(30) & 0xff 
		if k == ord("q"):
			break
			
	cv2.destroyAllWindows()
	cam.release()
