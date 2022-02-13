import random

def get_training_samples(batch_size):
	with open("train.csv") as file:
		text = file.read()
	textlines = text.strip().split("\n")
	random.shuffle(textlines)
	start = 0
	while start < len(textlines):
		digits = []
		targets = []
		inputs = []
		end = start + batch_size
		for textline in textlines[start:end]:
			cells = textline.split(",")
			digits.append(int(cells[0]))
			targets.append([float(c) for c in cells[1:11]])
			inputs.append([float(c) for c in cells[11:]])
		yield digits, targets, inputs
		start += batch_size

def get_test_samples():
	with open("test.csv", "r") as file:
		text = file.read()
	textlines = text.strip().split("\n")
	digits = []
	targets = []
	inputs = []
	for textline in textlines:
		cells = textline.split(",")
		value = int(cells[0])
		digits.append(int(cells[0]))
		targets.append([float(c) for c in cells[1:11]])
		inputs.append([float(c) for c in cells[11:]])
	return digits, targets, inputs

def plot_number(inputs):
	line = ""
	for p in inputs:
		line += ".░▒▓█"[round(p * 4)]
		if len(line) > 27:
			print(line)
			line = ""
