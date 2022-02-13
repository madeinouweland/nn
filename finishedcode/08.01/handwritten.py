import mnistreader as reader

epochs = 1
batch_size = 2000

digits, targets, inputs = reader.get_test_samples()
for v, i in zip(digits[:10], inputs[:10]):
	print(v)
	reader.plot_number(i)
	print()