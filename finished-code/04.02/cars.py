inputs = [(0.2, 1600), (1.0, 11000), (1.4, 23000), (1.6, 24000), (2.0, 30000),
    (2.2, 31000), (2.7, 35000), (2.8, 38000), (3.2, 40000), (3.3, 21000), (3.5, 45000),
    (3.7, 46000), (4.0, 50000), (4.4, 49000), (5.0, 60000), (5.2, 62000)]
targets = [230, 555, 815, 860, 1140, 1085, 1200, 1330, 1290,
    870, 1545, 1480, 1750, 1845, 1790, 1955]

w1 = 0.1
w2 = 0.2
b = 0.3
epochs = 4000
learning_rate = 0.000000000005

def predict(i1, i2):
    return w1 * i1 + w2 * i2 + b

# train the network
for epoch in range(epochs):
    pred = [predict(i1, i2) for i1, i2 in inputs]
    cost = sum([(p - t) ** 2 for p, t in zip(pred, targets)]) / len(inputs)
    print(f"ep:{epoch}, c:{cost:.2f}")

    errors_d = [2 * (p - t) for p, t in zip(pred, targets)]
    weight1_d = [e * i[0] for e, i in zip(errors_d, inputs)]
    weight2_d = [e * i[1] for e, i in zip(errors_d, inputs)]
    bias_d = [e * 1 for e in errors_d]
    w1 -= learning_rate * sum(weight1_d) / len(weight1_d)
    w2 -= learning_rate * sum(weight2_d) / len(weight2_d)
    b -= learning_rate * sum(bias_d) / len(bias_d)

print(f"w1:{w1:.4f}, w2:{w2:.4f}, b:{b:.4f}")

# test the network
print(predict(1, 20000))  # 200 + 50 + 500 = 750
print(predict(1, 50000))  # 200 + 50 + 1250 = 1500
print(predict(5, 10000))  # 200 + 250 + 250 = 700
