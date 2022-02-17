import flowersdata as data
import math

def softmax(predictions):
    m = max(predictions)
    temp = [math.exp(p - m) for p in predictions]
    total = sum(temp)
    return [t / total for t in temp]

def log_loss(activations, targets):
    losses = [-t * math.log(a) - (1 - t) * math.log(1 - a) for a, t in zip(activations, targets)]
    return sum(losses)

epochs = 5000
learning_rate = .3

w_i_h = [[0.1, -0.2], [-0.3, 0.25], [0.12, 0.23], [-0.11, -0.22]]  # 4 hidden neurons
w_h_o = [[0.2, 0.17, 0.3, -0.11], [0.3, -0.4, 0.5, -0.22], [0.12, 0.23, 0.15, 0.33]]
b_i_h = [0.2, 0.34, 0.21, 0.44]  # 4 hidden neurons
b_h_o = [0.3, 0.29, 0.37]  # 3 output neurons

for epoch in range(epochs):
    pred_h = [[sum([w * a for w, a in zip(weights, inp)]) +
        bias for weights, bias in zip(w_i_h, b_i_h)] for inp in data.inputs]
    act_h = [[max(0, p) for p in pred] for pred in pred_h]  # apply ReLU
    pred_o = [[sum([w * a for w, a in zip(weights, inp)]) +
        bias for weights, bias in zip(w_h_o, b_h_o)] for inp in act_h]
    act_o = [softmax(predictions) for predictions in pred_o]

    cost = sum([log_loss(a, t) for a, t in zip(act_o, data.targets)]) / len(act_o)
    print(f"epoch:{epoch} cost:{cost:.4f}")

    # Error derivatives
    errors_d_o = [[a - t for a, t in zip(ac, ta)] for ac, ta in zip(act_o, data.targets)]
    w_h_o_T = list(zip(*w_h_o))
    errors_d_h = [[sum([d * w for d, w in zip(deltas, weights)]) * (0 if p <= 0 else 1)
        for weights, p in zip(w_h_o_T, pred)] for deltas, pred in zip(errors_d_o, pred_h)]

    # Gradient hidden->output
    act_h_T = list(zip(*act_h))
    errors_d_o_T = list(zip(*errors_d_o))
    w_h_o_d = [[sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_o_T]
        for act in act_h_T]
    b_h_o_d = [sum([d for d in deltas]) for deltas in errors_d_o_T]

    # Gradient input->hidden
    inputs_T = list(zip(*data.inputs))
    errors_d_h_T = list(zip(*errors_d_h))
    w_i_h_d = [[sum([d * a for d, a in zip(deltas, act)]) for deltas in errors_d_h_T]
        for act in inputs_T]
    b_i_h_d = [sum([d for d in deltas]) for deltas in errors_d_h_T]

    # Update weights and biases for all layers
    w_h_o_d_T = list(zip(*w_h_o_d))
    for y in range(len(w_h_o_d_T)):
        for x in range(len(w_h_o_d_T[0])):
            w_h_o[y][x] -= learning_rate * w_h_o_d_T[y][x] / len(data.inputs)
        b_h_o[y] -= learning_rate * b_h_o_d[y] / len(data.inputs)

    w_i_h_d_T = list(zip(*w_i_h_d))
    for y in range(len(w_i_h_d_T)):
        for x in range(len(w_i_h_d_T[0])):
            w_i_h[y][x] -= learning_rate * w_i_h_d_T[y][x] / len(data.inputs)
        b_i_h[y] -= learning_rate * b_i_h_d[y] / len(data.inputs)

# test the network
pred_h = [[sum([w * a for w, a in zip(weights, inp)]) +
    bias for weights, bias in zip(w_i_h, b_i_h)] for inp in data.test_inputs]
act_h = [[max(0, p) for p in pre] for pre in pred_h]
pred_o = [[sum([w * a for w, a in zip(weights, inp)]) + bias
    for weights, bias in zip(w_h_o, b_h_o)] for inp in act_h]
act_o = [softmax(predictions) for predictions in pred_o]
correct = 0
counter = 1
for a, t in zip(act_o, data.test_targets):
    if a.index(max(a)) == t.index(max(t)):
        correct += 1
    else:
        print(counter)
        print(a)
        print(t)
    counter += 1
print(f"Correct: {correct}/{len(act_o)} ({correct / len(act_o):%})")
