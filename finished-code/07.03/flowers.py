import flowersdata as data
import math
import random

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
input_count, hidden_count, output_count = 2, 8, 3

w_i_h = [[random.random() - 0.5 for _ in range(input_count)] for _ in range(hidden_count)]
w_h_o = [[random.random() - 0.5 for _ in range(hidden_count)] for _ in range(output_count)]
b_i_h = [0 for _ in range(hidden_count)]
b_h_o = [0 for _ in range(output_count)]

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
    for y in range(output_count):
        for x in range(hidden_count):
            w_h_o[y][x] -= learning_rate * w_h_o_d_T[y][x] / len(data.inputs)
        b_h_o[y] -= learning_rate * b_h_o_d[y] / len(data.inputs)

    w_i_h_d_T = list(zip(*w_i_h_d))
    for y in range(hidden_count):
        for x in range(input_count):
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
for a, t in zip(act_o, data.test_targets):
    if a.index(max(a)) == t.index(max(t)):
        correct += 1
print(f"Correct: {correct}/{len(act_o)} ({correct / len(act_o):%})")
