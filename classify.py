from SimpleNN import simpleNN

p = simpleNN()

p.load_model('sum_model.model')

Xtest, Yexpected = p.read_input_data('test.csv')
Yout = p.test(Xtest)

print()
print('Test results:')

for i in range(len(Yout)):
    print('{:.3f} + {:.3f} = {:.3f} (expected {:.3f})'.format(Xtest[i][0], Xtest[i][1], Yout[i][0], Yexpected[i][0]))

