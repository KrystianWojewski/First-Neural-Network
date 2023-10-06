from SimpleNN import simpleNN

INPUT_SIZE = 2
HIDDEN_SIZE = 3
OUTPUT_SIZE = 1
EPOCHS = 1000
LEARNING_RATE = 0.01

p = simpleNN(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE, epochs=EPOCHS, learning_rate=LEARNING_RATE)

# Get the data from a CSV file
X, Y = p.read_input_data('train.csv')

# Train
p.train(X, Y)

p.save_model()
