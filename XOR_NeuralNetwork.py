import tensorflow as tf

inputStream = tf.placeholder(tf.float32, shape=[4,2])
inputWages = tf.Variable(tf.random_uniform([2,2], -1, 1))
inputBiases = tf.Variable(tf.zeros([2]))

outputStream = tf.placeholder(tf.float32, shape=[4,1])
outputWages = tf.Variable(tf.random_uniform([2,1], -1, 1))
outputBiases = tf.Variable(tf.zeros([1]))

inputTrainingData = [[0,0],[0,1],[1,0],[1,1]]
outputTrainingData = [[0],[1],[1],[0]]

hiddenNeuronsFormula = tf.sigmoid(tf.matmul(inputStream, inputWages) + inputBiases)
outputNeuronFormula = tf.sigmoid(tf.matmul(hiddenNeuronsFormula, outputWages) + outputBiases)
cost = tf.reduce_mean(( (outputStream * tf.log(outputNeuronFormula)) +
	((1 - outputStream) * tf.log(1.0 - outputNeuronFormula)) ) * -1)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(10000):
	sess.run(train_step, feed_dict={inputStream: inputTrainingData, outputStream: outputTrainingData})

print('Input wages: ', sess.run(inputWages))
print('Input biases: ', sess.run(inputBiases))
print('Output wages: ', sess.run(outputWages))
print('Output biases: ', sess.run(outputBiases))
