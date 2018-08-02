import tensorflow.python.platform
import math
import numpy as np
import random
import sys
import tensorflow as tf


# Global variables.
BATCH_SIZE = 100  # The number of training examples to use per training step.

tf.app.flags.DEFINE_string(
        'english',
        None,
        'English language vectors (test and train)',
        )
tf.app.flags.DEFINE_string(
        'foreign',
        None,
        'foreign language vectors (test and train)',
        )
tf.app.flags.DEFINE_string(
        'dict',
        None,
        'english--foreign word pairs',
        )
tf.app.flags.DEFINE_string(
        'to_project',
        None,
        'English vectors to project into foreign space',
        )
tf.app.flags.DEFINE_integer(
        'num_epochs',
        100,
        'Number of passes over the training data.',
        )
tf.app.flags.DEFINE_integer(
        'num_hidden',
        10000,
        'Number of nodes in the hidden layer.',
        )
tf.app.flags.DEFINE_boolean(
        'verbose',
        False,
        'Produce verbose output.',
        )
tf.app.flags.DEFINE_string(
        'gentestfile',
        None,
        'where to save test set word pairs',
        )
FLAGS = tf.app.flags.FLAGS

def readVectors(filename):
    vectors = {}
    for line in open(filename):
        row = line.strip().split()
        word = row.pop(0)
        vectors[word] = [float(i) for i in row]
    return vectors


def readVectorsInOrder(filename):
    data = [[float(i) for i in r.split()[1:]] for r in open(filename)]
    if data and len(data[0]) == 1:  # header
        data.pop(0)
    return data


def readWordPairs(dictfile):
    return [line.split()[:2] for line in open(dictfile)]


def getVectorPairs(english, foreign, pairs):
    return [
        (english[e], foreign[f])
        for [e,f] in pairs
        if e in english and f in foreign
    ]


def splitTrainAndTest(corpus):
    testSize = 0.1
    testLen = int(len(corpus) * testSize)
    random.shuffle(corpus)
    train = corpus[testLen:]
    test = corpus[:testLen]
    return train, test


def getDataVectors(english, foreign, train, test):
    trainVecs = getVectorPairs(english, foreign, train)
    testVecs = getVectorPairs(english, foreign, test)
    return (
            [f for (e,f) in trainVecs],
            [e for (e,f) in trainVecs],
            [f for (e,f) in testVecs],
            [e for (e,f) in testVecs],
            )


def writeTestSet(pairs):
    if FLAGS.gentestfile is not None:
        with open(FLAGS.gentestfile, 'w') as testfile:
            for (e,f) in pairs:
                testfile.write('{}\t{}\n'.format(f, e))


    # Extract numpy representations of the labels and features given rows consisting of:
#   label, feat_0, feat_1, ..., feat_n
def extract_data():
    en = readVectors(FLAGS.english)
    fr = readVectors(FLAGS.foreign)
    pairs = readWordPairs(FLAGS.dict)
    trainpairs, testpairs = splitTrainAndTest(pairs)
    writeTestSet(testpairs)
    labels, fvecs, testlabels, testfvecs = getDataVectors(
        en,
        fr,
        trainpairs,
        testpairs,
    )
    project = readVectorsInOrder(FLAGS.to_project)

    # Convert the array of float arrays into a numpy float matrix.
    project_np = np.matrix(project).astype(np.float32)

    fvecs_np = np.matrix(fvecs).astype(np.float32)
    labels_np = np.matrix(labels).astype(np.float32)
    labels_o = labels_np.transpose()

    testfvecs_np = np.matrix(testfvecs).astype(np.float32)
    testlabels_np = np.matrix(testlabels).astype(np.float32)
    testlabels_o = testlabels_np.transpose()

    # Return a pair of the feature matrix and the one-hot label matrix.
    return fvecs_np,project_np,labels_o,testfvecs_np,testlabels_o

# Init weights method. (Lifted from Delip Rao: http://deliprao.com/archives/100)
def init_weights(shape, init_method='xavier', xavier_params = (None, None)):
    if init_method == 'zeros':
        return tf.Variable(tf.zeros(shape, dtype=tf.float32))
    elif init_method == 'uniform':
        return tf.Variable(tf.random_normal(shape, stddev=0.01, dtype=tf.float32))
    else: #xavier
        (fan_in, fan_out) = xavier_params
        low = -4*np.sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1} 
        high = 4*np.sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32))

def main(argv=None):
    # Be verbose?
    verbose = FLAGS.verbose

    # Extract it into numpy arrays.
    train_data,project_data,train_labels,test_data,test_labels = extract_data()

    # Get the shape of the training data.
    train_size,num_features_in = train_data.shape
    num_features_out,train_size = train_labels.shape
    project_size,num_features_p = project_data.shape

    # Get the number of epochs for training.
    num_epochs = FLAGS.num_epochs

    # Get the size of layer one.
    num_hidden = FLAGS.num_hidden

    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    x = tf.placeholder("float", shape=[None, num_features_in])
    y_ = tf.placeholder("float", shape=[None, num_features_out])

    # Define and initialize the network.

    num_hidden_1 = num_features_in*10
    w_hidden_1 = init_weights([num_features_in, num_hidden_1],'xavier',xavier_params=(num_features_in, num_hidden_1))
    b_hidden_1 = init_weights([1,num_hidden_1],'zeros')
    hidden_1 = tf.nn.tanh(tf.matmul(x,w_hidden_1) + b_hidden_1)

    num_hidden_2 = num_features_in*10
    w_hidden_2 = init_weights([num_hidden_1, num_hidden_2],'xavier',xavier_params=(num_hidden_1, num_hidden_2))
    b_hidden_2 = init_weights([1,num_hidden_2],'zeros')
    hidden_2 = tf.nn.tanh(tf.matmul(hidden_1,w_hidden_2) + b_hidden_2)

    num_hidden_3 = num_features_in*10
    w_hidden_3 = init_weights([num_hidden_2, num_hidden_3],'xavier',xavier_params=(num_hidden_2, num_hidden_3))
    b_hidden_3 = init_weights([1,num_hidden_3],'zeros')
    hidden_3 = tf.nn.tanh(tf.matmul(hidden_2,w_hidden_3) + b_hidden_3)

    w_hidden_4 = init_weights([num_hidden_3, num_hidden],'xavier',xavier_params=(num_hidden_3, num_hidden))
    b_hidden_4 = init_weights([1,num_hidden],'zeros')
    hidden_4 = tf.nn.tanh(tf.matmul(hidden_3,w_hidden_4) + b_hidden_4)

    # Initialize the output weights and biases.
    w_out = init_weights(
            [num_hidden,num_features_out],
            'xavier',
            xavier_params=(num_hidden,num_features_out))    
    b_out = init_weights([1,num_features_out],'zeros')
    w_out = w_out / float(math.sqrt(num_hidden))
    b_out = b_out / float(math.sqrt(num_hidden))

    # The output layer.
    y = tf.matmul(hidden_4,w_out)+b_out

    # Optimization.
    myloss = tf.reduce_mean(tf.square(y-y_))
    train_step = tf.train.AdamOptimizer(1e-5).minimize(myloss)

    # Evaluation.
    predicted_class = y
    correct_prediction = y_
    accuracy = tf.reduce_mean(tf.square(y-y_))
    sumaccuracy = tf.reduce_sum(tf.square(y-y_))

    # Create a local session to run this computation.
    with tf.Session() as s:
        # Run all the initializers to prepare the trainable parameters.
        tf.initialize_all_variables().run()
        if verbose:
            sys.stderr.write('Initialized!\n\nTraining.\n')

        # Iterate and train.
        lossprev = 0
        idx = 0
        for step in range(num_epochs * train_size // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % train_size
            batch_data = train_data[offset:(offset + BATCH_SIZE), :]
            batch_labels = train_labels[:, offset:(offset + BATCH_SIZE)]
            train_step.run(feed_dict={x: batch_data, y_: batch_labels.transpose()})
            if verbose and offset >= train_size-BATCH_SIZE:
                idx = idx + 1
                losscurr = accuracy.eval(feed_dict={x: test_data, y_: test_labels.transpose()})
                sys.stderr.write('{}\t{}\n'.format(idx, losscurr))
                if lossprev == 0:
                    lossprev = losscurr
                if lossprev > losscurr:
                    lossprev = losscurr
                if lossprev < losscurr:
                    diff = losscurr - lossprev
                    lossprev = losscurr
                    if diff < 0.0001:
                        break
        offset = 0
        batch_data = project_data[offset:(offset + BATCH_SIZE), :]
        y_out_np = predicted_class.eval(feed_dict={x: batch_data})
        sumacc = sumaccuracy.eval(feed_dict={x: batch_data, y_:batch_data})
        if project_size > BATCH_SIZE:
            totalstep = project_size // BATCH_SIZE
            for step in range(project_size // BATCH_SIZE):
                step = step + 1;
                sys.stderr.write('{}\n{}\n'.format(step, totalstep))
                offset = (step * BATCH_SIZE) % project_size
                batch_data = project_data[offset:(offset + BATCH_SIZE), :]
                y_out=predicted_class.eval(feed_dict={x: batch_data})
                y_out_np = np.concatenate((y_out_np,y_out), axis = 0)
                sumacc = sumacc + sumaccuracy.eval(feed_dict={x: batch_data, y_:batch_data})
                if offset >= project_size-BATCH_SIZE:
                    break
        sumacc = sumacc / project_size
        np.savetxt(sys.stdout ,y_out_np,delimiter=" ")


if __name__ == '__main__':
    tf.app.run()
