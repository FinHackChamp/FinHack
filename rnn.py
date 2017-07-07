import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, LSTM
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.optimizers import Adam
import tensorflow as tf
import time
import csv
from random import shuffle
import random
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt

user_name = 'David Mendoza'
step_size = 40
batch_size = 5

# (2975, 40, 150)
train_data = np.load('oneHotEncoded.npy')
sample = None


# flags
tf.flags.DEFINE_float("epsilon", 0.1, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("l2_lambda", 0.3, "Lambda for l2 loss.")
tf.flags.DEFINE_float("learning_rate", 0.1, "Learning rate")
tf.flags.DEFINE_float("max_grad_norm", 20.0, "Clip gradients to this norm.")
tf.flags.DEFINE_float("keep_prob", 0.6, "Keep probability for dropout")
tf.flags.DEFINE_integer("hidden_layer_num", 1, "The number of hidden layers (Integer)")
tf.flags.DEFINE_integer("hidden_size", 200, "The number of hidden nodes (Integer)")
tf.flags.DEFINE_integer("evaluation_interval", 5, "Evaluate and print results every x epochs")

tf.flags.DEFINE_integer("epochs", 30, "Number of epochs to train for.")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_string("train_data_path", 'data/0910_b_train.csv', "Path to the training dataset")
tf.flags.DEFINE_string("test_data_path", 'data/0910_b_test.csv', "Path to the testing dataset")



log_file_path = '1layered.txt'
hidden_state_path = 'hidden_stateb2.npy'
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

def add_gradient_noise(t, stddev=1e-3, name=None):
    """
    Adds gradient noise as described in http://arxiv.org/abs/1511.06807 [2].
    The input Tensor `t` should be a gradient.
    The output will be `t` + gaussian noise.
    0.001 was said to be a good fixed value for memory networks [2].
    """
    with tf.name_scope( name, "add_gradient_noise",[t, stddev]) as name:
        t = tf.convert_to_tensor(t, name="t")
        gn = tf.random_normal(tf.shape(t), stddev=stddev)
        return tf.add(t, gn, name=name)

class UserModel(object):

    def __init__(self, is_training, config, graph):
        self.state_size = config.state_size
        self._batch_size = batch_size = config.batch_size
        self.num_skills = num_skills = config.num_skills
        self.hidden_layer_num = len(self.state_size)
        self.hidden_size = size = FLAGS.hidden_size
        self.num_steps = num_steps = config.num_steps
        input_size = num_skills

        inputs = self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps-1, 150])
        # self._target_id = target_id = tf.placeholder(tf.int32, [1248])
        self._target_correctness = target_correctness = tf.placeholder(tf.float32, [batch_size*(num_steps-1), 150])
        final_hidden_size = self.state_size[-1]

        hidden_layers = []
        
        for i in range(self.hidden_layer_num):
            
            hidden1 = tf.contrib.rnn.BasicLSTMCell(self.state_size[i], state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
            if is_training and config.keep_prob < 1:
                hidden1 = tf.contrib.rnn.DropoutWrapper(hidden1, output_keep_prob=FLAGS.keep_prob)
            hidden_layers.append(hidden1)
        
        cell = tf.contrib.rnn.MultiRNNCell(hidden_layers, state_is_tuple=True)

        
        x = inputs
        
        print x
        outputs, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
        # print state
        #output = [batch_size * num_steps, final_hidden_size]
        output = tf.reshape(tf.concat(outputs,1), [-1, final_hidden_size])
       
        # calculate the logits from last hidden layer to output layer
        # graph.get_tensor_by_name("op_to_restore:0")
        sigmoid_w = tf.get_variable("sigmoid_w", [final_hidden_size, num_skills])
        sigmoid_b = tf.get_variable("sigmoid_b", [num_skills])
        if config.isTrain == False:
            sigmoid_w = graph.get_tensor_by_name("model/sigmoid_w:0")
            sigmoid_b = graph.get_tensor_by_name("model/sigmoid_b:0")
        logits = tf.matmul(output, sigmoid_w) + sigmoid_b
        
        self._logits = logits
        print logits
        softmaxed_logits = tf.nn.softmax(logits)
        
        #make prediction
        self._pred = self._pred_values = pred_values = softmaxed_logits
        self._pred_class = tf.argmax(softmaxed_logits, axis = 1)
        
        print self.pred
        # loss function
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels= target_correctness))

       
        self._final_state = state
        self._cost = cost = loss

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def input_data(self):
        return self._input_data

    @property
    def auc(self):
        return self._auc

    @property
    def pred(self):
        return self._pred

    @property
    def logits(self):
        return self._logits


    @property
    def target_id(self):
        return self._target_id

    @property
    def target_correctness(self):
        return self._target_correctness

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def pred_values(self):
        return self._pred_values

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property 
    def pred_class(self):
        return self._pred_class

class HyperParamsConfig(object):
    """Small config."""
    init_scale = 0.05
    num_steps = 0
    max_grad_norm = FLAGS.max_grad_norm
    max_max_epoch = FLAGS.epochs
    keep_prob = FLAGS.keep_prob
    num_skills = 0
    state_size = [200]
    batch_size = 32
    isTrain = True
def run_epoch(session, m, students, eval_op, verbose=False):
    """Runs the model on the given data."""
    start_time = time.time()
  
    index = 0
    pred_labels = []
    actual_labels = []
    pred_classes= []
    last_logit = None
    for i in range(int(students.shape[0] / m.batch_size)):
    	target_id = []
    	target_correctness = []
    	
    	x = students[i*m.batch_size: (i+1) * m.batch_size,:-1,:]
    	for b in range(m.batch_size):
        	for s in range(m.num_steps-1):
        		# get next time step
        		index = (list(students[b][s+1]).index(1))
        		# print index
        		target_id.append(b * m.num_steps + s * m.num_skills + index)
        		# print target_id
        		target_correctness.append(students[b][s+1])
        		actual_labels.append(students[b][s+1])
        pred, _, final_state, pred_class, last_logit  = session.run([m.pred, eval_op, m.final_state, m.pred_class, m.logits], feed_dict={
            m.input_data: x, m.target_correctness: target_correctness})
        #h: [batch_size, num_unit]

        h = final_state[0][1]
        for i in range(len(final_state)):
            if i == 0: continue
            h = np.concatenate((h,final_state[i][1]), axis=1)
        index += m.batch_size
        

        for p in pred:
            pred_labels.append(p)
        for p in pred_class:
            pred_classes.append(p)
   
    # print final_state[0][0].shape
    print np.array(pred_labels).shape
    print np.array(actual_labels).shape
    actual_classes = np.argmax(actual_labels, axis=1)
    # print actual_classes
    correct_prediction = [actual_classes[i] == pred_classes[i] for i in range(len(pred_classes))]
    # print correct_prediction
    accuracy = (sum(correct_prediction) + 0.0) / len(correct_prediction)


    rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
    # fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
    # auc = metrics.auc(fpr, tpr)
    last_logit = last_logit[-1, :]

    #calculate r^2
    r2 = r2_score(actual_labels, pred_labels)
    return rmse, accuracy, r2, final_state, last_logit



def train():
    

    config = HyperParamsConfig()
    config.isTrain = True
    config.batch_size = 32
    eval_config = HyperParamsConfig()
    timestamp = str(time.time())
    train_data_path = FLAGS.train_data_path
    #path to your test data set
    test_data_path = FLAGS.test_data_path
    #the file to store your test results
    result_file_path = "run_logs_{}".format(timestamp)

   

    train_max_num_problems, train_max_skill_num = (40, 150)
    train_students = np.load('oneHotEncoded.npy')
    config.num_steps = train_max_num_problems
    
    config.num_skills = train_max_skill_num
    # test_students, test_max_num_problems, test_max_skill_num = read_data_from_csv_file(test_data_path)
    # eval_config.num_steps = test_max_num_problems
    # eval_config.num_skills = test_max_skill_num
    
    with tf.Graph().as_default() as g:
        
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # decay learning rate
        starter_learning_rate = FLAGS.learning_rate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.96, staircase=True)

        

        with tf.Session(config=session_conf) as session:
            
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            
              
            
            # training model
            with tf.variable_scope("model", reuse=None, initializer=initializer):

                
                m = UserModel(is_training=True, config=config, graph=g)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)
            # # testing model
            # with tf.variable_scope("model", reuse=True, initializer=initializer):
            #     mtest = UserModel(is_training=False, config=eval_config)

            grads_and_vars = optimizer.compute_gradients(m.cost)
            grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                              for g, v in grads_and_vars if g is not None]
            grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
            train_op = optimizer.apply_gradients(grads_and_vars, name="train_op", global_step=global_step)
            session.run(tf.global_variables_initializer())
            # print tf.get_collection(tf.GraphKeys.VARIABLES, scope='model')
            # print "--------------------"
            # saver = tf.train.import_meta_graph('user_model-1000.meta')

            # saver.restore(session,tf.train.latest_checkpoint('./')) 
            # print tf.get_collection(tf.GraphKeys.VARIABLES, scope='model')
            # print(session.run('model/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases/Adam:0'))   
            # log hyperparameters to results file
            with open(result_file_path, "a+") as f:
                print("Writing hyperparameters into file")
                f.write("Hidden layer size: %d \n" % (FLAGS.hidden_size))
                f.write("Dropout rate: %.3f \n" % (FLAGS.keep_prob))
                f.write("Batch size: %d \n" % (config.batch_size))
                f.write("Max grad norm: %d \n" % (FLAGS.max_grad_norm))
            # saver = tf.train.Saver(tf.all_variables())

            cs = []
            hs = []
            for i in range(config.max_max_epoch):
                rmse, accuracy, r2, final_state, _ = run_epoch(session, m, train_students, train_op, verbose=True)
                print("Epoch: %d Train Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f \n" % (i + 1, rmse, accuracy, r2))
                with open(log_file_path, "a+") as f:
                    f.write("Epoch: %d Train Metrics:\n rmse: %.3f \t auc: %.3f \t r2: %.3f \n" % (i + 1, rmse, accuracy, r2))
                if((i+1) % FLAGS.evaluation_interval == 0):
                    print "Save variables to disk"
                    
            saver = tf.train.Saver()
            saver.save(session, 'user_model',global_step=1000)
def predict():
    
    config = HyperParamsConfig()
    config.isTrain = False
    config.batch_size = 1

    eval_config = HyperParamsConfig()
    timestamp = str(time.time())
    train_data_path = FLAGS.train_data_path
    #path to your test data set
    test_data_path = FLAGS.test_data_path
    #the file to store your test results
    result_file_path = "run_logs_{}".format(timestamp)
  
    

    train_max_num_problems, train_max_skill_num = (40, 150)
    train_students = np.array([np.load('oneHotEncoded.npy')[np.random.randint(2975),:,:]])
    # train_students = np.load('oneHotEncoded.npy')
    config.num_steps = train_max_num_problems
    
    config.num_skills = train_max_skill_num
    # test_students, test_max_num_problems, test_max_skill_num = read_data_from_csv_file(test_data_path)
    # eval_config.num_steps = test_max_num_problems
    # eval_config.num_skills = test_max_skill_num
    new_graph = tf.Graph()
  
        
   
    
    with new_graph.as_default():
        
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                  log_device_placement=FLAGS.log_device_placement)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        # decay learning rate
        starter_learning_rate = FLAGS.learning_rate
        with tf.Session(graph = new_graph) as session:
             
            initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
            
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 3000, 0.96, staircase=True)
            
            # training model
            with tf.variable_scope("model", reuse=None, initializer=initializer):   
                m = UserModel(is_training=True, config=config, graph = new_graph)
            
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=FLAGS.epsilon)
            # # testing model
            # with tf.variable_scope("model", reuse=True, initializer=initializer):
            #     mtest = UserModel(is_training=False, config=eval_config)

            grads_and_vars = optimizer.compute_gradients(m.cost)
            grads_and_vars = [(tf.clip_by_norm(g, FLAGS.max_grad_norm), v)
                              for g, v in grads_and_vars if g is not None]
            grads_and_vars = [(add_gradient_noise(g), v) for g, v in grads_and_vars]
            train_op = optimizer.apply_gradients(grads_and_vars, name="train_op1", global_step=global_step)
            session.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph('user_model-1000.meta')
            saver.restore(session,tf.train.latest_checkpoint('./'))
            # print tf.get_collection(tf.GraphKeys.VARIABLES, scope='model')
            # print "--------------------"
            
            # print tf.get_collection(tf.GraphKeys.VARIABLES, scope='model')
            # print(session.run('model/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases/Adam:0'))   
            # log hyperparameters to results file
            with open(result_file_path, "a+") as f:
                print("Writing hyperparameters into file")
                f.write("Hidden layer size: %d \n" % (FLAGS.hidden_size))
                f.write("Dropout rate: %.3f \n" % (FLAGS.keep_prob))
                f.write("Batch size: %d \n" % (config.batch_size))
                f.write("Max grad norm: %d \n" % (FLAGS.max_grad_norm))
            # saver = tf.train.Saver(tf.all_variables())

            cs = []
            hs = []
            
            rmse, accuracy, r2, final_state, last_logit = run_epoch(session, m, train_students, train_op, verbose=True)
            output = []
            output.append(np.argmax(last_logit))
            last_logit[output[-1]] = min(last_logit)
            output.append(np.argmax(last_logit))
            last_logit[output[-1]] = min(last_logit)
            output.append(np.argmax(last_logit))
            df = pd.read_csv('label.csv')
            names = list(df.name)
            output = [names[index] for index in output]
            return output


if __name__ == "__main__":
    # train()
    print predict()
    # train(train= False)
    


