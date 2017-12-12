import tensorflow as tf
import ExpBuff
import Tic
import numpy as np

class RL():
    def __init__(self, struct):
        self.struct = struct
        self.build(struct)

    def build(self,struct):
        self.input = tf.placeholder(shape=[None,struct[0]],dtype=tf.float32)
        out = self.input

        for i in range(1,len(struct)):
            init = tf.contrib.layers.xavier_initializer()
            w = tf.Variable(init([struct[i-1],struct[i]]))
            b = tf.Variable(init([struct[i]]))
            out = tf.add(tf.matmul(out,w),b)
            if i < len(struct) - 1:
                out = tf.nn.leaky_relu(out)

        self.out = out

        self.softmax = tf.nn.softmax(self.out)

        mask = tf.mod(tf.add(1.0, tf.abs(self.input)), 2)

        self.masked = tf.multiply(self.softmax,mask)

        self.predict = tf.argmax(self.masked,1)

        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.onehot = tf.one_hot(self.actions,struct[-1],dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.out, self.onehot), axis=1)

        self.error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update = self.trainer.minimize(self.loss)

def updateTargetGraph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars // 2]):
        op_holder.append(tfVars[idx + total_vars // 2].assign(
            (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
    return op_holder

def updateTarget(op_holder, sess):
    for op in op_holder:
        sess.run(op)

def train(load,nn):
    numEps = 20000

    size = int(nn.struct[-1] ** (1/2))

    path = "./nns"

    batchSize = 32
    updateFreq = 1

    y = 0.99
    startE = 1.0
    finalE = 0.1

    dE = (finalE - startE) / (numEps * 0.75)

    e = startE

    tau = 0.001

    target = RL(nn.struct)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=1)

    trainables = tf.trainable_variables()
    targetOps = updateTargetGraph(trainables,tau)

    myBuffer = ExpBuff.expBuff()

    totSteps = 0

    with tf.Session() as sess:
        sess.run(init)
        if load == True:
            print("Loading Model")
            ckpt = tf.train.get_checkpoint_state(path)
            saver.restore(sess,ckpt.model_checkpoint_path)
        for i in range(numEps):
            avgLoss = 0.0
            counter = 0
            epBuff = ExpBuff.expBuff()
            board = Tic.Tic(size)
            done = False

            j = 0

            while not done:
                actor = 1

                if j % 2 == 1:
                    actor = -1

                s = board.state(actor)

                a = -1
                loc = [0,0]
                if np.random.rand(1) < e:
                    isValid = False
                    while not isValid:
                        a = np.random.randint(0,size**2)
                        loc[0] = int(a / size)
                        loc[1] = a % size
                        if board.valid(loc):
                            isValid = True
                else:
                    a = sess.run(nn.predict, feed_dict={nn.input: s})[0]
                    loc[0] = int(a/size)
                    loc[1] = a % size

                board.play(actor,loc)
                output = board.done()
                done = output[0]
                r = actor * output[1]
                s1 = board.state(actor * -1)

                j += 1
                totSteps += 1

                epBuff.add(np.reshape(np.array([s,a,r,s1,done]),[1,5]))

                e = max(e+dE,finalE)

                if len(myBuffer.buffer) > batchSize:
                    if totSteps % updateFreq == 0:
                        trainBatch = myBuffer.sample(batchSize)
                        Q1 = sess.run(nn.predict,feed_dict={nn.input: np.vstack(trainBatch[:,3])})
                        Q2 = sess.run(target.out,feed_dict={target.input: np.vstack(trainBatch[:,3])})

                        endMult = -(trainBatch[:,4] - 1)
                        doubleQ = -1 * Q2[range(batchSize),Q1]
                        targetQ = trainBatch[:,2] + (y*doubleQ*endMult)

                        _,loss = sess.run([nn.update,nn.loss], feed_dict={nn.input: np.vstack(trainBatch[:,0]), nn.targetQ:targetQ, nn.actions:trainBatch[:,1]})

                        counter += 1
                        avgLoss += loss

                        updateTarget(targetOps,sess)

            if i % 100 == 0 and len(myBuffer.buffer) > batchSize:
                if counter > 0:
                    print(str(i) + ", " + str(avgLoss / counter))
                else:
                    print(str(i) + ", 0.0")

            if i % 1000 == 0:
                saver.save(sess,path+"/model"+str(size)+"-"+str(i)+".ckpt")

            myBuffer.add(epBuff.buffer)

        saver.save(sess, path + "/model" + str(size) + "-" + str(i) + ".ckpt")