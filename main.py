import tensorflow as tf
import numpy as np
import Tic
import RL
import ExpBuff

size = 3
squ = size**2

path = "./nns"

def play():
    board = Tic.Tic(size)

    nn = RL.RL([squ,10*squ,10*squ,10*squ,squ])

    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    done = False

    i = 0

    ai = 1 #0 is x, 1 is o

    agent = -1 * (ai * 2 - 1)

    winner = 0

    while not done:
        loc = [-1,-1]
        if i % 2 == ai:
            a,m = sess.run([nn.predict,nn.out], feed_dict={nn.input: board.state(agent)})
            a = a[0]
            print(m)
            loc[0] = int(a / size)
            loc[1] = a % size
            board.play(agent, loc)
        else:
            board.print()
            text = input("Please enter play position 'row,column': ")
            loc = text.split(',')
            loc[0] = int(loc[0])
            loc[1] = int(loc[1])
            board.play(-1 * agent, loc)

        output = board.done()
        done = output[0]
        winner = output[1]

        i += 1

    board.print()

    if winner == 0:
        print("Tie!")
    elif winner == agent:
        print("Computer Wins")
    else:
        print("Human Wins")

def train():
    nn = RL.RL([squ,10*squ,10*squ,10*squ,squ])
    RL.train(False,nn)


#play()
train()
