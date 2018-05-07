from collections import deque
import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


env = gym.make("MsPacman-v0")
input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8, 8), (4, 4), (3, 3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10  # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n  # 9 discrete actions are available
initializer = tf.keras.initializers.he_normal()
replay_memory_size = 500000
replay_memory = deque([], maxlen=replay_memory_size)
greed_min = 0.1
greed_max = 1.0
greed_decay_steps = 2000000
MSPACMAN_COLOR = np.array([210, 164, 74]).mean()
training_steps = 4000000  # total number of training steps


def q_network(X_state, name):
    prev_layer = X_state

    with tf.variable_scope(name) as scope:
        conv_iter = zip(conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation)
        for n_maps, kernel_size, strides, padding, activation in conv_iter:
            prev_layer = tf.layers.conv2d(
                prev_layer,
                filters=n_maps,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                activation=activation,
                kernel_initializer=initializer,
            )
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])

        hidden = tf.layers.dense(
            last_conv_layer_flat,
            n_hidden,
            activation=hidden_activation,
            kernel_initializer=initializer
        )

        outputs = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)

    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}
    return outputs, trainable_vars_by_name


def sample_memories(batch_size):
    indices = np.random.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)


def greedy(q_values, step):
    greed = max(greed_min, greed_max - (greed_max - greed_min) * step / greed_decay_steps)
    if np.random.rand() < greed:
        return np.random.randint(n_outputs)  # random_action
    else:
        return np.argmax(q_values)  # "optimal" action


def preprocess_observation(obs):
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.mean(axis=2)  # to grayscale
    img[img == MSPACMAN_COLOR] = 0  # improve contrast
    img = (img - 128) / 128 - 1  # normalize from -1 to 1
    return img.reshape(input_height, input_width, input_channels)


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=40):
    plt.close()  # or else nbagg sometimes plots in the pervious cell
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch),
                                   frames=len(frames), repeat=repeat, interval=interval)


class MspacmanDQN:
    def __init__(self, learning_rate=0.001, momentum=0.95, checkpoint_path="./mspacman.ckpt"):
        self.X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])
        self.y = tf.placeholder(tf.float32, shape=[None, 1])

        self.online_q_values, online_vars = q_network(self.X_state, name="q_networks/online")
        self.target_q_values, target_vars = q_network(self.X_state, name="q_networks/target")
        copy_ops = [target_var.assign(online_vars[var_name]) for var_name, target_var in target_vars.items()]
        self.copy_online_to_target = tf.group(*copy_ops)

        self.X_action = tf.placeholder(tf.int32, shape=[None])
        q_value = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, n_outputs), axis=1, keepdims=True)

        error = tf.abs(self.y - q_value)
        clipped_error = tf.clip_by_value(error, 0.0, 1.0)
        linear_error = 2 * (error - clipped_error)
        self.loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
        self.training_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.checkpoint_path = checkpoint_path

    def train(self):
        training_start = 10000  # start training after 10,000 game iterations
        training_interval = 4  # run a training step every 4 game iterations
        save_steps = 1000  # save the model every 1,000 training steps
        copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
        discount_rate = 0.99
        skip_start = 90  # Skip the start of every game (it's just waiting time)
        batch_size = 50
        iteration = 0  # game iterations
        done = True  # env needs to be reset
        loss_val = np.infty
        game_length = 0
        total_max_q = 0
        mean_max_q = 0.0

        env = gym.make("MsPacman-v0")

        with tf.Session() as sess:
            if os.path.isfile(self.checkpoint_path + ".index"):
                self.saver.restore(sess, self.checkpoint_path)
            else:
                self.init.run()
                self.copy_online_to_target.run()
            while True:
                step = self.global_step.eval()
                if step >= training_steps:
                    break
                iteration += 1
                print("\rIteration {}\tTraining step {}/{} ({:.1f})%\tLoss {:5f}\tMean Max-Q {:5f}   "
                      .format(iteration, step, training_steps, step * 100 / training_steps, loss_val, mean_max_q),
                      end="")
                if done:  # game over, start again
                    obs = env.reset()
                    for skip in range(skip_start):  # skip the start of the game
                        obs, reward, done, info = env.step(0)
                    state = preprocess_observation(obs)

                # Online DQN evaluates what to do
                q_values = self.online_q_values.eval(feed_dict={self.X_state: [state]})
                action = greedy(q_values, step)

                # Online DQN plays
                obs, reward, done, info = env.step(action)
                next_state = preprocess_observation(obs)

                # Memorize what happened
                replay_memory.append((state, action, reward, next_state, 1.0 - done))
                state = next_state

                # Compute statistics for tracking progress
                total_max_q += q_values.max()
                game_length += 1
                if done:
                    mean_max_q = total_max_q / game_length
                    total_max_q = 0.0
                    game_length = 0

                if iteration < training_start or iteration % training_interval != 0:
                    continue  # only train after warmup period and at regular intervals

                # Sample memories and use the target DQN to produce the target Q-values
                X_state_val, X_action_val, rewards, X_next_state_val, continues = (sample_memories(batch_size))
                next_q_values = self.target_q_values.eval(feed_dict={self.X_state: X_next_state_val})
                max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
                y_val = rewards + continues * discount_rate * max_next_q_values

                # Train the online DQN
                feed_dict = {
                    self.X_state: X_state_val,
                    self.X_action: X_action_val,
                    self.y: y_val
                }
                _, loss_val = sess.run([self.training_op, self.loss], feed_dict=feed_dict)

                # Regularly copy the online DQN to the target DQN
                if step % copy_steps == 0:
                    self.copy_online_to_target.run()

                # And save regularly
                if step % save_steps == 0:
                    self.saver.save(sess, self.checkpoint_path)

    def run(self, max_steps=10000):
        frames = []
        env = gym.make("MsPacman-v0")

        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)

            obs = env.reset()
            for step in range(max_steps):
                state = preprocess_observation(obs)

                # Online DQN evaluates what to do
                q_values = self.online_q_values.eval(feed_dict={self.X_state: [state]})
                action = np.argmax(q_values)

                # Online DQN plays
                obs, reward, done, info = env.step(action)

                img = env.render(mode="rgb_array")
                frames.append(img)

                if done:
                    break

            env.close()

        video = plot_animation(frames)
        plt.show()
