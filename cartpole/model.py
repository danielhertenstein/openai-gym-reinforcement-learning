import gym
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def update_scene(num, frames, patch):
    patch.set_data(frames[num])
    return patch,


def plot_animation(frames, repeat=False, interval=40):
    plt.close()
    fig = plt.figure()
    patch = plt.imshow(frames[0])
    plt.axis('off')
    return animation.FuncAnimation(
        fig,
        update_scene,
        fargs=(frames, patch),
        frames=len(frames),
        repeat=repeat,
        interval=interval
    )


def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards


def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]


class Model:
    def __init__(self, learning_rate=0.01, checkpoint_path="./cartpole.ckpt"):
        env = gym.make("CartPole-v0")
        self.n_inputs = env.observation_space.shape[0]  # 4
        n_hidden = 4
        n_outputs = 1  # outputs the probability of accelerating left
        initializer = tf.keras.initializers.he_normal()

        self.X = tf.placeholder(tf.float32, shape=[None, self.n_inputs])
        hidden = tf.layers.dense(self.X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
        logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer)
        outputs = tf.nn.sigmoid(logits)

        p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
        self.action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)

        y = 1. - tf.to_float(self.action)  # Assume the chosen action is the best possible action

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(cross_entropy)

        self.gradients = [grad for grad, variable in grads_and_vars]

        self.gradient_placeholders = []
        grads_and_vars_feed = []
        for grad, variable in grads_and_vars:
            gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
            self.gradient_placeholders.append(gradient_placeholder)
            grads_and_vars_feed.append((gradient_placeholder, variable))

        self.training_op = optimizer.apply_gradients(grads_and_vars_feed)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        self.checkpoint_path = checkpoint_path

    def train(self):
        n_iterations = 250  # number of training iterations
        n_games_per_update = 10  # train the policy every 10 episodes
        save_iterations = 10  # save the model every 10 training iterations
        discount_rate = 0.95

        env = gym.make("CartPole-v0")

        with tf.Session() as sess:
            self.init.run()
            for iteration in range(n_iterations):
                all_rewards = []  # all sequences of raw rewards for each episode
                all_gradients = []  # gradients saved at each step of each episode
                for game in range(n_games_per_update):
                    current_rewards = []  # all raw rewards from the current episode
                    current_gradients = []  # all gradients from the current episode
                    obs = env.reset()
                    for step in range(env._max_episode_steps):
                        feed_dict = {self.X: obs.reshape(1, self.n_inputs)}
                        action_val, gradients_val = sess.run([self.action, self.gradients], feed_dict=feed_dict)
                        obs, reward, done, info = env.step(action_val[0][0])
                        current_rewards.append(reward)
                        current_gradients.append(gradients_val)
                        if done:
                            break
                    all_rewards.append(current_rewards)
                    all_gradients.append(current_gradients)

                # At this point we have run the policy for 10 episodes, and we are
                # ready for a policy update using the algorithm described earlier.
                all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
                feed_dict = {}
                for var_index, grad_placeholder in enumerate(self.gradient_placeholders):
                    # multiply the gradients by the action scores, and compute the mean
                    mean_gradients = np.mean(
                        [
                            reward * all_gradients[game_index][step][var_index]
                            for game_index, rewards in enumerate(all_rewards)
                            for step, reward in enumerate(rewards)
                        ],
                        axis=0
                    )
                    feed_dict[grad_placeholder] = mean_gradients

                sess.run(self.training_op, feed_dict=feed_dict)
                if iteration % save_iterations == 0:
                    self.saver.save(sess, self.checkpoint_path)

    def render(self):
        env = gym.make("CartPole-v0")
        frames = []
        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)
            obs = env.reset()
            for step in range(env._max_episode_steps):
                img = env.render(mode="rgb_array")
                frames.append(img)
                action_val = self.action.eval(feed_dict={self.X: obs.reshape(1, self.n_inputs)})
                obs, reward, done, info = env.step(action_val[0][0])
                if done:
                    break
            env.close()
        plot_animation(frames)
        plt.show()

    def evaluate(self):
        env = gym.make("CartPole-v0")
        totals = []
        with tf.Session() as sess:
            self.saver.restore(sess, self.checkpoint_path)
            for episode in range(100):  # 100 episodes is the evaluation length
                episode_rewards = 0
                obs = env.reset()
                for step in range(env._max_episode_steps):
                    action_val = self.action.eval(feed_dict={self.X: obs.reshape(1, self.n_inputs)})
                    obs, reward, done, info = env.step(action_val[0][0])
                    episode_rewards += reward
                    if done:
                        break
                totals.append(episode_rewards)
        env.close()
        print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
