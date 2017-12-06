import tensorflow as tf
import tensorflow.contrib as tc
## MAIN CHANGES
from baselines.common import tf_util as U

def generate_dropout_layer(apply_layer, prev_dropout_layer, keep_prob):
    new_networks = []
    for dropout_network in prev_dropout_layer:
        dropout_network = apply_layer(dropout_network)
        dropout_network, mask = U.bayes_dropout(dropout_network, keep_prob)
        new_networks.append(dropout_network)
    return new_networks


def apply_to_layer(apply_layer, prev_dropout_layer):
    new_networks = []
    for dropout_network in prev_dropout_layer:
        dropout_network = apply_layer(dropout_network)
        new_networks.append(dropout_network)
    return new_networks


class Model(object):
    def __init__(self, name):
        self.name = name

    @property
    def vars(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)

    @property
    def trainable_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    @property
    def perturbable_vars(self):
        return [var for var in self.trainable_vars if 'LayerNorm' not in var.name]


class Actor(Model):
    def __init__(self, nb_actions, name='actor', layer_norm=True):
        super(Actor, self).__init__(name=name)
        self.nb_actions = nb_actions
        self.layer_norm = layer_norm

    def __call__(self, obs, reuse=False):
        with tf.variable_scope(self.name) as scope:
            if reuse:
                scope.reuse_variables()

            x = obs
            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, 64)
            if self.layer_norm:
                x = tc.layers.layer_norm(x, center=True, scale=True)
            x = tf.nn.relu(x)

            x = tf.layers.dense(x, self.nb_actions, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            x = tf.nn.tanh(x)
        return x


class Critic(Model):
    def __init__(self, hidden_layers = (64,64), merge_layer=1, name='critic', layer_norm=False,
                 mc_samples=50, keep_prob=0.95,dropout_on_v=None):
        super(Critic, self).__init__(name=name)
        self.layer_norm = layer_norm
        self.hidden_sizes = hidden_layers
        self.mc_samples = mc_samples
        self.keep_prob = keep_prob
        self.merge_layer = merge_layer
        self.dropout_on_v=dropout_on_v

    def __call__(self, obs, action, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse) as scope:
            x = obs

            dropout_networks = [x] * self.mc_samples
            # dropout_networks = generate_dropout_layer(lambda x: x, dropout_networks, self.V_keep_prob)

            for i, hid_size in enumerate(self.hidden_sizes):
                if  i == self.merge_layer:
                    x = tf.concat([x, action], axis=-1)

                    apply_layer = lambda y : tf.concat([y,action], axis=-1)
                    dropout_networks = apply_to_layer(apply_layer, dropout_networks)


                if self.layer_norm:
                    x = tf.nn.relu(
                                    tc.layers.layer_norm(
                                            tf.layers.dense(x, hid_size, name="critic_layer%i"%(i+1)),
                                        center=True,
                                        scope="critic_layer_norm%i"%(i+1),
                                        scale=True)
                                        )

                    apply_layer = lambda y : tf.nn.relu(
                                                tc.layers.layer_norm(
                                                    tf.layers.dense(y, hid_size, name="critic_layer%i"%(i+1), reuse=True),
                                                    center=True,
                                                    scope="critic_layer_norm%i"%(i+1),
                                                    scale=True,
                                                    reuse=True)
                                                    )
                else:
                    x = tf.nn.relu(
                                    tf.layers.dense(x, hid_size, name="critic_layer%i"%(i+1))
                                )

                    apply_layer = lambda y : tf.nn.relu(
                                                tf.layers.dense(y, hid_size,name="critic_layer%i"%(i+1), reuse=True)
                                                )

                dropout_networks=generate_dropout_layer(apply_layer,dropout_networks,self.keep_prob)

            ## final layer
            x = tf.layers.dense(x, 1, name="critic_output", activation=None, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))

            apply_layer = lambda y : tf.layers.dense(y, 1, activation=None, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3), name="critic_output", reuse=True)

            dropout_networks=apply_to_layer(apply_layer,dropout_networks)


        if self.dropout_on_v is not None:
            return x, tf.add_n(dropout_networks) / float(len(dropout_networks)), dropout_networks
        else:
            return x


    @property
    def output_vars(self):
        output_vars = [var for var in self.trainable_vars if 'output' in var.name]
        return output_vars
