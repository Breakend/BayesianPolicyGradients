from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import tensorflow.contrib as tc

import random

def generate_dropout_layer(apply_layer, prev_dropout_layer, keep_prob):
    new_networks = []
    for dropout_network in prev_dropout_layer:
        dropout_network = apply_layer(dropout_network)
        dropout_network, mask = U.bayes_dropout(dropout_network, keep_prob)
        new_networks.append(dropout_network)
    return new_networks


class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space,hid_size_V, hid_size_actor, num_hid_layers,V_keep_prob, pol_keep_prob,\
             mc_samples,layer_norm,activation_critic,activation_actor, dropout_on_V, dropout_on_policy,tau, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        self.dropout_on_policy = dropout_on_policy
#        self.pdtype = pdtype = make_pdtype(ac_space, dropout_on_policy)
        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))


        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz

        self.mc_samples = mc_samples
        self.pol_keep_prob = pol_keep_prob
        self.V_keep_prob=V_keep_prob

        ### MAIN CHANGES
        #######################
        # Value function

        with tf.variable_scope("value_function"):

            dropout_networks = [last_out] * self.mc_samples
#            dropout_networks = generate_dropout_layer(lambda x: x, dropout_networks, self.pol_keep_prob)

            for i in range(num_hid_layers):
                if layer_norm:
                    last_out = activation_critic(tc.layers.layer_norm(tf.layers.dense(last_out, hid_size_V, name="vffc%i"%(i+1), \
                    kernel_initializer=U.normc_initializer(1.0)), center=True,scope="vffc_activation%i"%(i+1) ,scale=True))

                    apply_layer = lambda x : activation_critic(tc.layers.layer_norm(tf.layers.dense(x, hid_size_V,name="vffc%i"%(i+1),
                                        reuse=True) ,center=True,scope="vffc_activation%i"%(i+1) ,scale=True,reuse=True) )
                else:
                    last_out = activation_critic(tf.layers.dense(last_out, hid_size_V, name="vffc%i"%(i+1), \
                    kernel_initializer=U.normc_initializer(1.0)))

                    apply_layer = lambda x : activation_critic(tf.layers.dense(x, hid_size_V,name="vffc%i"%(i+1),
                                        reuse=True))

                dropout_networks=generate_dropout_layer(apply_layer,dropout_networks,self.V_keep_prob)

            ## final layer
            self.vpred = tf.layers.dense(last_out, 1, name="vffinal", kernel_initializer=U.normc_initializer(1.0))[:,0]

            apply_layer = lambda x : tf.layers.dense(x, 1, activation=None, \
                        name="vffinal", reuse=True)[:,0]
            dropout_networks=generate_dropout_layer(apply_layer,dropout_networks,self.V_keep_prob)

            self.vpred_mc_mean=tf.add_n(dropout_networks) / float(len(dropout_networks))
            self.vpred_dropout_networks=dropout_networks


        #######################
        ## Policy
        last_out = obz
        with tf.variable_scope("policy"):
            if not self.dropout_on_policy:
                for i in range(num_hid_layers):
                    last_out = U.dense(last_out, hid_size_actor, "polfc%i"%(i+1), \
                    weight_init=U.normc_initializer(1.0))
                    last_out = activation_actor(last_out)
                if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                    mean = U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
                    logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                    pdparam = U.concatenate([mean, mean * 0.0 + logstd], axis=1)
                else:
                    pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
                self.pd = pdtype.pdfromflat(pdparam)
            else:
                dropout_networks = [last_out] * mc_samples
                dropout_networks = generate_dropout_layer(lambda x: x, dropout_networks, 1.0)

                for i in range(num_hid_layers):
                    last_out = activation_actor(tf.layers.dense(last_out, hid_size_actor, activation=None, name="polfc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0),  bias_initializer=tf.zeros_initializer()))
                    apply_layer = lambda x : activation_actor(tf.layers.dense(x, hid_size_actor, activation=None, name="polfc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0), bias_initializer=tf.zeros_initializer(), reuse=True))
                    dropout_networks = generate_dropout_layer(apply_layer, dropout_networks, pol_keep_prob)

                net = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name="polfinal", activation=None, kernel_initializer=U.normc_initializer(0.01))
                apply_layer = lambda x : tf.layers.dense(x, pdtype.param_shape()[0]//2, activation=None, name="polfinal", kernel_initializer=U.normc_initializer(0.01), reuse=True)
                dropout_networks=generate_dropout_layer(apply_layer, dropout_networks, pol_keep_prob)

                self.pd = pdtype.pdfromflat(dropout_networks, tau)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())

        last_out = obz

        ### MAIN CHANGES
        ## if dropout:

        if dropout_on_V:
            vact = self.vpred_mc_mean
        else:
            vact = self.vpred

        if dropout_on_policy:
            self._actsfunc = [U.function([ob], [x, vact]) for x in dropout_networks]

            self._act = self.dropout_act
        else:
            self._actfunc = U.function([stochastic, ob], [ac, vact])
            self._act = self.reg_act

    def reg_act(self, stochastic, ob):
        ac1, vpred1 =  self._actfunc(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def dropout_act(self, stochastic, ob):
        """
        """
        assert stochastic == True # PPO doesn't apply to deterministic case, therefore shouldn't be using deterministic policies
        random_index = random.randrange(0,len(self._actsfunc))

        ac1, vpred1 =  self._actsfunc[random_index](ob[None])
        return ac1[0], vpred1[0]

    def act(self, stochastic, ob):
        return self._act(stochastic, ob)

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []
