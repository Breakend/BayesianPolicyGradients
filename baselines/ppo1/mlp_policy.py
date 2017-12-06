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

def generate_layer(apply_layer, prev_dropout_layer, keep_prob):
    new_networks = []
    for dropout_network in prev_dropout_layer:
        dropout_network = apply_layer(dropout_network)
        new_networks.append(dropout_network)
    return new_networks

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space,hid_size_V, hid_size_actor, num_hid_layers,V_keep_prob,\
             mc_samples,layer_norm,activation_critic,activation_actor, dropout_on_V,gaussian_fixed_var=True, sample_dropout=False):
        assert isinstance(ob_space, gym.spaces.Box)
        self.sample_dropout = sample_dropout

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        
        
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        
        
        
        self.mc_samples=mc_samples
        self.V_keep_prob=V_keep_prob
        
        ### MAIN CHANGES
        #######################
        # Value function  

      
        with tf.variable_scope("value_function"):
            
            dropout_networks = [last_out] * self.mc_samples
           # dropout_networks = generate_dropout_layer(lambda x: x, dropout_networks, self.V_keep_prob)
            
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
            dropout_networks=generate_layer(apply_layer,dropout_networks,self.V_keep_prob)
            
            mean,variance=tf.nn.moments(tf.stack(dropout_networks), 0)
            
            self.vpred_mc_mean=tf.add_n(dropout_networks) / float(len(dropout_networks))
            self.vpred_dropout_networks=dropout_networks
            
            self.variance=variance
            LAMBDA = tf.placeholder(dtype=tf.float32, shape=())
            self.v_lambda_variance=self.vpred_mc_mean+LAMBDA*tf.sqrt(variance)
         
            

            
        #######################    
        ## Policy
        last_out = obz
      
        with tf.variable_scope("policy"):
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

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
       
        
        
        last_out = obz
        
        ## BUilding function Q(s,a)
        
#        last_out2=self.pd.sample()
#        activation=tf.nn.relu
#        #######################
#        # Action Value function  
#        with tf.variable_scope("Q"):        
#            dropout_networks = [last_out] * self.mc_samples
#            dropout_networks = generate_dropout_layer(lambda x: x, dropout_networks, self.keep_prob)
#                 
#            ## concatenate state and action
#            last_out = tf.concat([last_out, last_out2], axis=-1)
#            
#            new_networks = []
#            for dropout_network in dropout_networks:
#                dropout_network = tf.concat([dropout_network, last_out2], axis=-1)
#                dropout_network, mask = U.bayes_dropout(dropout_network, self.keep_prob)
#                new_networks.append(dropout_network)
#            dropout_networks = new_networks
#            
#            ### hidden layers
#            for i in range(num_hid_layers):
#                
#                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="Q%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
#                apply_layer = lambda x : activation(tf.layers.dense(x, hid_size, activation=None, \
#                        name="Q%i"%(i+1), reuse=True))
#                dropout_networks=generate_dropout_layer(apply_layer,dropout_networks,self.keep_prob)
#            
#            ## final layer
#            self.qpred = tf.layers.dense(last_out, 1, name="Qfinal", kernel_initializer=U.normc_initializer(1.0))[:,0]
#            
#            apply_layer = lambda x : tf.layers.dense(x, 1, activation=None, \
#                        name="Qfinal", reuse=True)[:,0]
#            dropout_networks=generate_dropout_layer(apply_layer,dropout_networks,self.keep_prob)
#            
#            self.qpred_mc_mean=tf.add_n(dropout_networks) / float(len(dropout_networks))
#            self.qpred_dropout_networks=dropout_networks
        
        
        
        
        ### MAIN CHANGES
        ## if dropout:
        if dropout_on_V:
            if self.sample_dropout:
                self._act = [U.function([stochastic, ob], [ac, x]) for x in dropout_networks]
            else:
                self._act = U.function([stochastic, ob], [ac, self.vpred_mc_mean])
                       


            
        else:
            self._act = U.function([stochastic, ob], [ac, self.vpred])
        
 
    
     


       

    def act(self, stochastic, ob):
        if self.sample_dropout: 
            random_index = random.randrange(0,len(self._act))
            ac1, vpred1 =  self._act[random_index](stochastic, ob[None])
        else:
            ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    
    
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
   
    def get_initial_state(self):
        return []

