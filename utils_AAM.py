import tensorflow as tf
from tensorflow.keras import backend as K
from ops import *
SN=True
def squash(s):
    n = tf.norm(s, axis=-1,keepdims=True)
    return tf.multiply(n**2/(1+n**2)/(n + tf.keras.backend.epsilon()), s)

def margin_loss(y_true, y_pred):
    lamb, margin = 0.5,0.3
    return K.sum((y_true * K.square(K.relu(1 - margin - y_pred)) + lamb * (
        1 - y_true) * K.square(K.relu(y_pred - margin))), axis=-1)


def AAM(x, channels, de=2, scope='AAM', trainable=True, reuse=False):
    # print(type(x), type(channels), channels)
    with tf.compat.v1.variable_scope(scope, reuse=reuse):
        f, offset1 = deform_con2v(x, num_outputs=channels, kernel_size=3, stride=1, trainable=trainable,
                                  name=scope + 'f_conv', reuse=reuse)
        #g, offset2 = deform_con2v(x, num_outputs=channels, kernel_size=3, stride=1, trainable=trainable,
                                  #name=scope + 'g_conv', reuse=reuse)
        #h, offset3 = deform_con2v(x, num_outputs=channels, kernel_size=3, stride=1, trainable=trainable,
                                  #name=scope + 'f_conv', reuse=reuse)
        #h = conv(x, channels, kernel=1, stride=1, sn=SN, scope='h_conv')

        # N = h * w
        s = tf.matmul(hw_flatten(f), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta_a = tf.compat.v1.nn.softmax(s, dim=-1)  # attention map

        o = tf.matmul(beta_a, hw_flatten(f))  # [bs, N, C]
  
        #gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        bs, h, w, C = x.get_shape().as_list()
        bs=16
        o = tf.reshape(o, (-1, h, w,C))  # [bs, h, w, C]
        #att = gamma * o
        #x = att + x

    return o