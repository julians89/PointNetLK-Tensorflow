import tensorflow as tf
import tensorflow_probability as tfp
import se3
import sys
sys.path.append("../utils")
import ply_utils
sys.path.append("../tf_models")
import tf_util
import transform as transform_utils
import numpy as np
import copy
from tensorflow.python import debug as tf_debug
import os
import tf_approxmatch

batch_size = 4
num_points = 2048

class PointLK():
    def __init__(self, delta=1.0e-2, learn_delta=False, max_iter=10, xtol=1e-8, learning_rate=1e-4):

        self.bn = False
        self.is_training = True
        self.delta = delta
        self.learn_delta = learn_delta
        w1 = delta
        w2 = delta
        w3 = delta
        v1 = delta
        v2 = delta
        v3 = delta


        self.dt = tf.get_variable('dt', [1, 6],
                                  initializer=tf.constant_initializer(np.array((w1, w2, w3, v1, v2, v3))),
                                  dtype=tf.float32,
                                  validate_shape=False,
                                  trainable=learn_delta)

        self.init = False

        self.p0 = tf.placeholder(tf.float32, shape=(None, num_points, 3))
        self.p1 = tf.placeholder(tf.float32, shape=(None, num_points, 3))
        self.igt = tf.placeholder(tf.float32, shape=(None, 4, 4))
        self.g0 = tf.placeholder(tf.float32, shape=(None, 4, 4))

        r, g, itr = self.iclk(self.g0, self.p0, self.p1, max_iter, xtol)

        #self.loss = self.loss(g, self.igt, r)
        self.loss = self.pw_loss(self.p0, se3.transform(g, self.p1), r)

        step = tf.Variable(0, trainable=False)

        learning_rate = tf.train.exponential_decay(
            1e-4,  # Base learning rate.
            step,  # Current index into the dataset.
            100000,  # Decay step.
            0.75,  # Decay rate.
            staircase=True)

        learning_rate = tf.maximum(learning_rate, 1e-8)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = optimizer.minimize(self.loss, step)

    def ptnet(self, point_cloud):

        with tf.variable_scope('pnet', reuse=tf.AUTO_REUSE):

            input_image = tf.expand_dims(point_cloud, -1)

            net = tf_util.conv2d(input_image, 64, [1, 3],
                                 padding='VALID', stride=[1, 1],
                                 bn=self.bn, is_training=self.is_training,
                                 scope='conv1')

            net = tf_util.conv2d(net, 64, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=self.bn, is_training=self.is_training,
                                 scope='conv2')

            net = tf_util.conv2d(net, 64, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=self.bn, is_training=self.is_training,
                                 scope='conv3')
            net = tf_util.conv2d(net, 64 * 2, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=self.bn, is_training=self.is_training,
                                 scope='conv4')
            net = tf_util.conv2d(net, 1024, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=self.bn, is_training=self.is_training,
                                 scope='conv5')

            net = tf.reduce_max(net, axis=1, keepdims=True)


            """
            net = tf_util.conv2d(net, 256, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=self.bn, is_training=self.is_training, activation_fn=None,
                                 scope='conv6')
            """


            net = tf.reshape(net, [-1, 1024])


        return net

    def approx_Jic(self, p0, f0, dt):
        # p0: [B, N, 3], Variable
        # f0: [B, K], corresponding feature vector
        # dt: [B, 6], Variable
        # Jk = (ptnet(p(-delta[k], p0)) - f0) / delta[k]

        #batch_size = tf.shape(p0)[0]
        num_points = tf.shape(p0)[1]

        # compute transforms
        d = tf.linalg.diag(dt) #[batch_size, 6, 6]
        transf = se3.exp(- tf.reshape(d, [-1, 6]))
        transf = tf.reshape(transf, [-1, 6, 1, 4, 4])

        p = se3.transform(transf, tf.reshape(p0, [-1, 1, num_points, 3]))  # x [B, 1, N, 3] -> [B, 6, N, 3]
        p = tf.reshape(p, [batch_size, 6, num_points, 3]) # Each point cloud is transformed w.r.t dt

        f0 = tf.expand_dims(f0, axis=-1)
        f = tf.transpose(tf.reshape(self.ptnet(tf.reshape(p, [-1, num_points, 3])), [batch_size, 6, -1]), perm=[0, 2, 1])

        df = f0 - f  # [B, K, 6]
        J = df / tf.expand_dims(dt, axis=1)

        return J

    def update(self, g, dx):
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = se3.exp(dx)

        return tf.matmul(dg, g)

    def h_iteration(self, H, max_steps=1000, dettol=1e-6):

        step = 1e-4

        def cond(H, i):
            check_trans = tf.less(tf.reduce_min(tf.abs(tf.linalg.det(H)), axis=-1), dettol)
            #check_trans = tf.less(tf.linalg.det(H), dettol)

            check_iter = tf.less(i, max_steps)

            return check_trans

        def body(H, i):

            #Get signs of H:
            signs = tf.sign(tf.linalg.diag_part(H))
            #H += tf.linalg.diag(signs) * 1e-6

            indicator = tf.where(tf.less(tf.abs(tf.linalg.det(H)), dettol), tf.ones([batch_size]), tf.zeros([batch_size]))
            indicator = tf.reshape(indicator, [-1, 1, 1])
            add = tf.tile(tf.reshape(tf.eye(6) * step, [1, 6, 6]), [batch_size, 1, 1])
            #add = tf.linalg.diag(signs) * step

            H += indicator * add

            return [H, i]

        i0 = tf.constant(0, dtype=tf.int32)

        H_final, i_final = tf.while_loop(cond, body, loop_vars=[H, i0], back_prop=True, parallel_iterations=1)

        return H_final

    def transform_iteration(self, pinv, f0, p1, g, max_steps=10, xtol=1.e-8):
        """
        :param A:
        :return:
        """

        ptnet = self.ptnet

        def cond(g, dx, r, i):

            check_trans = tf.greater(tf.reduce_max(tf.reduce_sum(dx ** 2, axis=1)), xtol)
            check_iter = tf.less(i, max_steps)

            #return check_iter
            return tf.logical_and(check_trans, check_iter)
            #return tf.logical_or(check_trans, check_iter)

        def body(g, dx, r, i):
            p = se3.transform(tf.reshape(g, [-1, 4, 4]), p1) # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            f = ptnet(p) # [B, N, 3] -> [B, K]
            r = f - f0

            dx = - tf.reshape(tf.matmul(pinv, tf.expand_dims(r, axis=-1)), [batch_size, 6])


            # All dx smaller xtol are set to 0.
            """
            indicator = tf.where(tf.less(tf.reduce_sum(dx ** 2, axis=-1), xtol), tf.zeros([batch_size]), tf.ones([batch_size]))
            indicator = tf.reshape(indicator, [-1, 1])
            dx *= indicator
            """

            dg = se3.exp(dx)
            g = tf.matmul(dg, g)

            i += 1

            return [g, dx, r, i]


        i0 = tf.constant(0, dtype=tf.int32)
        dx = tf.ones([batch_size, 6])

        r = tf.zeros_like(f0)


        g_final, dx_final, r_final, i_final = tf.while_loop(cond, body, loop_vars=[g, dx, r, i0], back_prop=True, parallel_iterations=1)

        return g_final, dx_final, r_final, i_final

    def iclk(self, g0, p0, p1, maxiter, xtol):


        #training = self.ptnet.training
        #batch_size = tf.shape(p0)[0]

        g = g0
        #self.g_series = []
        #self.g_series.append(g0)

        """
        if training:
            # first, update BatchNorm modules
            f0 = self.ptnet(p0)
            f1 = self.ptnet(p1)
        self.ptnet.eval() # and fix them.
        """
        # re-calc. with current modules
        f0 = self.ptnet(p0) # [B, N, 3] -> [B, K]

        self.f0 = f0

        # approx. J by finite difference
        dt = tf.tile(tf.cast(self.dt, dtype=p0.dtype), [batch_size, 1])


        J = self.approx_Jic(p0, f0, dt)
        # compute pinv(J) to solve J*x = -r
        Jt = tf.transpose(J, perm=[0, 2, 1])
        H = tf.matmul(Jt, J) # [B, 6, 6]
        H = tf.reshape(H, [-1, 6, 6])

        #TODO: ensure invertibility
        #H += tf.reshape(tf.eye(6) * 1e-2, [1, 6, 6])
        # itertively add values to diag until diag > thresh
        H = self.h_iteration(H)

        B = tf.linalg.inv(H)

        #TODO: do not use pinv --> this uses SVD, which causes unstable gradients!
        #B = tfp.math.pinv(H)


        self.B = B

        pinv = tf.matmul(B, Jt)

        g, r, dx, itr = self.transform_iteration(pinv, f0, p1, g, max_steps=maxiter, xtol=xtol)

        self.dx = tf.reduce_max(tf.reduce_sum(dx ** 2, axis =-1))
        self.g = g
        self.itr = itr

        return r, g, (itr+1)

    def loss(self, g, igt, r=None):
        """ |g*igt - I| (should be 0) """

        A = tf.matmul(g, tf.cast(igt, g.dtype))
        I = tf.reshape(tf.eye(4, dtype=A.dtype), [1, 4, 4])

        loss = tf.nn.l2_loss(A - I)


        #Regularization
        #mat_diff = tf.matmul(g, tf.linalg.matrix_transpose(g))
        #loss += 1e-2 * tf.nn.l2_loss(mat_diff - I)

        """
        if r is not None:
            loss += tf.nn.l2_loss(r)
        """

        return loss

    def pw_loss(self, p0, p1, r=None):

        """
        match = tf_approxmatch.approx_match(p0, p1)
        loss = tf.reduce_mean(tf_approxmatch.match_cost(p0, p1, match))
        """

        loss = tf.reduce_mean((p0 - p1) ** 2)

        if r is not None:
            loss += tf.nn.l2_loss(r)

        return loss

    def train(self, np_p0, np_p1, np_igt, g0, sess):

        feed_dict = {self.p0: np_p0, self.p1: np_p1, self.igt: np_igt, self.g0: g0}

        if not self.init:
            sess.run(tf.global_variables_initializer())
            self.init = True

        _, l, g, dx, itr = sess.run([self.train_op, self.loss, self.g, self.dx, self.itr], feed_dict)
        #l, g = sess.run([self.loss, self.g], feed_dict)

        #print ('dx: ', dx)
        #print ('itr: ', itr)


        return l, g, itr, dx

        """
        print ('loss: ', l)
        print ('f0: ', f0)
        print ('g: ', g)
        print ('igt: ', np_igt)
        """

        """
        h, b, dx, g = sess.run([self.H, self.B, self.dx, self.g], feed_dict)
        print ('H: ', h)
        print ('B: ', b)
        print ('dx', dx)
        print ('g: ', g)
        """

if __name__ == '__main__':
    plk = PointLK(max_iter=10, learning_rate=1.e-4)


    sess = tf.Session()

    data_dir = "/media/data/Datasets/ModelNet10_ply"
    categories = ["bathtub", "bed", "chair", "desk", "dresser", "monitor", "nightstand", "sofa", "table", "toilet"]

    i = 0
    avg_loss = 0.
    avg_itr = 0.
    avg_dx = 0.
    while True:

        p0 = []
        p1 = []
        igt = []
        g0 = []

        """
        for b in range(batch_size):

            # Generate Sphere
            vec = np.random.randn(3, num_points)
            vec /= np.linalg.norm(vec, axis=0)
            vec = vec.transpose()
            

            noise = 0.0001 * np.random.randn(num_points, 3)
            vec += noise

            vec -= np.mean(vec, axis=0, keepdims=True)
            vec /= np.max(np.sum(vec, axis=-1, keepdims=True), keepdims=True)


            
            #init_rot = transform_utils.euler_to_so3([(np.random.ranf() - 0.5) * 4 * np.pi,
            #                                        (np.random.ranf() - 0.5) * 4 * np.pi,
            #                                        (np.random.ranf() - 0.5) * 4 * np.pi])


            #vec = np.matmul(vec, init_rot.T)

            

            rnd_rot = transform_utils.euler_to_so3([(np.random.ranf() - 0.5) * 4 * np.pi,
                                                    (np.random.ranf() - 0.5) * 4 * np.pi,
                                                    (np.random.ranf() - 0.5) * 4 * np.pi])

            b_igt = np.eye(4)
            b_igt[:3, :3] = rnd_rot

            #p0.append(copy.deepcopy(vec))
            #p1.append(np.matmul(copy.deepcopy(vec), rnd_rot.T))
            p0.append(np.matmul(copy.deepcopy(vec), rnd_rot.T))
            p1.append(copy.deepcopy(vec))

            igt.append(b_igt)
            g0.append(np.eye(4))
        """



        for b in range(batch_size):

            rnd_cat = np.random.choice(categories)
            path = os.path.join(data_dir, rnd_cat)
            path = os.path.join(path, "train")
            files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
            rnd_file = np.random.choice(files)

            data = ply_utils.read_ply(os.path.join(path, rnd_file))
            idx = np.random.choice(np.arange(len(data)), num_points)
            data = data[idx]

            data -= np.mean(data, axis=0, keepdims=True)
            data /= np.max(np.sum(data ** 2, axis=-1, keepdims=True), axis=0, keepdims=True)


            
            init_rot = transform_utils.euler_to_so3([(np.random.ranf() - 0.5) * 4 * np.pi,
                                                    (np.random.ranf() - 0.5) * 4 * np.pi,
                                                    (np.random.ranf() - 0.5) * 4 * np.pi])

            data = np.matmul(data, init_rot.T)
            

            rnd_rot = transform_utils.euler_to_so3([(np.random.ranf() - 0.5) * 1. * np.pi,
                                                     (np.random.ranf() - 0.5) * 1. * np.pi,
                                                     (np.random.ranf() - 0.5) * 1. * np.pi])


            b_p0 = copy.deepcopy(data)
            noise = 0.01 * np.random.randn(num_points, 3)
            noise = np.clip(noise, -0.05, 0.05)
            b_p0 += noise
            b_p0 = np.matmul(b_p0, rnd_rot.T)

            b_p1 = copy.deepcopy(data)
            noise = 0.01 * np.random.randn(num_points, 3)
            noise = np.clip(noise, -0.05, 0.05)
            b_p1 += noise

            b_igt = np.eye(4)
            b_igt[:3, :3] = rnd_rot

            b_g0 = np.eye(4)

            p0.append(b_p0)
            p1.append(b_p1)
            igt.append(b_igt)
            g0.append(b_g0)

            b += 1


        loss, g, itr, dx = plk.train(p0, p1, igt, g0, sess)
        #print ('transf: ', transf)

        #print ('loss: ', loss)

        avg_loss += loss
        avg_itr += itr
        avg_dx += dx

        info_step = 250
        if i % info_step == 0 and i > 0:
            print ('step: ', i)
            print ('avg loss: ', avg_loss / float(info_step))
            avg_loss = 0.

            print ('avg itr: ', avg_itr / float(info_step))
            avg_itr = 0.

            print ('avg dx: ', avg_dx / float(info_step))
            avg_dx = 0.

            print ('snapshot:')
            print ('g:')
            print (g)

            print ('igt:')
            print (igt)

            p1_transformed = np.matmul(np.concatenate((p1, np.ones((batch_size, num_points, 1))), axis=-1), np.transpose(g, (0, 2, 1)))[:, :, :3]

            for b in range(batch_size):
                ply_utils.write_to_ply(p0[b], props=['x', 'y', 'z'], path='../../testdata/gt_' + str(b) +'.ply')

                ply_utils.write_to_ply(p1_transformed[b], props=['x', 'y', 'z'], path='../../testdata/transformed_' + str(b) +'.ply')


        i += 1