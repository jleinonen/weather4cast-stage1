import tensorflow as tf
from tensorflow.keras.layers import Layer


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1,1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        return (
            s[0], 
            None if s[1] is None else s[1]+2*self.padding[0],
            None if s[2] is None else s[2]+2*self.padding[1],
            s[3]
        )

    def call(self, x):
        (i_pad,j_pad) = self.padding
        return tf.pad(x, [[0,0], [i_pad,i_pad], [j_pad,j_pad], [0,0]], 'REFLECT')


class SpatialWarp(Layer):
    def __init__(self, **kwargs):
        super(SpatialWarp, self).__init__(**kwargs)

    def build(self, input_shape):
        (img_shape, U_shape) = input_shape
        (i,j) = tf.meshgrid(
            tf.range(img_shape[1]),
            tf.range(img_shape[2]),
            indexing="ij"
        )
        self.i = tf.cast(tf.expand_dims(i,0), tf.float32)
        self.j = tf.cast(tf.expand_dims(j,0), tf.float32)

    def warp_slice(self, inputs):
        # morph image from given coordinates with bilinear interpolation
        (img, i, j) = inputs
        img_shape = img.shape

        # compute coordinates surrounding target points
        clip = lambda x, max: tf.clip_by_value(x, 0, max)
        i0 = clip(tf.cast(tf.math.floor(i),tf.int32), img_shape[0]-1)
        j0 = clip(tf.cast(tf.math.floor(j),tf.int32), img_shape[1]-1)
        i1 = clip(i0+1, img_shape[0]-1)
        j1 = clip(j0+1, img_shape[1]-1)
        dj = tf.reshape(j-tf.cast(j0, tf.float32), img_shape)
        di = tf.reshape(i-tf.cast(i0, tf.float32), img_shape)

        # compute bilinear interpolation
        f00 = tf.gather_nd(img, tf.stack((i0,j0), axis=-1))
        f01 = tf.gather_nd(img, tf.stack((i0,j1), axis=-1))
        f10 = tf.gather_nd(img, tf.stack((i1,j0), axis=-1))
        f11 = tf.gather_nd(img, tf.stack((i1,j1), axis=-1))
        f0 = f00+(f01-f00)*dj
        f1 = f10+(f11-f10)*dj
        img_t = f0+(f1-f0)*di
        
        return img_t
        
    def call(self, inputs):
        (img, U) = inputs

        # create output frames by advecting the input frame
        def transform_frame(frame_ind):
            di = U[:,frame_ind,:,:,0]
            dj = U[:,frame_ind,:,:,1]
            
            return tf.map_fn(self.warp_slice, 
                (img,self.i-di,self.j-dj), dtype=tf.float32)
        
        num_frames = U.get_shape()[1]
        img_t = [transform_frame(k) for k in range(num_frames)]
        return tf.stack(img_t, axis=1)

    def compute_output_shape(self, input_shapes):
        (img_shape, U_shape) = input_shapes
        return (img_shape[0],U_shape[1])+img_shape[1:]
