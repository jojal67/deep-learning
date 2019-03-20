from keras.layers import Input, Conv2D, Activation, BatchNormalization
from keras.layers.merge import Add
from keras.layers.core import Dropout
from keras.layers import Input, Activation, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.core import Lambda
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
import keras.backend as K
from keras.applications.vgg16 import VGG16
from layer_utils import ReflectionPadding2D, res_block



def res_block(input, filters, kernel_size=(3,3), strides=(1,1), use_dropout=False):
	"""
	Instanciate a Keras Resnet Block using sequential API.
	:param input: Input tensor
	:param filters: Number of filters to use
	:param kernel_size: Shape of the kernel for the convolution
	:param strides: Shape of the strides for the convolution
	:param use_dropout: Boolean value to determine the use of dropout
	:return: Keras Model
	"""
	x = ReflectionPadding2D((1,1))(input)
	x = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,)(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	if use_dropout:
		x = Dropout(0.5)(x)
	x = ReflectionPadding2D((1,1))(x)
	x = Conv2D(filters=filters,kernel_size=kernel_size,strides=strides,)(x)
	x = BatchNormalization()(x)
	# Two convolution layers followed by a direct connection between input and output
	merged = Add()([input, x])
	return merged





def generator_model():
	"""Build generator architecture."""
	# Current version : ResNet block
	inputs = Input(shape=image_shape)

	x = ReflectionPadding2D((3, 3))(inputs)
	x = Conv2D(filters=ngf, kernel_size=(7,7), padding='valid')(x)
	x = BatchNormalization()(x)
	x = Activation('relu')(x)

	# Increase filter number
	n_downsampling = 2
	for i in range(n_downsampling):
		mult = 2**i
		x = Conv2D(filters=ngf*mult*2, kernel_size=(3,3), strides=2, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

        # Apply 9 ResNet blocks
	mult = 2**n_downsampling
	for i in range(n_blocks_gen):
		x = res_block(x, ngf*mult, use_dropout=True)

	# Decrease filter number to 3 (RGB)
	for i in range(n_downsampling):
		mult = 2**(n_downsampling - i)
		x = Conv2DTranspose(filters=int(ngf * mult / 2), kernel_size=(3,3), strides=2, padding='same')(x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)

	x = ReflectionPadding2D((3,3))(x)
	x = Conv2D(filters=output_nc, kernel_size=(7,7), padding='valid')(x)
	x = Activation('tanh')(x)

	# Add direct connection from input to output and recenter to [-1, 1]
	outputs = Add()([x, inputs])
	outputs = Lambda(lambda z: z/2)(outputs)

	model = Model(inputs=inputs, outputs=outputs, name='Generator')
	return model



ndf = 64
output_nc = 3
input_shape_discriminator = (256, 256, output_nc)


def discriminator_model():
	"""Build discriminator architecture."""
	n_layers, use_sigmoid = 3, False
	inputs = Input(shape=input_shape_discriminator)

	x = Conv2D(filters=ndf, kernel_size=(4,4), strides=2, padding='same')(inputs)
	x = LeakyReLU(0.2)(x)

	nf_mult, nf_mult_prev = 1, 1
	for n in range(n_layers):
		nf_mult_prev, nf_mult = nf_mult, min(2**n, 8)
		x = Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=2, padding='same')(x)
		x = BatchNormalization()(x)
		x = LeakyReLU(0.2)(x)

	nf_mult_prev, nf_mult = nf_mult, min(2**n_layers, 8)
	x = Conv2D(filters=ndf*nf_mult, kernel_size=(4,4), strides=1, padding='same')(x)
	x = BatchNormalization()(x)
	x = LeakyReLU(0.2)(x)

	x = Conv2D(filters=1, kernel_size=(4,4), strides=1, padding='same')(x)
	if use_sigmoid:
		x = Activation('sigmoid')(x)

	x = Flatten()(x)
	x = Dense(1024, activation='tanh')(x)
	x = Dense(1, activation='sigmoid')(x)

	model = Model(inputs=inputs, outputs=x, name='Discriminator')
	return model





def generator_containing_discriminator_multiple_outputs(generator, discriminator):
	inputs = Input(shape=image_shape)
	generated_images = generator(inputs)
	outputs = discriminator(generated_images)
	model = Model(inputs=inputs, outputs=[generated_images, outputs])
	return model

def perceptual_loss(y_true, y_pred):
	vgg = VGG16(include_top=False, weights='imagenet', input_shape=image_shape)
	loss_model = Model(inputs=vgg.input, outputs=vgg.get_layer('block3_conv3').output)
	loss_model.trainable = False
	return K.mean(K.square(loss_model(y_true) - loss_model(y_pred)))

def wasserstein_loss(y_true, y_pred):
	return K.mean(y_true*y_pred)




image_shape = (256, 256, 3)
# Initialize models
g = generator_model()
d = discriminator_model()
d_on_g = generator_containing_discriminator_multiple_outputs(g, d)


