from __future__ import division
from keras.models import Model, Input
from keras.layers import Conv2D, Concatenate, Activation, Lambda, Add

class DenoisingAutoEncoderSR:

    def __init__(self, scale_factor, action_type):
        def __init__(self, model_name, scale_factor, action_type):
        """
        Base model to provide a standard interface of adding Super Resolution models
        """
        self.model = None  # type: Model
        self.model_name = model_name
        self.scale_factor = scale_factor
        self.weight_path = None
        self.action_type = action_type

        self.type_scale_type = "norm"  # Default = "norm" = 1. / 255
        self.type_requires_divisible_shape = False
        self.type_true_upscaling = False

        self.evaluation_func = None
        self.uses_learning_phase = False


        self.n1 = 64
        self.n2 = 32

        self.weight_path = "weights/Denoising_AutoEncoder_%dX.h5" % self.scale_factor
        self.file_format = "tiff"

    def create_model(self, height=32, width=32, channels=3, load_weights=False, batch_size=128):
        """
            Creates a model to remove / reduce noise from upscaled images.
        """

        # Perform check that model input shape is divisible by 4
        if self.action_type == 'training':
            height *= self.scale_factor
            width *= self.scale_factor

        if self.type_requires_divisible_shape and height is not None and width is not None:
            assert height % 4 == 0, "Height of the image must be divisible by 4"
            assert width % 4 == 0, "Width of the image must be divisible by 4"

        if width is not None and height is not None:
            shape = (width, height, channels)
        else:
            shape = (None, None, channels)

        init = Input(shape=shape)

        level1_1 = Conv2D(self.n1, (3, 3), activation='relu', padding='same')(init)
        level2_1 = Conv2D(self.n1, (3, 3), activation='relu', padding='same')(level1_1)

        level2_2 = Conv2DTranspose(self.n1, (3, 3), activation='relu', padding='same')(level2_1)
        level2 = Add()([level2_1, level2_2])

        level1_2 = Conv2DTranspose(self.n1, (3, 3), activation='relu', padding='same')(level2)
        level1 = Add()([level1_1, level1_2])

        decoded = Conv2D(channels, (5, 5), activation='linear', padding='same')(level1)

        model = Model(init, decoded)
        return model


def resBlock(x, channels, kernel_size, scale=0.1):
    tmp = Conv2D(
        channels,
        kernel_size,
        data_format="channels_first",
        kernel_initializer="he_uniform",
        padding="same",
    )(x)
    tmp = Activation("relu")(tmp)
    tmp = Conv2D(
        channels,
        kernel_size,
        data_format="channels_first",
        kernel_initializer="he_uniform",
        padding="same",
    )(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([x, tmp])


def s2model(input_shape, num_layers=32, feature_size=256):

    input10 = Input(shape=input_shape[0])
    input20 = Input(shape=input_shape[1])
    if len(input_shape) == 3:
        input60 = Input(shape=input_shape[2])
        x = Concatenate(axis=1)([input10, input20, input60])
    else:
        x = Concatenate(axis=1)([input10, input20])

    # Treat the concatenation
    x = Conv2D(
        feature_size,
        (3, 3),
        data_format="channels_first",
        kernel_initializer="he_uniform",
        activation="relu",
        padding="same",
    )(x)

    for _ in range(num_layers):
        x = resBlock(x, feature_size, [3, 3])

    # One more convolution, and then we add the output of our first conv layer
    x = Conv2D(
        input_shape[-1][0],
        (3, 3),
        data_format="channels_first",
        kernel_initializer="he_uniform",
        padding="same",
    )(x)
    if len(input_shape) == 3:
        x = Add()([x, input60])
        model = Model(inputs=[input10, input20, input60], outputs=x)
    else:
        x = Add()([x, input20])
        model = Model(inputs=[input10, input20], outputs=x)
    return model
