from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Add, Flatten, Dense


def define_model(input_shape, num_classes, loop=7):
    input = Input(shape=input_shape, name="input")
    # Block 1
    x1 = Conv1D(filters=250, kernel_size=3, padding="same", name="conv1_1")(input)
    x2 = Conv1D(filters=250, kernel_size=3, padding="same", name="conv1_2")(x1)
    x = Add(name="add_1")([x1, x2])
    # Block 2
    for i in range(2, loop + 1):
        x3 = MaxPool1D(pool_size=3, strides=2, padding="same", name="maxpool_%d" % i)(x)
        x4 = Conv1D(filters=250, kernel_size=3, padding="same", name="conv%d_1" % i)(x3)
        x5 = Conv1D(filters=250, kernel_size=3, padding="same", name="conv%d_2" % i)(x4)
        x = Add(name="add_%d" % i)([x3, x5])

    x = MaxPool1D(
        pool_size=3, strides=2, padding="same", name="maxpool_%d" % (loop + 1)
    )(x)
    x = Flatten()(x)
    x = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=x)
    return model


if __name__ == "__main__":
    net = define_model()
    import numpy as np
    import time

    for i in range(100):
        input = np.random.rand(256, 300)
        start = time.time()
        net.predict(input[None])
        runtime = time.time() - start
        print(runtime)
