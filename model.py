from keras.applications import ResNet50, InceptionResNetV2
from keras.layers import Dense
from keras.models import Model
from keras import backend as K


def get_model(model_name="ResNet50"):
    base_model = None

    if model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(
            224, 224, 3), pooling="avg")
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(
            include_top=False, weights='imagenet', input_shape=(299, 299, 3), pooling="avg")

    # prediction = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax",
    #                    name="pred_age")(base_model.output)

    # model = Model(inputs=base_model.input, outputs=prediction)
    predictions_g = Dense(units=2, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                          name="pred_gender")(base_model.output)
    predictions_a = Dense(units=101, kernel_initializer="he_normal", use_bias=False, activation="softmax",
                          name="pred_age")(base_model.output)
    model = Model(inputs=base_model.input, outputs=[
                  predictions_g, predictions_a])
    return model


def main():
    model = get_model("InceptionResNetV2")
    model.summary()


if __name__ == '__main__':
    main()
