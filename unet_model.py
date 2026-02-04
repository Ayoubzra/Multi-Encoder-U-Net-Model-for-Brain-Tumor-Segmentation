from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

def multi_encoder_unet(inputs, inputs_, inputs__, ker_init='he_normal', dropout=0.2):
    # First Encoder
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(inputs)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv1)
    pool = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool)
    conv = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv2)
    drop2 = Dropout(dropout)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv3)
    drop3 = Dropout(dropout)(conv3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop3)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv5)
    drop5 = Dropout(dropout)(conv5)

    # Second Encoder
    conv1_ = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(inputs_)
    conv1_ = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv1_)
    pool_ = MaxPooling2D(pool_size=(2, 2))(conv1_)
    conv_ = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool_)
    conv_ = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv_)
    pool1_ = MaxPooling2D(pool_size=(2, 2))(conv_)
    conv2_ = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool1_)
    conv2_ = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv2_)
    drop2_ = Dropout(dropout)(conv2_)
    pool2_ = MaxPooling2D(pool_size=(2, 2))(drop2_)
    conv3_ = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool2_)
    conv3_ = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv3_)
    drop3_ = Dropout(dropout)(conv3_)

    # Third Encoder
    conv1__ = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(inputs__)
    conv1__ = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv1__)
    pool__ = MaxPooling2D(pool_size=(2, 2))(conv1__)
    conv__ = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool__)
    conv__ = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv__)
    pool1__ = MaxPooling2D(pool_size=(2, 2))(conv__)
    conv2__ = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool1__)
    conv2__ = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv2__)
    drop2__ = Dropout(dropout)(conv2__)
    pool2__ = MaxPooling2D(pool_size=(2, 2))(drop2__)
    conv3__ = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(pool2__)
    conv3__ = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv3__)
    drop3__ = Dropout(dropout)(conv3__)

    # Decoder / Upsampling
    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer=ker_init)(UpSampling2D(size=(2,2))(drop5))
    merge7 = concatenate([up7, drop3, drop3_, drop3__], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer=ker_init)(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([up8, drop2, drop2_, drop2__], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer=ker_init)(UpSampling2D(size=(2,2))(conv8))
    merge9 = concatenate([up9, conv, conv_, conv__], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv9)

    up = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer=ker_init)(UpSampling2D(size=(2,2))(conv9))
    merge = concatenate([up, conv1, conv1_, conv1__], axis=3)
    conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(merge)
    conv = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer=ker_init)(conv)
    conv10 = Conv2D(4, (1,1), activation='softmax')(conv)

    return Model(inputs=[inputs, inputs_, inputs__], outputs=conv10)