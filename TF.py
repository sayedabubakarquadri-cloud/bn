import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train=x_train/255.0
x_test=x_test/255.0
model=keras.Sequential([keras.Input(shape=(28,28)),layers.Flatten(),layers.Dense(512,activation="relu"),layers.Dropout(0.2),layers.Dense(10,activation="softmax")])
log_dir="logs/fit"+datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callbacks=keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)
model.compile(optimizer=keras.optimizers.SGD(),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
model.fit(x_train,y_train,epochs=10,validation_data=(x_test,y_test),callbacks=[tensorboard_callbacks])
test_accuracy,test_loss,=model.evaluate(x_test,y_test)
print("Test Accuracy:",test_accuracy)
print("Test loss:",test_loss)
model.summary()
