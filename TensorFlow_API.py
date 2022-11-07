import tensorflow as tf

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('val_loss')<0.3:
            print('\nReached 30.0% binary crossentropy so canceling training')
            self.model.stop_training=True
            
def custom_loss(pos_weight):
  def loss(y_true,y_pred):
    return tf.nn.weighted_cross_entropy_with_logits(y_true,y_pred, 
                                                    pos_weight, name=None)
  return loss

def model(pos_weight=2.5):
    model=tf.keras.models.Sequential([
                                      
                                      tf.keras.layers.Dense(64,activation=tf.nn.relu),
                                      tf.keras.layers.Dense(32,activation=tf.nn.relu),
                                      tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
                                    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-8),
                  loss=custom_loss(pos_weight))
    return model
  
y_train = np.asarray(y_train, dtype=np.float32)
y_val = np.asarray(y_val, dtype=np.float32)

dnn_model=model(3.5)

mycallback = MyCallback()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))

history = dnn_model.fit(X_train_upd,y_train,epochs=120,validation_split=0.2,
                      shuffle=True,
                      steps_per_epoch=len(X)//64,
                      workers=4,
                      verbose=1,callbacks=[lr_schedule,mycallback])

results=dnn_model.evaluate(X_val_upd,y_val,verbose=1,return_dict=True)
y_pred=dnn_model.predict(X_val_upd)
y_pred_val = [1 if pred>0.5 else 0 for pred in y_pred]
print('\nValidation Accuracy : %.2f'%accuracy_score(y_val,y_pred_val))

print('\nConfusion Matrix : ')
print(confusion_matrix(y_val,y_pred_val))

print('\nClassification Report : ')
print(classification_report(y_val,y_pred_val))

