# -- 1. Libraries
import tensorflow as tf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score,classification_report,\
                        confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
    
# -- 2. Data preprocessing
data=pd.read_csv('train.csv')

y = data['Survived'].copy()
X = data.drop(['Survived','PassengerId'], axis=1).copy()

# take small train size for demonstration purposes
X_train_full, X_val_full, y_train, y_val = train_test_split(X, y, train_size=0.6,
                                                                stratify=y,random_state=0)

# Select categorical columns with relatively low cardinality
categorical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 40 and 
                        X_train_full[cname].dtype == "object"]
# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]

# preprocess data without pipeline
num_imputer=SimpleImputer(strategy='mean')
X_train_num_imputed=pd.DataFrame(num_imputer.fit_transform(X_train_full[numerical_cols]))
X_val_num_imputed=pd.DataFrame(num_imputer.transform(X_val_full[numerical_cols]))

X_train_num_imputed.columns=X_train_full[numerical_cols].columns
X_val_num_imputed.columns=X_val_full[numerical_cols].columns

cat_imputer=SimpleImputer(strategy='most_frequent')

X_train_cat_imputed=pd.DataFrame(cat_imputer.fit_transform(X_train_full[categorical_cols]))
X_val_cat_imputed=pd.DataFrame(cat_imputer.transform(X_val_full[categorical_cols]))

X_train_cat_imputed.columns=X_train_full[categorical_cols].columns
X_val_cat_imputed.columns=X_val_full[categorical_cols].columns

OH_encoder=OneHotEncoder(handle_unknown='ignore',sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train_cat_imputed))
OH_cols_val = pd.DataFrame(OH_encoder.transform(X_val_cat_imputed))

OH_cols_train.index = X_train_cat_imputed.index
OH_cols_val.index = X_val_cat_imputed.index

X_train=pd.concat([X_train_num_imputed,OH_cols_train],axis=1)
X_val=pd.concat([X_val_num_imputed,OH_cols_val],axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[['Age','Fare']])
X_val_scaled = scaler.transform(X_val[['Age','Fare']])
X_train_scaled = pd.DataFrame(X_train_scaled,columns=['Age','Fare'])
X_val_scaled = pd.DataFrame(X_val_scaled,columns=['Age','Fare'])

X_train_upd = X_train.copy()
X_val_upd = X_val.copy()

X_train_upd['Age'] = X_train_scaled['Age']
X_train_upd['Fare'] = X_train_scaled['Fare']
X_val_upd['Age'] = X_val_scaled['Age']
X_val_upd['Fare'] = X_val_scaled['Fare']

# -- 3. Prepare DNN
class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('val_loss')<0.38:
            print('\nReached 38.0% binary crossentropy so canceling training')
            self.model.stop_training=True
          
def model():
    model=tf.keras.models.Sequential([
                                      
                                      tf.keras.layers.Dense(64,activation=tf.nn.relu),
                                      tf.keras.layers.Dense(32,activation=tf.nn.relu),
                                      tf.keras.layers.Dense(1,activation=tf.nn.sigmoid)
                                    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-8),
                  loss=tf.keras.losses.BinaryCrossentropy())
    return model
    
def plot_loss(history):
    plt.plot(history.history['loss'],label='training_loss')
    plt.plot(history.history['val_loss'],label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
dnn_model=model()

mycallback = MyCallback()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch / 20))


# -- 4. Train the model
history=dnn_model.fit(X_train_upd,y_train,epochs=110,validation_split=0.2,
                      shuffle=True,
                      steps_per_epoch=len(X)//64,
                      workers=4,
                      verbose=1,callbacks=[lr_schedule,mycallback])

# -- 5. Evaluation of the model performance
plot_loss(history)
plt.semilogx(history.history["lr"], history.history["loss"])
results=dnn_model.evaluate(X_val_upd,y_val,verbose=1,return_dict=True)
y_pred=dnn_model.predict(X_val_upd)
y_pred_val = [1 if pred>0.5 else 0 for pred in y_pred]

print('\nValidation Accuracy : %.2f'%accuracy_score(y_val,y_pred_val))

print('\nConfusion Matrix : ')
print(confusion_matrix(y_val,y_pred_val))

print('\nClassification Report : ')
print(classification_report(y_val,y_pred_val))
