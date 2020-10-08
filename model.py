import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
data=pd.read_csv('mobile_price.csv')

data.dropna(inplace=True)
x=data[['battery_power','int_memory','pc','px_height','px_width','ram']]
y=data['price_range']       
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=10)
model=knn.fit(x_train,y_train)
y_pred=model.predict(x_test)

pickle.dump(knn, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))


