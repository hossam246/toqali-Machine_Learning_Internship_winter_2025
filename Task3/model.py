from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle

df = pd.read_csv('./first inten project.csv')

def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

df = df[['number of adults', 'number of children',
       'number of weekend nights', 'number of week nights', 'type of meal',
       'car parking space', 'room type', 'lead time', 'market segment type',
       'repeated', 'P-C', 'P-not-C', 'average price ', 'special requests', 'booking status']]

for i in ['number of children',
            'number of weekend nights', 'number of week nights','lead time',
            'P-C', 'P-not-C', 'average price ', 'special requests']:
            df  = remove_outliers_iqr(df, i)

df = df.rename(columns={ 'number of adults'         : 'number_of_adults',
                             'number of children'       : 'number_of_children',
                             'type of meal'             : 'type_of_meal',
                             'car parking space'        : 'car_parking_space',
                             'room type'                : 'room_type',
                             'market segment type'      : 'market_segment_type',
                             'lead time'                : 'lead_time',
                             'average price '           : 'average_price',
                             'special requests'         : 'special_requests',
                             'date of reservation'      : 'date_of_reservation',
                             'number of week nights'    : 'number_of_week_nights',
                             'number of weekend nights' : 'number_of_weekend_nights'
                           })


type_of_meal_encoder = preprocessing.LabelEncoder()
room_type_encoder = preprocessing.LabelEncoder()
market_segment_type_encoder = preprocessing.LabelEncoder()

df['type_of_meal']= type_of_meal_encoder.fit_transform(df['type_of_meal'])
df['room_type']= room_type_encoder.fit_transform(df['room_type'])
df['market_segment_type']= market_segment_type_encoder.fit_transform(df['market_segment_type'])

label_encoders = {
    'market_segment_type': market_segment_type_encoder,
    'room_type': room_type_encoder,
    'type_of_meal': type_of_meal_encoder
}

print(df.columns)

X, y = df.iloc[:, :-1], df.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)
y_pred_tree = tree.predict(x_test)
accuracy_tree = accuracy_score(y_test, y_pred_tree)

confusion_matrix(y_test, y_pred_tree)

print(classification_report(y_test, y_pred_tree))

pickle.dump(tree, open('model.pkl', 'wb'))
pickle.dump(label_encoders, open('label_encoder.pkl', 'wb'))
