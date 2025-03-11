#importing libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle

#setting up the dataframe
hotel_reservations = pd.read_csv(r"C:\Cellula\first project\first inten project.csv")
hotel_reservations.columns = hotel_reservations.columns.str.replace(' ', '_')
hotel_reservations.columns = hotel_reservations.columns.str.replace('price_', 'price')
#the date 2018-2-29 does not exist on the calendar so it was replaced with 2018-2-28
hotel_reservations['date_of_reservation'] = hotel_reservations['date_of_reservation'].str.replace('2018-2-29', '2/28/2018') 
hotel_reservations['date_of_reservation'] = pd.to_datetime(hotel_reservations['date_of_reservation'])
hotel_reservations['total_nights'] = hotel_reservations.iloc[:, 3:5].sum(axis= 1)
hotel_reservations['total_individuals'] = hotel_reservations.iloc[:, 1:3].sum(axis=1)
hotel_reservations['reservation_month'] = hotel_reservations['date_of_reservation'].dt.month
hotel_reservations['reservation_season'] = 'winter'
hotel_reservations.loc[hotel_reservations['reservation_month'].between(3,5), 'reservation_season'] = 'spring'
hotel_reservations.loc[hotel_reservations['reservation_month'].between(6,8), 'reservation_season'] = 'summer'
hotel_reservations.loc[hotel_reservations['reservation_month'].between(9,11), 'reservation_season'] = 'autumn'
hotel_reservations['no_lead'] = (hotel_reservations['lead_time'] == 0).astype(int)
hotel_reservations['log_lead'] = hotel_reservations['lead_time']


#replacing average price outliers with min/max since it has alot of ouliers
q1 = np.percentile(hotel_reservations['average_price'], 25, method= 'midpoint')
q3 = np.percentile(hotel_reservations['average_price'], 75, method= 'midpoint')
iqr = q3-q1
max = q3 + 1.5*iqr
min = q1 - 1.5*iqr
hotel_reservations.loc[hotel_reservations['average_price'] > max, 'average_price'] = max
hotel_reservations.loc[hotel_reservations['average_price'] < min, 'average_price'] = min

#dropping outliers for number of nights and special requests since they have a few outliers
for col in hotel_reservations[['number_of_week_nights', 'number_of_weekend_nights', 'total_nights', 'special_requests']]:
    q1 = np.percentile(hotel_reservations[col], 25, method= 'midpoint')
    q3 = np.percentile(hotel_reservations[col], 75, method= 'midpoint')
    iqr = q3-q1
    max = round(q3 + 1.5*iqr)
    min = round(q1 - 1.5*iqr)
    upper_indices = hotel_reservations[hotel_reservations[col] > max].index
    lower_indices = hotel_reservations[hotel_reservations[col] < min].index
    hotel_reservations.drop(index= upper_indices, inplace = True)
    hotel_reservations.drop(index= lower_indices, inplace = True)

#only 22 reservations have above 2 children, 16 reservations have above 3 adults and 18 reservations have above 4 total individuals so these reservations will be dropped
for col in hotel_reservations[['number_of_children', 'number_of_adults', 'total_individuals']]:
    if col == 'number_of_children':
        outlier_index = hotel_reservations[hotel_reservations[col] > 2].index
        hotel_reservations.drop(index= outlier_index, inplace= True)
    elif col == 'number_of_adults':
        outlier_index = hotel_reservations[hotel_reservations[col] > 3].index
        hotel_reservations.drop(index= outlier_index, inplace= True)
    else:
        outlier_index = hotel_reservations[hotel_reservations[col] > 4].index
        hotel_reservations.drop(index= outlier_index, inplace= True)


hotel_reservations.reset_index(drop= True, inplace= True)


#dropping costing features
hotel_reservations.drop(['Booking_ID', 'date_of_reservation', 'reservation_month', 'room_type', 'type_of_meal', 'lead_time'], axis= 1, inplace= True)

#booking status label encoding
label_encoder= preprocessing.LabelEncoder()
hotel_reservations['booking_status'] = label_encoder.fit_transform(hotel_reservations['booking_status'])

#one-hot encoding of categorical features
hotel_reservations.rename(columns={'market_segment_type' : 'mst', 'reservation_season' : 'rs'}, inplace= True)
onehotencoder = preprocessing.OneHotEncoder(sparse_output= False)
ohe = onehotencoder.fit_transform(hotel_reservations[['mst', 'rs']])
ohe_df = pd.DataFrame(ohe, columns= onehotencoder.get_feature_names_out(['mst', 'rs']))
ohe_df.columns = [x.lower() for x in ohe_df.columns]
hs_encoded = pd.concat([hotel_reservations, ohe_df], axis= 1)
hs_encoded.drop(['mst', 'rs'], axis= 1, inplace= True)
reg_features = ['repeated', 'no_lead', 'mst_corporate', 'rs_winter', 'special_requests', 'mst_online', 'rs_summer', 'log_lead', 'average_price', 'total_individuals']
reg_encoded_hs = hs_encoded[reg_features]
reg_encoded_hs.to_csv('preprocessed.csv')