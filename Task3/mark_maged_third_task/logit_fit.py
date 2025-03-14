#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn import preprocessing
from sklearn. model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils.validation import check_is_fitted

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

#handling lead time outliers
hotel_reservations['log_lead'] = np.log(hotel_reservations['lead_time'].replace(0, 1))

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
reg_encoded_hs = hs_encoded[['repeated', 'no_lead', 'mst_corporate', 'rs_winter', 'special_requests', 'mst_online', 'rs_summer', 'log_lead', 'average_price', 'total_individuals']]

#normalizing non-encoded features
scaler = preprocessing.StandardScaler()
non_scaled_feat = ['number_of_adults', 'number_of_children', 'number_of_weekend_nights', 
                    'number_of_week_nights', 'log_lead', 'P-C', 'P-not-C', 'average_price', 
                    'special_requests', 'total_nights', 'total_individuals']
scaled_data = scaler.fit_transform(hs_encoded[non_scaled_feat])
scaled_df = pd.DataFrame(scaled_data, columns= hs_encoded[non_scaled_feat].columns)
hs_encoded.drop(non_scaled_feat, axis= 1, inplace= True)
hs_scaled_encoded = pd.concat([hs_encoded, scaled_df], axis= 1)

#finding features that have a weak to moderate correlation with booking status
hs_corr = hs_scaled_encoded.corr()
postive_corr = hs_corr['booking_status'].loc[hs_corr['booking_status'].between(0.1, 0.99)].index.tolist()
print('Positvely correlated:', postive_corr)
negative_corr = hs_corr['booking_status'].loc[hs_corr['booking_status'] < -0.1].index.tolist()
print('Negatively correlated:', negative_corr)
corr = postive_corr + negative_corr

#setting target variable
hs_target = hs_scaled_encoded['booking_status']
hs_scaled_encoded.drop('booking_status', axis= 1, inplace= True)

#imbalanced data ratio
bs_count = hs_target.value_counts()
bs_ratio = bs_count[1] / bs_count[0]
bs_ratio

#multi collinearity analysis
reg_hs = hs_scaled_encoded[corr]
vif_data = pd.DataFrame()
vif_data['feature'] = corr
vif_data['vif'] = [variance_inflation_factor(reg_hs.values, i) for i in range(len(corr))]
vif_data

#weighted logistic regression
sk_folds = StratifiedKFold(n_splits= 5, shuffle= True, random_state= 36)
weights = {0: bs_ratio, 1: 1}
weighted_logit = LogisticRegression(random_state= 36, class_weight= weights)
scores = cross_val_score(weighted_logit, reg_hs, hs_target, cv= sk_folds)
print('Cross validation scores:', scores)
print('Average vross validation score:', round(scores.mean()*100, 2), '%')
weighted_y_pred = cross_val_predict(weighted_logit, reg_hs, hs_target, cv= sk_folds)
conf_mat = pd.DataFrame(confusion_matrix(hs_target, weighted_y_pred))
sns.heatmap(conf_mat, annot= True, cmap= 'Blues', fmt= ',d', xticklabels= ['canceled', 'not canceled'],
             yticklabels= ['canceled', 'not canceled']).set(title= 'weighted logit confusion matrix', xlabel= 'Predicted', ylabel= 'True')
plt.show()

#fitting and saving the model
weighted_logit.fit(reg_hs, hs_target)
check_is_fitted(weighted_logit)
pickle.dump(weighted_logit, open('logit.pkl', 'wb'))