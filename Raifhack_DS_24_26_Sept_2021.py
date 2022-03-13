import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score


df_train0 = pd.read_csv('INPUT/train.csv')
df_test_sub0 = pd.read_csv('INPUT/test.csv')
result = pd.DataFrame(df_test_sub0['id'])

pred_res = []

def df_train_region(df, region, city, realty_type):
    df_train = df.loc[df['region'] == region]
    if realty_type in df_train['realty_type']:
        df_train = df_train.loc[df_train['realty_type'] == realty_type]
    else: 
        df_train = df_train.loc[df_train['realty_type'] == 10]
    if city in df_train['city']:
        df_train = df_train.loc[df_train['city'] == city]
    else:
        df_train = df_train
    return df_train

i = 0
print(i)
while i < 2974:
    df_test_sub = df_test_sub0.iloc[i:i+1]
    #id = city = df_test_sub['id'][i]
    city = df_test_sub['city'][i]
    region = df_test_sub['region'][i]
    realty_type = df_test_sub['realty_type'][i]
    print(city)
    print(region)
    print(realty_type)

    df_train = df_train_region(df_train0, region, city, realty_type)

    df_train1 = df_train.drop(['lat', 'lng', 'osm_amenity_points_in_0.001', 'osm_amenity_points_in_0.0075', 
                            'osm_amenity_points_in_0.01', 'osm_building_points_in_0.001', 'osm_building_points_in_0.005', 'osm_building_points_in_0.0075',
                            'osm_building_points_in_0.01', 'osm_catering_points_in_0.001', 'osm_catering_points_in_0.005', 'osm_catering_points_in_0.0075',
                            'osm_catering_points_in_0.01', 'osm_city_closest_dist', 'osm_city_nearest_name', 'osm_city_nearest_population', 
                            'osm_crossing_closest_dist', 'osm_crossing_points_in_0.001', 'osm_crossing_points_in_0.005', 'osm_crossing_points_in_0.0075',
                            'osm_crossing_points_in_0.01', 'osm_culture_points_in_0.001', 'osm_culture_points_in_0.005', 'osm_culture_points_in_0.0075',
                            'osm_culture_points_in_0.01', 'osm_finance_points_in_0.001', 'osm_finance_points_in_0.005', 'osm_finance_points_in_0.0075',
                            'osm_finance_points_in_0.01', 'osm_healthcare_points_in_0.005', 'osm_healthcare_points_in_0.0075', 'osm_healthcare_points_in_0.01',
                            'osm_historic_points_in_0.005', 'osm_historic_points_in_0.0075', 'osm_historic_points_in_0.01', 'osm_hotels_points_in_0.005',
                            'osm_hotels_points_in_0.0075', 'osm_hotels_points_in_0.01', 'osm_leisure_points_in_0.005', 'osm_leisure_points_in_0.0075',
                            'osm_leisure_points_in_0.01', 'osm_offices_points_in_0.001', 'osm_offices_points_in_0.005', 'osm_offices_points_in_0.0075',
                            'osm_offices_points_in_0.01', 'osm_shops_points_in_0.001', 'osm_shops_points_in_0.005', 'osm_shops_points_in_0.0075',
                            'osm_shops_points_in_0.01', 'osm_subway_closest_dist', 'osm_train_stop_closest_dist', 'osm_train_stop_points_in_0.005',
                            'osm_train_stop_points_in_0.0075', 'osm_train_stop_points_in_0.01', 'osm_transport_stop_closest_dist',
                            'osm_transport_stop_points_in_0.005', 'osm_transport_stop_points_in_0.0075', 'osm_transport_stop_points_in_0.01',
                            'reform_count_of_houses_1000', 'reform_count_of_houses_500', 'reform_house_population_1000', 'reform_house_population_500',
                            'reform_mean_floor_count_1000', 'reform_mean_floor_count_500', 'reform_mean_year_building_1000',
                            'reform_mean_year_building_500', 'street', 'date', 'id', 'realty_type', 'region', 'city',
                            'price_type', 'date_n', 'oil', 'osm_amenity_points_in_0.005', 'floor'], axis=1)


    df_test_sub1 = df_test_sub.drop(['lat', 'lng', 'osm_amenity_points_in_0.001', 'osm_amenity_points_in_0.0075', 
                            'osm_amenity_points_in_0.01', 'osm_building_points_in_0.001', 'osm_building_points_in_0.005', 'osm_building_points_in_0.0075',
                            'osm_building_points_in_0.01', 'osm_catering_points_in_0.001', 'osm_catering_points_in_0.005', 'osm_catering_points_in_0.0075',
                            'osm_catering_points_in_0.01', 'osm_city_closest_dist', 'osm_city_nearest_name', 'osm_city_nearest_population', 
                            'osm_crossing_closest_dist', 'osm_crossing_points_in_0.001', 'osm_crossing_points_in_0.005', 'osm_crossing_points_in_0.0075',
                            'osm_crossing_points_in_0.01', 'osm_culture_points_in_0.001', 'osm_culture_points_in_0.005', 'osm_culture_points_in_0.0075',
                            'osm_culture_points_in_0.01', 'osm_finance_points_in_0.001', 'osm_finance_points_in_0.005', 'osm_finance_points_in_0.0075',
                            'osm_finance_points_in_0.01', 'osm_healthcare_points_in_0.005', 'osm_healthcare_points_in_0.0075', 'osm_healthcare_points_in_0.01',
                            'osm_historic_points_in_0.005', 'osm_historic_points_in_0.0075', 'osm_historic_points_in_0.01', 'osm_hotels_points_in_0.005',
                            'osm_hotels_points_in_0.0075', 'osm_hotels_points_in_0.01', 'osm_leisure_points_in_0.005', 'osm_leisure_points_in_0.0075',
                            'osm_leisure_points_in_0.01', 'osm_offices_points_in_0.001', 'osm_offices_points_in_0.005', 'osm_offices_points_in_0.0075',
                            'osm_offices_points_in_0.01', 'osm_shops_points_in_0.001', 'osm_shops_points_in_0.005', 'osm_shops_points_in_0.0075',
                            'osm_shops_points_in_0.01', 'osm_subway_closest_dist', 'osm_train_stop_closest_dist', 'osm_train_stop_points_in_0.005',
                            'osm_train_stop_points_in_0.0075', 'osm_train_stop_points_in_0.01', 'osm_transport_stop_closest_dist',
                            'osm_transport_stop_points_in_0.005', 'osm_transport_stop_points_in_0.0075', 'osm_transport_stop_points_in_0.01',
                            'reform_count_of_houses_1000', 'reform_count_of_houses_500', 'reform_house_population_1000', 'reform_house_population_500',
                            'reform_mean_floor_count_1000', 'reform_mean_floor_count_500', 'reform_mean_year_building_1000',
                            'reform_mean_year_building_500', 'street', 'date', 'id', 'realty_type', 'region', 'city',
                                'date_n', 'oil', 'osm_amenity_points_in_0.005', 'floor', 'price_type'], axis=1)

    x = df_train1.drop('per_square_meter_price', axis=1)
    y = df_train1['per_square_meter_price']



    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)




    def cross_val(model):
        pred = cross_val_score(model, x, y, cv=10)
        return pred.mean()

    def print_evaluate(true, predicted):  
        mae = metrics.mean_absolute_error(true, predicted)
        mse = metrics.mean_squared_error(true, predicted)
        rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
        r2_square = metrics.r2_score(true, predicted)
        print('MAE:', mae)
        print('MSE:', mse)
        print('RMSE:', rmse)
        print('R2 Square', r2_square)
        print('__________________________________')
        
    def evaluate(true, predicted):
        mae = metrics.mean_absolute_error(true, predicted)
        mse = metrics.mean_squared_error(true, predicted)
        rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
        r2_square = metrics.r2_score(true, predicted)
        return mae, mse, rmse, r2_square


    pipeline = Pipeline([
        ('std_scalar', StandardScaler())
    ])

    x_train = pipeline.fit_transform(x_train)
    x_test = pipeline.transform(x_test)
    #df_test_sub1 = pipeline.transform(df_test_sub1)


    lin_reg = LinearRegression(normalize=True)
    lin_reg.fit(x_train,y_train)

    
    #print(lin_reg.intercept_)

    df_test_sub1 = pipeline.transform(df_test_sub1)
    pred = lin_reg.predict(df_test_sub1)
    i += 1
    res = pred[0]
    pred_res.append(res)
    with open('result.txt', 'a') as file:
        file.write(str(abs(res)) + '\n')
