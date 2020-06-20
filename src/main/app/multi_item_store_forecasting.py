#!/usr/bin/env python3

from main.utils.spark_utils import init_spark

from pyspark.sql.types import StructType, StructField, DateType, IntegerType
from pyspark.sql.functions import sum, max, col

from fbprophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from datetime import date
import time
import pandas as pd

import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)


def input_data_exploration(spark):
    """
    Read input data to perform time series analysis, data comprehension using some visualization
    :param spark: sparkSession
    :return: history_pd
    """

    # structure of the training data set
    train_schema = StructType([
      StructField('date', DateType()),
      StructField('store', IntegerType()),
      StructField('item', IntegerType()),
      StructField('sales', IntegerType())
      ])

    # read the training file into a DataFrame
    train_df = spark.read.csv(
      '../../../src/main/resources/kaggle/train.csv',
      header=True,
      schema=train_schema
      )

    stores = train_df.select('store').orderBy('store').distinct()

    distinct_items = train_df.select('item').distinct().count()
    print(f'Available data has {distinct_items} distinct items on stock')

    # Top 10 sales per store and item
    sale_items_per_store = train_df.groupBy(['store', 'item']).agg(sum('sales').alias('total_sale_items'))\
        .orderBy('total_sale_items', ascending=False)
    sale_items_per_store.show(10)

    # make the DataFrame available as query temporary view
    train_df.createOrReplaceTempView('train')

    sql_statement = '''
      SELECT
        store,
        item,
        CAST(date as date) as ds,
        SUM(sales) as y
      FROM train
      GROUP BY store, item, ds
      ORDER BY store, item, ds
      '''

    # assemble DataSet in Pandas DataFrame
    store_item_history_df = spark.sql(sql_statement)\
        .repartition(spark.sparkContext.defaultParallelism, ['store', 'item'])

    history_pd = store_item_history_df.toPandas()

    print(history_pd.describe())

    # drop any missing records
    history_pd = history_pd.dropna()
    print(history_pd.head())

    # Pivot data to represent each store
    stores_to_pivot = list(stores.toPandas()['store'])

    store_item_history_pivot = store_item_history_df.groupBy(['item', 'ds']) \
        .pivot('store', stores_to_pivot) \
        .agg(sum("y").alias("total_sale_items")
             )

    store_item_history_pivot.show()
    # |item| ds|  1|  2|  3|  4|  5|  6|  7|  8|  9| 10|

    store_item_history_pivot_pd = store_item_history_pivot.toPandas()

    # x = store_item_history_pivot_pd.select([c for c in store_item_history_pivot_pd.columns if c not in to_drop])

    store_item_history_plot = store_item_history_pivot_pd.drop('item', axis=1)
    print(store_item_history_plot.head())
    history_pd.pivot_table(index='ds', columns='store', values='y', aggfunc='sum').plot()
    plt.show()

    return history_pd


def build_forecast(history_pd, period_days=90):
    """
    Build a Forecast
    Before attempting to generate forecasts for individual combinations of stores and items, it might be helpful to
    build a single forecast for no other reason than to orient ourselves to the use of FBProphet.
    :param history_pd: Sales per date DataFrame for one store and one item.
    :param period_days
    :return: forecast data
    """

    # Our first step is to assemble the historical DataSet on which we will train the model
    # instantiate the model and set parameters
    model = Prophet(
        interval_width=0.95,
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative'
    )

    # fit the model to historical data
    model.fit(history_pd)

    '''Now that we have a trained model, let's use it to build a 90-day forecast:'''

    # define a dataset including both historical dates & 90-days beyond the last available date
    future_pd = model.make_future_dataframe(
        periods=period_days,
        freq='d',
        include_history=True
    )

    # predict over the dataset
    forecast_pd = model.predict(future_pd)

    print(forecast_pd)

    # How did our model perform? Here we can see the general and seasonal trends in our model presented as graphs:
    trends_fig = model.plot_components(forecast_pd)
    print(trends_fig)

    ''' 
    And here, we can see how our actual and predicted data line up as well as a forecast for the future,
    though we will limit our graph to the last year of historical data just to keep it readable: 
    '''
    predict_fig = model.plot(forecast_pd, xlabel='date', ylabel='sales')

    # adjust figure to display dates from last year + the 90 day forecast
    x_lim = predict_fig.axes[0].get_xlim()
    new_x_lim = (x_lim[1] - (180.0 + 365.0), x_lim[1] - 90.0)
    predict_fig.axes[0].set_xlim(new_x_lim)

    print(predict_fig)

    plt.show()

    return forecast_pd


def compute_metrics(history_pd, forecast_pd):
    """
    Calculate evaluation metrics over the predicted data over actual values in our set:
    :param history_pd:
    :param forecast_pd:
    :return: None
    """

    # get historical actuals & predictions for comparison
    actuals_pd = history_pd[history_pd['ds'] < date(2018, 1, 1)]['y']
    predicted_pd = forecast_pd[forecast_pd['ds'] < date(2018, 1, 1)]['yhat']

    # calculate evaluation metrics
    mae = mean_absolute_error(actuals_pd, predicted_pd)
    mse = mean_squared_error(actuals_pd, predicted_pd)
    rmse = sqrt(mse)

    # print metrics to the screen
    print('\n'.join(['MAE: {0}', 'MSE: {1}', 'RMSE: {2}']).format(mae, mse, rmse))


def main():
    """
    Main program to be executed
    :return:
    """
    spark = init_spark('forecasting')

    time_start_all = time.time()

    history_pd = input_data_exploration(spark)
    # forecast_pd = build_forecast(history_pd)
    # compute_metrics(history_pd, forecast_pd)

    time_finish_all = time.time()
    print(f'Computed time was : {time_finish_all - time_start_all} seconds')

    spark.stop()


if __name__ == '__main__':
    main()
