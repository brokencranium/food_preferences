# Databricks notebook source
import seaborn as sns
import matplotlib.pyplot as plt
import math 
from datetime import timedelta, datetime
from functools import reduce
from operator import add

import numpy as np
from numpy import linalg as LA

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, concat, sum, when, desc, asc, sqrt, abs
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import DenseVector
from pyspark.ml.linalg import DenseMatrix
from pyspark.ml.feature import Normalizer
# from pyspark.sql import Row

import pandas as pd
from pandas.plotting import parallel_coordinates

from sklearn.preprocessing import MinMaxScaler

# import networkx as nx
# from pyspark.mllib.linalg.distributed import RowMatrix, IndexedRow, IndexedRowMatrix

# COMMAND ----------

dbutils.widgets.removeAll()

# COMMAND ----------


dbutils.widgets.text(name="start-date", defaultValue="2019-01-01", label="Start Date")
dbutils.widgets.text(name="end-date", defaultValue="2019-12-31", label="End Date")
dbutils.widgets.text(name="step-days", defaultValue="30", label="Step size of days")

start_date_str = dbutils.widgets.get("start-date")
end_date_str = dbutils.widgets.get("end-date")
step_size_days = int(dbutils.widgets.get("step-days"))

start_date =  datetime.strptime(start_date_str, "%Y-%m-%d")
end_date =  datetime.strptime(end_date_str, "%Y-%m-%d") 


# COMMAND ----------

dates_list = [(start_date + timedelta(n)).date() for n in range(step_size_days,int ((end_date - start_date).days), step_size_days) ]
print(dates_list)

# COMMAND ----------

def plot_corr_matrix(correlations, attr, fig_no):
    fig=plt.figure(fig_no, figsize=(10,10))
    cmap = sns.diverging_palette(220, 120, as_cmap=True)
    sns.set(style="white")
    sns.heatmap(correlations, cmap=cmap, vmax=.3, center=0, xticklabels= attr, yticklabels=attr,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    display(fig)

# COMMAND ----------

stores_df = (spark
             .read.table("prodedap_pub_store.store_edw_restricted")
             .filter(col('StoreOwnershipTypeCode') == 'CO')
             .filter(col('StoreStatusCode') == 'ACTIVE')
#              .sample(fraction=0.01, seed=1).limit(1)
             .filter(col('StoreNumber') == 19311)
            )

# COMMAND ----------

# MAGIC %md 
# MAGIC Product Type ID  
# MAGIC 20 - Food - 48   
# MAGIC 10 - Beverage

# COMMAND ----------

prod_df = ( spark
           .read
           .table("prodedap_pub_productitem.enterprise_product_hierarchy")
           .select("ProductStyleDescription", "ItemID")
           .filter(col("ProductTypeId").isin([20]))
          )

item_df =  ( spark
              .read
              .table("prodedap_pub_customersales.pos_item_detail")
              .filter(col("SalesTransactionUTCDate").between(start_date, end_date))
              .filter(col("CustomerTransactionInd") == 1)
              .filter(col("ChildLineSequenceNumber") == 0)
              .filter(col("RevenueItemFlag") == 'Y')
              .filter(col("VoidFlag") == 'N')
              .select("SalesTransactionId", "SalesTransactionUTCDate", "StoreNumber", 
                      "ItemID", "ItemQuantity", "ItemSaleAmount")
             )
item_store_df = item_df.join(stores_df, ["StoreNumber"], "inner")

item_prod_df = (item_store_df
                .join(prod_df, "ItemID", "inner")
                .groupBy('StoreNumber', "SalesTransactionUTCDate")
                .pivot("ProductStyleDescription")
                .agg(
                      sum("ItemQuantity").alias("ItemQuantity"), 
    #                       sum("ItemSaleAmount").alias("ItemSaleAmount")
                    )
                .na.fill(0)
           )


# COMMAND ----------

corr_columns = [ name for name in item_prod_df.schema.names[2:]]
feature_names = [ name.replace(' ', '') for name in corr_columns]

vec_assembler = VectorAssembler(inputCols=corr_columns,outputCol="Items")
vector_items_df = (vec_assembler
                 .transform(item_prod_df)
                 .select("StoreNumber", "SalesTransactionUTCDate", "Items")
#                    .orderBy([col('StoreNumber'), col('SalesTransactionUTCDate')])
                )


# COMMAND ----------

# MAGIC %md 
# MAGIC ##### POS Data as random matrix

# COMMAND ----------

# items_corr_df.collect()[0]["pearson({})".format("Items")].values
eigen_dates = []

for index, to_date in enumerate(dates_list):
    print('Data processing upto: {}'.format(to_date))
    vectors_slice_df = (vector_items_df
                        .filter(col('SalesTransactionUTCDate').between(start_date, to_date))
                     )
    items_corr_df = Correlation.corr(vectors_slice_df, "Items")
    
    corr_result_mat = items_corr_df.collect()[0][0]
    corr_list = corr_result_mat.toArray().tolist()
    corr_clean_list = [ [0.00 if math.isnan(corr_col) else corr_col for corr_col in corr_row] for corr_row in corr_list]
    
    eigen_vals, eigen_vectors = LA.eig(corr_clean_list)
    eigen_list = (np.concatenate((np.full((eigen_vectors.shape[0], 1), to_date.strftime('%Y-%m-%d')), 
                                 eigen_vals.reshape(eigen_vals.shape[0],1), 
                                 eigen_vectors), 
                  axis=1).tolist()
                  )
    
    eigen_dates = eigen_dates + eigen_list

eigen_df = spark.createDataFrame(eigen_dates)


# COMMAND ----------

eigen_names_df =  (eigen_df
                  .withColumn("DatesAlpha", col("_1").cast("date"))
                  .withColumn('EVal', col('_2').cast(DoubleType()))
                  .drop('_1', '_2')
                 )

for str_col, col_name in zip(eigen_names_df.columns, feature_names):
  eigen_names_df =  (eigen_names_df
                     .withColumn(col_name + 'EVector', col(str_col).cast(DoubleType()))
                     .drop(str_col)
              )  


# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Eigen vectors will sum to 1 if sign is ignored

# COMMAND ----------

# spark.sql('Drop table FoodPillars')
eigen_names_df.write.saveAsTable('FoodPillars')
# eigen_names_df = spark.read.table('FoodPillars')

# COMMAND ----------

# MAGIC %sql 
# MAGIC select DatesAlpha, max(EVal) from FoodPillars group by DatesAlpha order by DatesAlpha

# COMMAND ----------

eigen_names_df = eigen_names_df.toDF(* ['DatesAlpha', 'EVal'] + feature_names)
eigen_names_pd = eigen_names_df.toPandas().sort_values(["DatesAlpha", "EVal"])
eigen_names_pd['DatesAlpha'] = pd.to_datetime(eigen_names_pd['DatesAlpha'], format='%Y-%m-%d')

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Plot max eigen values for every t. t is inclusive of time previous timeframes

# COMMAND ----------

eigen_max_eval_pd = eigen_names_pd.groupby(['DatesAlpha']).agg({'EVal':'max'})
eigen_max_eval_pd.plot()
plt.ylabel('Eigen Value')
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Plot sorted by desc eigen values for 2019-11-27 to 2019-12-27. 

# COMMAND ----------

eigen_val_12_27_pd = eigen_names_pd[(eigen_names_pd['DatesAlpha'] == pd.Timestamp(2019, 12, 27))][['EVal']]
eigen_val_12_27_sort_pd = eigen_val_12_27_pd.sort_values(['EVal']).reset_index(drop=True)
eigen_val_12_27_sort_pd.plot()
plt.ylabel('Eigen Value')
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Plot abs of eigen vectors for 2019-11-27 to 2019-12-27 (Markov Transition Matrix)

# COMMAND ----------

eigen_12_27_pd = eigen_names_pd[(eigen_names_pd['DatesAlpha'] == pd.Timestamp(2019, 12, 27))]
eigen_12_27_pd = eigen_12_27_pd.drop(columns=['DatesAlpha', 'EVal'])
eigen_12_27_abs_pd = eigen_12_27_pd.abs()

# COMMAND ----------

eigen_12_27_abs_pd.plot.area(stacked=True)
plt.xticks([])
plt.xlabel('Features')
plt.legend(loc=(1.03,0),labels=feature_names)
plt.subplots_adjust(right=0.7)
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Plot scaled eigen vectors for 2019-11-27 to 2019-12-27

# COMMAND ----------

scaler = MinMaxScaler()
eigen_12_27_pd = eigen_names_pd[(eigen_names_pd['DatesAlpha'] == pd.Timestamp(2019, 12, 27))]
eigen_12_27_pd = eigen_12_27_pd.drop(columns=['DatesAlpha', 'EVal'])
eigen_12_27_scaled_pd = pd.DataFrame(scaler.fit_transform(eigen_12_27_pd))
eigen_12_27_scaled_pd.plot.area()
plt.xticks([])
plt.xlabel('Features')
plt.legend(loc=(1.03,0),labels=feature_names)
plt.subplots_adjust(right=0.7)
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Plot scaled eigen vectors based on max eigen values for every t. t is inclusive of time previous timeframes

# COMMAND ----------

scaler = MinMaxScaler()
eigen_max_eval_pd = eigen_names_pd.loc[eigen_names_pd.groupby(['DatesAlpha'])['EVal'].idxmax()]
eigen_max_eval_pd = eigen_max_eval_pd.drop(columns=['DatesAlpha', 'EVal'])
eigen_max_scaled_pd = pd.DataFrame(scaler.fit_transform(eigen_max_eval_pd))
eigen_max_scaled_pd.plot.area()
plt.xticks([])
plt.xlabel('Features')
plt.legend(loc=(1.03,0),labels=feature_names)
plt.subplots_adjust(right=0.7)
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Stochiometric Matrix for t 2019-11-27 to 2019-12-27

# COMMAND ----------

eigen_12_27_1_pd = eigen_names_pd[(eigen_names_pd['DatesAlpha'] == pd.Timestamp(2019, 12, 27))]
eigen_12_27_1_pd = eigen_12_27_1_pd.drop(columns=['DatesAlpha', 'EVal'])
eigen_12_27_1_pd[eigen_12_27_1_pd[feature_names] < 0 ] = -1
eigen_12_27_1_pd[eigen_12_27_1_pd[feature_names] > 0 ] = 1
eigen_12_27_1_pd

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Stochiometric matrix based on max eigen values for every t. t is inclusive of time previous timeframes

# COMMAND ----------

eigen_max_eval_pd = eigen_names_pd.loc[eigen_names_pd.groupby(['DatesAlpha'])['EVal'].idxmax()]
eigen_max_eval_pd = eigen_max_eval_pd.drop(columns=['DatesAlpha', 'EVal'])
eigen_max_eval_pd[eigen_max_eval_pd[feature_names] < 0 ] = -1
eigen_max_eval_pd[eigen_max_eval_pd[feature_names] > 0 ] = 1
eigen_max_eval_pd
