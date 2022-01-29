import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import pickle
from datetime import datetime as dte
from flask import Flask,request,render_template

import flask
app = Flask(__name__)

######################################################################################################

# READING DATA RELEVANT TO CLEANING,IMPUTATION,PRE-PROCESSING & FEATURIZATION
def get_relevant_files():
  """fetching data relevant to cleaning, imputation, pre-processing & featurization"""
  def get_file(f_path):
    """Takes path & returns the data present in that file"""
    data = pickle.load( open(f_path,"rb") )
    return data

  f_path = "drop_columns.pkl"
  drop_columns = get_file(f_path)

  f_path = "impute_columns_with_zero.pkl"
  impute_columns_with_zero = get_file(f_path)

  f_path = "impute_columns_with_OTHERS.pkl"
  impute_columns_with_OTHERS = get_file(f_path)

  f_path = "to_replace_dict.pkl"
  to_replace_dict = get_file(f_path)

  f_path = "train_columns_sequence.pkl"
  train_columns_sequence = get_file(f_path)

  f_path = "CUSTOM_ENCODING_DICT.pkl"
  CUSTOM_ENCODING_DICT = get_file(f_path)

  f_path = "set1.pkl"
  set1 = get_file(f_path)

  f_path = "set2.pkl"
  set2 = get_file(f_path)

  f_path = "set3.pkl"
  set3 = get_file(f_path)

  f_path = "set4.pkl"
  set4 = get_file(f_path)

  f_path = "set5.pkl"
  set5 = get_file(f_path)

  f_path = "pre_gap_dates.pkl"
  pre_gap_dates = get_file(f_path)

  f_path = "final_test_data_type_dict.pkl"
  final_test_data_type_dict = get_file(f_path)

  f_path = "cpu_gbdt_cls_model_newxgb.pkl"
  loaded_model_cls = get_file(f_path)
  f_path = "cpu_gbdt_reg_model_newxgb.pkl"
  loaded_model_reg = get_file(f_path)

  return drop_columns,impute_columns_with_zero,impute_columns_with_OTHERS,to_replace_dict,train_columns_sequence,CUSTOM_ENCODING_DICT,\
  set1,set2,set3,set4,set5,pre_gap_dates,final_test_data_type_dict,loaded_model_cls,loaded_model_reg

# GET ALL FILES
drop_columns,impute_columns_with_zero,impute_columns_with_OTHERS,to_replace_dict,train_columns_sequence,CUSTOM_ENCODING_DICT,\
set1,set2,set3,set4,set5,pre_gap_dates,final_test_data_type_dict,loaded_model_cls,loaded_model_reg=get_relevant_files()

# PIPELINE FOR PREDICTION
def predict_revenue(X):
  """Creates entire pipeline for predicting revenue for a test data point and returns the predicted revenue(s)"""
  # MAKING TEST COLUMNS SEQUENCE THE SAME AS IT WAS IN TRAIN.
  X = X[train_columns_sequence]
  #-----------------------------------------------------------------------------
  # DATA CLEANING.
  for col in X.columns:
    if col in drop_columns:
      X.drop(columns=[col],axis=1,inplace=True)
    elif col in impute_columns_with_zero:
      X[col].fillna(value=0,inplace=True)
    elif col in impute_columns_with_OTHERS:
      X[col].fillna(value="OTHERS",inplace=True)
      X[col] = X[col].replace(to_replace=to_replace_dict)
  #-----------------------------------------------------------------------------
  # CONVERTING DATA COLUMN INTO DATETIME OBJECT.
  X['date'] = pd.to_datetime(arg=X['date'], format="%Y%m%d")
  #-----------------------------------------------------------------------------
  # EXTRACTING USEFUL FEATURES FROM DATE.
  X["year"] = X["date"].dt.year
  X["month"] = X["date"].dt.month
  X["dayOfMonth"] = X["date"].dt.day
  X["dayOfWeek"] = X["date"].dt.dayofweek
  X["dayName"] = X["date"].dt.day_name()
  X["weekOfYear"] = X["date"].dt.isocalendar().week
  X["dayOfYear"] = X["date"].dt.dayofyear
  X["quarter"] = X["date"].dt.quarter
  X["dayOfYear"] = X["date"].dt.dayofyear
  #-----------------------------------------------------------------------------
  # DESIGN NEW FEATURES USING DATE.
  def detectYearEnd(x):
    """Takes in date and detects whether it is year end (Oct-Dec) or not"""
    if x.month in [10,11,12]:
      return "Yes"
    else:
      return "No"
  X["isYearEnd"] = X["date"].apply(detectYearEnd)
  #-----------------------------------------------------------------------------
  # DESIGN NEW FEATURES USING DATE.
  def check_weekend(x):
    """Takes in a day name and detects whether it is weekend or not"""
    if x in ["Saturday","Sunday"]:
      return "Yes"
    else:
      return "No"
  X["is_weekend"] = X["dayName"].apply(check_weekend)
  #-----------------------------------------------------------------------------
  # DESIGN NEW FEATURES USING VISIT START TIME.
  def getFeaures_fromPOSIXtimestamp(x):
    """Takes time and returns timestamp in current format."""
    return dte.fromtimestamp(x)
  def getHr_fromPOSIX(x):
    """Returns hour from timestamp"""
    return x.hour
  def getMin_fromPOSIX(x):
    """Returns minute from timestamp"""
    return x.minute
  def getSec_fromPOSIX(x):
    """Returns second from timestamp"""
    return x.second
  def createFeaturesFromVisitStartTime(df):
    """Returns the dataframe with newly formed features using visitStartTime"""
    whole_timestamp = df["visitStartTime"].apply(getFeaures_fromPOSIXtimestamp)
    df["visit_hr"] = whole_timestamp.apply(getHr_fromPOSIX)
    df["visit_min"] = whole_timestamp.apply(getMin_fromPOSIX)
    df["visit_sec"] = whole_timestamp.apply(getSec_fromPOSIX)
    return df
  X = createFeaturesFromVisitStartTime(X)
  #-----------------------------------------------------------------------------
  # DROPPING COLUMN NO LONGER NEEDED.
  X.drop(columns=["visitStartTime"],axis=1,inplace=True)
  #-----------------------------------------------------------------------------
  # DESIGN NEW FEATURES USING TIME.
  def return_TimeOfDay(x):
    """Takes in hour of the day and detects time of day (midnight/morning/afternoon/evening/night)"""
    if x<6:
      return "midnight(12am-6am)"
    elif x>=6 and x<12:
      return "morning(6am-12pm)"
    elif x>=12 and x<18:
      return "afternoon_evening(12pm-6pm)"
    else:
      return "night(6pm-12am)"
  X["time_of_day"] = X["visit_hr"].apply(return_TimeOfDay)
  #-----------------------------------------------------------------------------
  # DESIGN NEW FEATURES USING TIME.
  def detect_office_hours(x):
    """Takes in hour of day; returns 'Yes' if it is office hours (i.e. 9am to 5pm) else returns 'No'"""
    if x>=9 and x<=17:
      return "Yes"
    else:
      return "No"
  X["is_office_hours"] = X["visit_hr"].apply(detect_office_hours)
  #-----------------------------------------------------------------------------
  # PROCESSING A BOOLEAN FEATURE.
  def detectMobile(x):
    """Returns 'Yes' if it is mobile, else 'No'"""
    if x==True:
      return "Yes"
    else:
      return "No"
  X['device_isMobile'] = X['device_isMobile'].apply(detectMobile)
  #-----------------------------------------------------------------------------
  # DROPPING COLUMN NO LONGER NEEDED.
  X.drop(columns=["dayName"],axis=1,inplace=True)
  #-----------------------------------------------------------------------------
  # PROCESSING YEAR COLUMN.
  X['year'] = X['year'].astype("object")
  #-----------------------------------------------------------------------------
  # CUSTOM CATEGORICAL ENCODING.
  def perform_custom_encoding(x,feature_specfic_dict):
    """This is like transform method. It will take a value -->x and encode it using the feature-specific dictionary passed here."""
    if x in feature_specfic_dict.keys():
      return feature_specfic_dict[x]
    else:
      return 0 # this 0 is for those feature values which wasn't seen during the fit method in train dataset
  for feature in CUSTOM_ENCODING_DICT.keys():
    X[feature] = X[feature].apply(perform_custom_encoding, args=(CUSTOM_ENCODING_DICT[feature],))
  #-----------------------------------------------------------------------------
  # GETTING VISITOR-LEVEL DATA VIA HELPER FUNCTIONS.
  def aggApply_ModeMax(x):
    """
    eg-1: For browser feature:if customer visited 3 times via Chrome and never via any other browser, then return 3.
    eg-2: For weekend feature: if customer visited 2 times on weekdays and 2 times on weekends, i.e. we'll have two 1s and two 0s --> then return 1.
          Max is taken for this kind of case just to select 1 value. Note: Taking max/min won't affect the actual mode obtained.
    """
    return x.mode().max()
  def agg_apply_Median(x):
    """return median of all values"""
    return x.median()
  def agg_apply_Mean(x):
    """return mean of all values"""
    return x.mean()
  def agg_apply_Sum(x):
    """return sum of all values"""
    return x.mean()
  def agg_apply_Min(x):
    """return minimum of all values"""
    return x.min()
  def agg_apply_Max(x):  
    """return max of all values"""
    return x.max()
  def agg_apply_LogOfSum(x):
    """return log of (sum of all values + 1). NOTE: adding 1 just to handle 0 sum."""
    return np.log1p(np.sum(x))
  def agg_apply_SpanVisits(x):
    """returns no. of days in between 1st visit & last visit in this particular aggregated time-frame"""
    return (x.max() - x.min()).days
  def get_apply_dict(df):
    """Returns a dictionary with feature name and list of tuples of new aggregated feature name & aggregation functions to apply."""
    apply_dict = {}
    for col in df.columns:
      if (col in set1) or (col in set2):
        new_feature_name = col+"_ModeMax"
        apply_dict[col] = [(new_feature_name, aggApply_ModeMax)]
      if col in set3:
        new_feature_name1 = col+"_Median"
        new_feature_name2 = col+"_Mean"
        new_feature_name3 = col+"_Sum"
        new_feature_name4 = col+"_Min"
        new_feature_name5 = col+"_Max"
        apply_list = [(new_feature_name1, agg_apply_Median), (new_feature_name2, agg_apply_Mean),
                      (new_feature_name3, agg_apply_Sum), (new_feature_name4, agg_apply_Min),
                      (new_feature_name5, agg_apply_Max)]
        if col in apply_dict.keys(): # checking if this col is already present in the dict
          apply_dict[col].extend(apply_list)
        else:
          apply_dict[col] = apply_list
      if col in set4:
        new_feature_name1 = col+"_Span"
        new_feature_name2 = col+"_FirstVisit"
        new_feature_name3 = col+"_LastVisit"
        apply_dict[col] = [(new_feature_name1, agg_apply_SpanVisits),(new_feature_name2, "min"),
                          (new_feature_name3, "max")]
      if col in set5:
        new_feature_name = col+"_LogOfSum"
        apply_dict[col] = [(new_feature_name, agg_apply_LogOfSum)]
    return apply_dict
  def vistor_level_data(data):
    data = data.groupby(by="fullVisitorId").agg(get_apply_dict(data))
    return data
  X = vistor_level_data(X)
  # dropping the multi-index level, resetting index and set fullVisitorId as the 1st column.
  X.columns = X.columns.droplevel()
  X = X.reset_index()
  #-----------------------------------------------------------------------------
  # MORE NEW FEATURES.
  def firstVisitAfterStart(x,start):
    """No. of days after current period's start date, the First visit occurred"""
    return (x - start).days
  def lastVisitBeforeEnd(x,end):
    """No. of days before current period's end date, the Last visit occurred"""
    return (end - x).days
  def apply_to_splits(split_df,current_period_start,current_period_end):
    """Applies function to 2 date columns and returns the modified dataframe"""
    split_df["firstVisit_AfterStart"] = split_df["date_FirstVisit"].apply(firstVisitAfterStart, args=(current_period_start,))
    split_df["lastVisit_BeforeEnd"] = split_df["date_LastVisit"].apply(lastVisitBeforeEnd, args=(current_period_end,))
    split_df = split_df.drop(columns=["date_FirstVisit","date_LastVisit"], axis=1)
    return split_df
  X = apply_to_splits(X, pre_gap_dates[0], pre_gap_dates[1])
  #-----------------------------------------------------------------------------
  # MAKING SURE DATA TYPES ARE SAME AS IT WAS WHILE TRAINING.
  X = X.astype(final_test_data_type_dict)
  #-----------------------------------------------------------------------------
  #PREDICTIONS USING SAVED MODELS.
  test_columnsToDrop = ["fullVisitorId","totals_transactionRevenue_LogOfSum"]
  X = X.drop(columns=test_columnsToDrop,axis=1)
  cls_prediction = loaded_model_cls.predict_proba(X)[:,1]
  reg_prediction = loaded_model_reg.predict(X)
  final_prediction = cls_prediction*reg_prediction
  return final_prediction

######################################################################################################

@app.route('/', methods=['GET'])
def welcome():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        queryfile=request.files['query-file']
        if queryfile:
            q = pd.read_csv(queryfile)
            return render_template('index.html',prediction=predict_revenue(q))    

if __name__ == "__main__":
	app.run(host='0.0.0.0',port=8080)