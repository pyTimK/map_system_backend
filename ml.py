from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Dict, Any, TypedDict, List
from itertools import chain
# create datasets
# Initialize empty lists for X and y

#! TYPES
RawData = Dict[str, Any]
MonthData = Dict[str, RawData]
YearData = Dict[str, MonthData]
DataDict = Dict[str, YearData]

class InputModel(TypedDict):
    X: List[List[int]]
    y: List[List[float]]

InputModels = Dict[str, InputModel]



barangays_list = ["POBLACION", "BALAGTAS-BMA", "BANCA-BANCA", "CAINGIN", "CORAL NA BATO", "CRUZ NA DAAN", "DAGAT-DAGATAN", "DILIMAN I", "DILIMAN II", "CAPIHAN", "LIBIS", "LICO", "MAASIM", "MABALAS-BALAS", "MAGUINAO", "MARONQUILLO", "PACO", "PANSUMALOC", "PANTUBIG", "PASONG BANGKAL", "PASONG CALLOS", "PASONG INCHIC", "PINAC-PINACAN", "PULO", "PULONG BAYABAS", "SALAPUNGAN", "SAMPALOC", "SAN AGUSTIN", "SAN ROQUE", "SAPANG PAHALANG", "TALACSAN", "TAMBUBONG", "TUKOD", "ULINGAO"]
outputs_list = ["Residential", "Agricultural", "Commercial", "Industrial", "Special", "Mineral", "ED", "R4", "SX", "TZ", "SB", "male_population", "female_population"]

years_to_add = 5


#! BARANGAY TO INT
def barangay_to_int(b: str):
  try:
    return barangays_list.index(b)
  except ValueError:
    return -1

#! GET ROWS TO PREDICT
def get_rows_to_predict(largest_year: int) -> List[List[int]]:
  rows_to_predict = []
  for years_i in range(1, years_to_add + 1):
    for month_i in range(12):
      rows_to_predict.append([(largest_year + years_i) * 12 + month_i])

  return rows_to_predict

#! LARGEST YEAR IN THE DATA
def find_largest_year(data_dict: DataDict) -> int:
  largest_year = 0
  for yearData in data_dict.values():
      for yearKey in yearData.keys():
        # Update the largest year
        if int(yearKey) > largest_year:
          largest_year = int(yearKey)

  return largest_year


#! CONVERT TO INPUT MODELS
def convert_to_input_models(yearData: YearData) -> InputModels:
  input_models: InputModels = {}

  for yearKey, monthData in yearData.items():
    for monthKey, rawData in monthData.items():
      for outputKey, value in rawData.items():

        # Populate input_models
        if outputKey not in input_models:
          input_models[outputKey] = {
            "X": [[int(yearKey) * 12 + int(monthKey) - 1]],
            "y": [[value]]
          }
        else:
          input_models[outputKey]["X"].append([int(yearKey) * 12 + int(monthKey) - 1])
          input_models[outputKey]["y"].append([value])
  
  return input_models

#! GENERATE MODEL
def generate_model(input_model: InputModel) -> LinearRegression:
  X = np.array(input_model["X"])
  y = np.array(input_model["y"])
  
  model = LinearRegression()
  model.fit(X, y)

  return model


#! APPEND PREDICTED DATA TO DICT
def append_predicted_data_to_dict(barangayKey: str, outputKey: str, X_hat: List[List[int]], y_hat: List[List[float]], data_dict: DataDict):
  y_hat_flat: List[float] = list(chain.from_iterable(y_hat)) # Flatten the list
  y_hat_positive = [y if y > 0 else 0 for y in y_hat_flat] # Replace negative values with 0
  y_hat_rounded = [int(np.round(y)) for y in y_hat_positive] # Round the list

  X_hat_flat: List[int] = list(chain.from_iterable(X_hat)) # Flatten the list

  for i in range(len(X_hat_flat)):
    X = X_hat_flat[i]
    y = y_hat_rounded[i]

    yearKey = str(X // 12)
    monthKey = str((X % 12) + 1)

    if yearKey not in data_dict[barangayKey]:
      data_dict[barangayKey][yearKey] = {}
    
    if monthKey not in data_dict[barangayKey][yearKey]:
      data_dict[barangayKey][yearKey][monthKey] = {}

    data_dict[barangayKey][yearKey][monthKey][outputKey] = y


#! MAIN FUNCTION
def predict_barangay_data(data_dict: DataDict):
  # Initialize empty lists for X and y
  X = []
  y = []
  
  # Find largest year
  largest_year = find_largest_year(data_dict)


  # Create models and generate predictions
  for barangayKey, yearData in data_dict.items():
    # {Residential : {X: [[1331], [1332]] y: [[100], [200]]}, ...}
    input_models = convert_to_input_models(yearData)

    for outputKey, input_model in input_models.items():
      model = generate_model(input_model)
      X_hat = get_rows_to_predict(largest_year)
      y_hat = model.predict(np.array(X_hat))
      append_predicted_data_to_dict(barangayKey, outputKey, X_hat, y_hat, data_dict)

      # Print the shape of X and y
      # print("Shape of X:", X.shape)
      # print("Shape of y:", y.shape)
      # print("----- x")
      # print(X)
      # print("----- y")
      # print(y)



