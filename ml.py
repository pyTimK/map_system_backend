from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
import numpy as np
# create datasets
# Initialize empty lists for X and y

barangays_list = ["POBLACION", "BALAGTAS-BMA", "BANCA-BANCA", "CAINGIN", "CORAL NA BATO", "CRUZ NA DAAN", "DAGAT-DAGATAN", "DILIMAN I", "DILIMAN II", "CAPIHAN", "LIBIS", "LICO", "MAASIM", "MABALAS-BALAS", "MAGUINAO", "MARONQUILLO", "PACO", "PANSUMALOC", "PANTUBIG", "PASONG BANGKAL", "PASONG CALLOS", "PASONG INCHIC", "PINAC-PINACAN", "PULO", "PULONG BAYABAS", "SALAPUNGAN", "SAMPALOC", "SAN AGUSTIN", "SAN ROQUE", "SAPANG PAHALANG", "TALACSAN", "TAMBUBONG", "TUKOD", "ULINGAO"]
outputs_list = ["Residential", "Agricultural", "Commercial", "Industrial", "Special", "Mineral", "ED", "R4", "SX", "TZ", "SB", "male_population", "female_population"]

years_to_add = 5


def barangay_to_int(b: str):
  try:
    return barangays_list.index(b)
  except ValueError:
    return -1


def create_model(data_dict):
  X = []
  y = []

  largest_year = 0

  # Loop through the nested dictionary and flatten it
  for key1, value1 in data_dict.items():
      for key2, value2 in value1.items():
          for key3, value3 in value2.items():

              # Update the largest year
              if int(key2) > largest_year:
                largest_year = int(key2)

              # Extract the input features from the innermost dictionary
              x_values = [barangay_to_int(key1), int(key2), int(key3)]
              y_values = [(value3[key] if value3[key] != None else 0) for key in outputs_list if key in value3]

              # Append the input features and target to X and y
              X.append(x_values)
              y.append(y_values)
  
  X = np.array(X)
  y = np.array(y)

  # Print the shape of X and y
  # print("Shape of X:", X.shape)
  # print("Shape of y:", y.shape)

  # print("----- x")
  # print(X)
  # print("----- y")
  # print(y)

  # define model
  model = LinearRegression()

  # fit model
  model.fit(X, y)




  # make a prediction
  rows_to_predict = []
  for i in range(years_to_add):
    for j in range(len(barangays_list)):
      for k in range(1, 13):
        rows_to_predict.append([j, largest_year + i + 1, k])
    
  yhat = model.predict(rows_to_predict)
  # print(yhat)

  for i in range(len(yhat)):
    rounded_y = []
    for j in range(len(yhat[i])):
      if yhat[i][j] < 0:
        rounded_y.append(0)
      else:
        rounded_y.append(int(np.round(yhat[i][j])))
    
    y_dict = dict(zip(outputs_list, rounded_y))

    barangay_int, year_int, month_int = rows_to_predict[i]

    barangay_str = barangays_list[barangay_int]
    year_str = str(year_int)
    month_str = str(month_int)

    if barangay_str not in data_dict:
      data_dict[barangay_str] = {}
    
    if year_str not in data_dict[barangay_str]:
      data_dict[barangay_str][year_str] = {}

    data_dict[barangay_str][year_str][month_str] = y_dict
    

  # print(data_dict)
  # summarize prediction
  # y_dict = {outputs_list[i]: int(round(yhat[i])) for i in range(len(outputs_list))}



