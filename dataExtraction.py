import csv
import pandas as pd

# Read the CSV file with headers
dfr1 = pd.read_csv("/content/ResaleFlatPricesBasedonApprovalDate19901999.csv")

# Initialize the columns dictionary
columns = {"Year & month":[],"Town":[],"Room":[],"Block":[],"Street Name":[],"Storey Range":[],
           "Floor Area SQM":[],"Flat Model":[],"Lease Commence Date":[],"Resale Price":[]}

# Iterate through the DataFrame and extract
for index, row in dfr1.iterrows():
  date = row["month"]
  town = row["town"]
  room = row["flat_type"]
  block = row["block"]
  street = row["street_name"]
  storey = row["storey_range"]
  floor = row["floor_area_sqm"]
  model = row["flat_model"]
  lease = row["lease_commence_date"]
  price = row["resale_price"]
  columns["Year & month"].append(str(date+ "-01"))
  columns["Town"].append(town)
  columns["Room"].append(room)
  columns["Block"].append(block)
  columns["Street Name"].append(street)
  columns["Storey Range"].append(storey)
  columns["Floor Area SQM"].append(floor)
  columns["Flat Model"].append(model)
  columns["Lease Commence Date"].append(lease)
  columns["Resale Price"].append(price)

dfr2 = pd.read_csv("/content/ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv")

for index, row in dfr2.iterrows():
  date = row["month"]
  town = row["town"]
  room = row["flat_type"]
  block = row["block"]
  street = row["street_name"]
  storey = row["storey_range"]
  floor = row["floor_area_sqm"]
  model = row["flat_model"]
  lease = row["lease_commence_date"]
  price = row["resale_price"]
  columns["Year & month"].append(str(date+ "-01"))
  columns["Town"].append(town)
  columns["Room"].append(room)
  columns["Block"].append(block)
  columns["Street Name"].append(street)
  columns["Storey Range"].append(storey)
  columns["Floor Area SQM"].append(floor)
  columns["Flat Model"].append(model)
  columns["Lease Commence Date"].append(lease)
  columns["Resale Price"].append(price)

dfr3 = pd.read_csv("/content/ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv")

for index, row in dfr3.iterrows():
  date = row["month"]
  town = row["town"]
  room = row["flat_type"]
  block = row["block"]
  street = row["street_name"]
  storey = row["storey_range"]
  floor = row["floor_area_sqm"]
  model = row["flat_model"]
  lease = row["lease_commence_date"]
  price = row["resale_price"]
  columns["Year & month"].append(str(date+ "-01"))
  columns["Town"].append(town)
  columns["Room"].append(room)
  columns["Block"].append(block)
  columns["Street Name"].append(street)
  columns["Storey Range"].append(storey)
  columns["Floor Area SQM"].append(floor)
  columns["Flat Model"].append(model)
  columns["Lease Commence Date"].append(lease)
  columns["Resale Price"].append(price)

dfr4 = pd.read_csv("/content/ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv")

for index, row in dfr3.iterrows():
  date = row["month"]
  town = row["town"]
  room = row["flat_type"]
  block = row["block"]
  street = row["street_name"]
  storey = row["storey_range"]
  floor = row["floor_area_sqm"]
  model = row["flat_model"]
  lease = row["lease_commence_date"]
  price = row["resale_price"]
  columns["Year & month"].append(str(date+ "-01"))
  columns["Town"].append(town)
  columns["Room"].append(room)
  columns["Block"].append(block)
  columns["Street Name"].append(street)
  columns["Storey Range"].append(storey)
  columns["Floor Area SQM"].append(floor)
  columns["Flat Model"].append(model)
  columns["Lease Commence Date"].append(lease)
  columns["Resale Price"].append(price)

dfr5 = pd.read_csv("/content/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv")

for index, row in dfr3.iterrows():
  date = row["month"]
  town = row["town"]
  room = row["flat_type"]
  block = row["block"]
  street = row["street_name"]
  storey = row["storey_range"]
  floor = row["floor_area_sqm"]
  model = row["flat_model"]
  lease = row["lease_commence_date"]
  price = row["resale_price"]
  columns["Year & month"].append(str(date)+ "-01")
  columns["Town"].append(town)
  columns["Room"].append(room)
  columns["Block"].append(block)
  columns["Street Name"].append(street)
  columns["Storey Range"].append(storey)
  columns["Floor Area SQM"].append(floor)
  columns["Flat Model"].append(model)
  columns["Lease Commence Date"].append(lease)
  columns["Resale Price"].append(price)


# Create a new DataFrame from the dictionary and clean
singapoor_flat_df = pd.DataFrame(columns)
singapoor_flat_df = singapoor_flat_df.drop_duplicates()
singapoor_flat_df = singapoor_flat_df.dropna()


  #              OR                      *******

# Load the CSV files into DataFrames
df1 = pd.read_csv('/content/ResaleFlatPricesBasedonApprovalDate19901999.csv')
df2 = pd.read_csv('/content/ResaleFlatPricesBasedonApprovalDate2000Feb2012.csv')
df3 = pd.read_csv('/content/ResaleFlatPricesBasedonRegistrationDateFromMar2012toDec2014.csv')
df4 = pd.read_csv('/content/ResaleFlatPricesBasedonRegistrationDateFromJan2015toDec2016.csv')
df5 = pd.read_csv('/content/ResaleflatpricesbasedonregistrationdatefromJan2017onwards.csv')

# Concatenate the DataFrames
data = pd.concat([df1, df2, df3, df4, df5], ignore_index=True)   # BUT IT THROW SOME ERROR BECOUSE COLOUMNS NOT MATCH

#  SO

data1 = pd.concat([df1, df2, df3], ignore_index=True)
data2 = pd.concat([df4, df5], ignore_index=True) # DF 4 AND 5 HAVE ADDITIONAL ONE COLOUMN

data3 = pd.concat([data1,data2], ignore_index=True)   # IN THIS CODE THAT ADDITIONAL COLOUMN SET AS NUL IF IT HAS NO VALUE
