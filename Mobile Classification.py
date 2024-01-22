# %%
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# %%
#Reading The Csv Data
data = pd.read_csv('D:/Study/Projects/Data Analytics Project/train.csv')
#Summarizing The Data
data.info() 

# %%
#Showing first 5 row
data.head(5)

# %%
#Showing last 5 columns
data.tail(5)

# %%
#Summarizing the numeric cols
data.describe()
#All Columns are numeric

# %%
#Check redundent data
data.duplicated()
data.drop_duplicates(inplace=True)
#Data has no duplicates to remove

# %%
# Checking number of cols & rows
data.shape

# %%
#Renaming columns
data.rename(columns={'battery_power':'BatteryPower' },inplace=True) # Total energy a battery can store in one time measured in mAh
data.rename(columns={'blue':'Bluetooth' },inplace=True) # Has bluetooth or not
data.rename(columns={'clock_speed':'ClockSpeed' },inplace=True) # speed at which microprocessor executes instructions
data.rename(columns={'dual_sim':'DualSim' },inplace=True) # Has dual sim support or not
data.rename(columns={'fc':'FrontCameraPixels' },inplace=True) # Front Camera mega pixels
data.rename(columns={'four_g':'four4G' },inplace=True) # Has 4G or not
data.rename(columns={'int_memory':'InternalMemory' },inplace=True) # Internal Memory in Gigabytes
data.rename(columns={'m_dep':'MobileDepth' },inplace=True) # Mobile Depth in cm
data.rename(columns={'mobile_wt':'MobileWeight' },inplace=True) # Weight of mobile phone
data.rename(columns={'n_cores':'NumberOfCores' },inplace=True) # Number of cores of processor
data.rename(columns={'pc':'PrimaryCamereaPixels' },inplace=True) # Primary Camera mega pixels
data.rename(columns={'px_height':'PixelHeight' },inplace=True) # Pixel Resolution Height
data.rename(columns={'px_width':'PixelWidth' },inplace=True) # Pixel Resolution Width
data.rename(columns={'ram':'Ram' },inplace=True) # Random Access Memory in Mega Bytes
data.rename(columns={'sc_h':'ScreenHeight' },inplace=True) # Screen Height of mobile in cm
data.rename(columns={'sc_w':'ScreenWidth' },inplace=True) # Screen Width of mobile in cm
data.rename(columns={'talk_time':'TalkTime' },inplace=True) # longest time that a single battery charge will last when you are
data.rename(columns={'three_g':'three3G' },inplace=True) # Has 3G or not
data.rename(columns={'touch_screen':'TouchScreen' },inplace=True) # Has touch screen or not
data.rename(columns={'wifi':'Wifi' },inplace=True) # Has wifi or not
data.rename(columns={'price_range':'PriceRange' },inplace=True) # This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).


# %%
data.head()

# %%
#applying anomlay detection using visulization with BoxPlot
plt.boxplot(data['BatteryPower'] , vert=False)
plt.title("BatteryPower")
plt.show()
#No outliers @ BatteryPower

# %%
plt.boxplot(data['Bluetooth'] , vert=False) 
plt.title("Bluetooth")
plt.show()
# No outliers @ bluetooth

# %%
plt.boxplot(data['ClockSpeed'] , vert=False) 
plt.title("ClockSpeed")
plt.show()
# No outliers in ClockSpeed

# %%
plt.boxplot(data['FrontCameraPixels'] , vert=False) 
plt.title("FrontCameraPixels")
plt.show()
#FrontCameraPixels has ouliers

# %%
#Printing postion of outliers
print(np.where(data['FrontCameraPixels'] > 16))

# %%
plt.boxplot(data['InternalMemory'] , vert=False) 
plt.title("InternalMemory")
plt.show()
# No outliers

# %%
plt.boxplot(data['MobileDepth'] , vert=False) 
plt.title("MobileDepth")
plt.show()
#No outliers

# %%
plt.boxplot(data['MobileWeight'] , vert=False)  
plt.title("MobileWeight") 
plt.show()
#No Outliers

# %%
plt.boxplot(data['NumberOfCores'] , vert=False)
plt.title("NumberOfCores")
plt.show() 
#No Outliers

# %%
plt.boxplot(data['PrimaryCamereaPixels'] , vert=False) 
plt.title("PrimaryCamereaPixels")
plt.show()
#No Outliers

# %%
plt.boxplot(data['PixelHeight'] , vert=False)
plt.title("PixelHeight")
plt.show()
#There are outliers @ PixelHeight

# %%
#Printing postion of outliers
print(np.where(data['PixelHeight'] > 1900))

# %%
plt.boxplot(data['PixelWidth'] , vert=False) 
plt.title("PixelWidth")
plt.show()
#No Outliers

# %%
plt.boxplot(data['Ram'] , vert=False) 
plt.title("Ram")
plt.show()
#No Outliers 

# %%
plt.boxplot(data['ScreenHeight'] , vert=False) 
plt.title("ScreenHeight")
plt.show()
#No Outliers

# %%
plt.boxplot(data['ScreenWidth'] , vert=False) 
plt.title("ScreenWidth")
plt.show()
#No outliers

# %%
plt.boxplot(data['TalkTime'] , vert=False) 
plt.title("TalkTime")
plt.show()
#No outliers

# %%
plt.boxplot(data['PriceRange'] , vert=False)
plt.title("PriceRange") 
plt.show()
# No ouliers

# %%
#find absolute value of z-score for each observation
z = np.abs(stats.zscore(data))

# %%
#only keep rows in dataframe with all z-scores less than absolute value of 2.75 
data_clean = data[(z<2.75).all(axis=1)]

# %%
#find how many rows are left in the dataframe 
data_clean.shape

# %%
#check outliers still exists
plt.boxplot(data_clean['PixelHeight'] , vert=False)
plt.title("PixelHeight After Removing Outliers")
plt.show()

# %%
plt.boxplot(data_clean['FrontCameraPixels'] , vert=False) 
plt.title("FrontCameraPixels After Removing Outliers")
plt.show()

# %%
 #--------------------------------------------------------
#There is a very big difference between the data and each other, and the data of these columns is considered dispersed

#variance of BatteryPower
BatteryPower_var = data_clean.BatteryPower.var()
print ("Battery Power var : ", BatteryPower_var)

#variance of InternalMemory
InternalMemory_var = data_clean.InternalMemory.var()
print ("Internal Memory var : ", InternalMemory_var)

#variance of MobileWeight
MobileWeight_var = data_clean.MobileWeight.var()
print ("Mobile Weight var : ", MobileWeight_var)

#variance of PixelHeight
PixelHeight_var = data_clean.PixelHeight.var()
print ("Pixel Height var : ", PixelHeight_var)

#variance of PixelWidth
PixelWidth_var = data_clean.PixelWidth.var()
print ("Pixel Width var : ", PixelWidth_var)

#variance of Ram
Ram_var = data_clean.Ram.var()
print ("Ram var : ", Ram_var)

# %%
#--------------------------------------------------
#The data are close to each other in these columns

#variance of Bluetooth
Bluetooth_var = data_clean.Bluetooth.var()
print ("Bluetooth  var : ", Bluetooth_var)

#variance of ClockSpeed
ClockSpeed_var = data_clean.ClockSpeed.var()
print ("Clock Speed var : ", ClockSpeed_var)

#variance of DualSim
DualSim_var = data_clean.DualSim.var()
print ("Dual Sim var : ", DualSim_var)

#variance of TouchScreen
TouchScreen_var = data_clean.TouchScreen.var()
print ("Touch Screen var : ", TouchScreen_var)

#variance of Wifi
Wifi_var = data_clean.Wifi.var()
print ("Wifi var : ", Wifi_var)

#variance of PriceRange
PriceRange_var = data_clean.PriceRange.var()
print ("Price Range var : ", PriceRange_var)

#variance of four4G
four4G_var = data_clean.four4G.var()
print ("four4G var : ", four4G_var)

#variance of three3G
three3G_var = data_clean.three3G.var()
print ("three3G var : ", three3G_var)

#variance of MobileDepth
MobileDepth_var = data_clean.MobileDepth.var()
print ("Mobile Depth var : ", MobileDepth_var)
#---------------------------------------------------------------

# %%
#variance of FrontCameraPixels
FrontCameraPixels_var = data_clean.FrontCameraPixels.var()
print ("Front Camera Pixels var : ", FrontCameraPixels_var)

#variance of NumberOfCores
NumberOfCores_var = data_clean.NumberOfCores.var()
print ("Number Of Cores var : ", NumberOfCores_var)

#variance of PrimaryCamereaPixels
PrimaryCamereaPixels_var = data_clean.PrimaryCamereaPixels.var()
print ("Primary Camerea Pixels var : ", PrimaryCamereaPixels_var)

#variance of ScreenHeight
ScreenHeight_var = data_clean.ScreenHeight.var()
print ("Screen Height var : ", ScreenHeight_var)

#variance of ScreenWidth
ScreenWidth_var = data_clean.ScreenWidth.var()
print ("Screen Width var : ", ScreenWidth_var)

#variance of TalkTime
TalkTime_var = data_clean.TalkTime.var()
print ("Talk Time var : ", TalkTime_var)

# %%
# mode for all columns in dataset
import statistics 

#mode of BatteryPower
BatteryPower_mod = statistics.mode(data_clean["BatteryPower"])
print ("Battery Power mode : ", BatteryPower_mod)

#mode of InternalMemory
InternalMemory_mod = statistics.mode(data_clean["InternalMemory"])
print ("Internal Memory mode : ", InternalMemory_mod)

#mode of MobileWeight
MobileWeight_mod = statistics.mode(data_clean["MobileWeight"])
print ("Mobile Weight mode : ", MobileWeight_mod)

#mode of PixelHeight
PixelHeight_mod = statistics.mode(data_clean["PixelHeight"])
print ("Pixel Height mode : ", PixelHeight_mod)

#mode of PixelWidth
PixelWidth_mod = statistics.mode(data_clean["PixelWidth"])
print ("Pixel Width mode : ", PixelWidth_mod)

#mode of Ram
Ram_mod = statistics.mode(data_clean["Ram"])
print ("Ram mode : ", InternalMemory_mod)

#mode of Bluetooth
Bluetooth_mod = statistics.mode(data_clean["Bluetooth"])
print ("Bluetooth mode : ", Bluetooth_mod)


# %%
#mode of ClockSpeed
ClockSpeed_mod = statistics.mode(data_clean["ClockSpeed"])
print ("Clock Speed mode : ", ClockSpeed_mod)

#mode of DualSim
DualSim_mod = statistics.mode(data_clean["DualSim"])
print ("Dual Sim mode : ", DualSim_mod)

#mode of TouchScreen
TouchScreen_mod = statistics.mode(data_clean["TouchScreen"])
print ("Touch Screen mode : ", TouchScreen_mod)

#mode of Wifi
Wifi_mod = statistics.mode(data_clean["Wifi"])
print ("Wifi mode : ", Wifi_mod)

#mode of PriceRange
PriceRange_mod = statistics.mode(data_clean["PriceRange"])
print ("Price Range mode : ", PriceRange_mod)

#mode of four4G
four4G_mod = statistics.mode(data_clean["four4G"])
print ("4G mode : ", four4G_mod)

#mode of three3G
three3G_mod = statistics.mode(data_clean["three3G"])
print ("3G mode : ", three3G_mod)


# %%
#mode of MobileDepth
MobileDepth_mod = statistics.mode(data_clean["MobileDepth"])
print ("Mobile Depth mode : ", MobileDepth_mod)

#mode of FrontCameraPixels
FrontCameraPixels_mod = statistics.mode(data_clean["FrontCameraPixels"])
print ("Front Camera Pixels mode : ", FrontCameraPixels_mod)

#mode of NumberOfCores
NumberOfCores_mod = statistics.mode(data_clean["NumberOfCores"])
print ("Number Of Cores mode : ", NumberOfCores_mod)

#mode of PrimaryCamereaPixels
PrimaryCamereaPixels_mod = statistics.mode(data_clean["PrimaryCamereaPixels"])
print ("Primary Camerea Pixels mode : ", PrimaryCamereaPixels_mod)

#mode of ScreenHeight
ScreenHeight_mod = statistics.mode(data_clean["ScreenHeight"])
print ("Screen Height mode : ", ScreenHeight_mod)

#mode of ScreenWidth
ScreenWidth_mod = statistics.mode(data_clean["ScreenWidth"])
print ("Screen Width mode : ", ScreenWidth_mod)

#mode of TalkTime
TalkTime_mod = statistics.mode(data_clean["TalkTime"])
print ("Talk Time mode : ", TalkTime_mod)


# %%
#There is a very big difference between the data and each other, and the data of these columns is considered dispersed

# Inter Quartile Range of BatteryPower 
BatteryPower_IQR = data_clean.BatteryPower.describe()['75%'] - data_clean.BatteryPower.describe()['25%']
print ("Battery Power IQR : ", BatteryPower_IQR)

# Inter Quartile Range of PixelHeight
PixelHeight_IQR = data_clean.PixelHeight.describe()['75%'] - data_clean.PixelHeight.describe()['25%']
print ("Pixel Height  IQR : ", PixelHeight_IQR)

# Inter Quartile Range of PixelWidth
PixelWidth_IQR = data_clean.PixelWidth.describe()['75%'] - data_clean.PixelWidth.describe()['25%'] 
print ("Pixel Width  IQR : ", PixelWidth_IQR)

# Inter Quartile Range of Ram
Ram_IQR = data_clean.Ram.describe()['75%'] - data_clean.Ram.describe()['25%']
print ("Ram  IQR : ", Ram_IQR)

# %%
#The data are close to each other

# Inter Quartile Range of Bluetooth 
Bluetooth_IQR = data_clean.Bluetooth.describe()['75%'] - data_clean.Bluetooth.describe()['25%']
print ("Bluetooth IQR : ", Bluetooth_IQR)

# Inter Quartile Range of ClockSpeed 
ClockSpeed_IQR = data_clean.ClockSpeed.describe()['75%'] - data_clean.ClockSpeed.describe()['25%']
print ("Clock Speed IQR : ", ClockSpeed_IQR)

# Inter Quartile Range of DualSim 
DualSim_IQR = data_clean.DualSim.describe()['75%'] - data_clean.DualSim.describe()['25%']
print ("DualSim IQR : ", DualSim_IQR)

# Inter Quartile Range of four4G 
four4G_IQR = data_clean.four4G.describe()['75%'] - data_clean.four4G.describe()['25%']
print ("four4G IQR : ", four4G_IQR)

# Inter Quartile Range of MobileDepth 
MobileDepth_IQR = data_clean.MobileDepth.describe()['75%'] - data_clean.MobileDepth.describe()['25%']
print ("Mobile Depth IQR : ", MobileDepth_IQR)

# Inter Quartile Range of TouchScreen
TouchScreen_IQR = data_clean.TouchScreen.describe()['75%'] - data_clean.TouchScreen.describe()['25%']
print ("TouchScreen  IQR : ", TouchScreen_IQR)

# Inter Quartile Range of Wifi
Wifi_IQR = data_clean.Wifi.describe()['75%'] - data_clean.Wifi.describe()['25%']
print ("Wifi  IQR : ", Wifi_IQR)

# Inter Quartile Range of Price Range
PriceRange_IQR = data_clean.PriceRange.describe()['75%'] - data_clean.PriceRange.describe()['25%']
print ("Price Range  IQR : ", PriceRange_IQR)

# %%
#The data is very similar

# Inter Quartile Range of three3G
three3G_IQR = data_clean.three3G.describe()['75%'] - data_clean.three3G.describe()['25%']
print ("three 3G  IQR : ", three3G_IQR)

# %%
# Inter Quartile Range of FrontCameraPixels 
FrontCameraPixels_IQR = data_clean.FrontCameraPixels.describe()['75%'] - data_clean.FrontCameraPixels.describe()['25%']
print ("Front Camera Pixels IQR : ", FrontCameraPixels_IQR)

# Inter Quartile Range of InternalMemory 
InternalMemory_IQR = data_clean.InternalMemory.describe()['75%'] - data_clean.InternalMemory.describe()['25%']
print ("Internal Memory IQR : ", InternalMemory_IQR)

# Inter Quartile Range of MobileWeight
MobileWeight_IQR = data_clean.MobileWeight.describe()['75%'] - data_clean.MobileWeight.describe()['25%']
print ("Mobile Weight IQR : ", MobileWeight_IQR)

# Inter Quartile Range of NumberOfCores
NumberOfCores_IQR = data_clean.NumberOfCores.describe()['75%'] - data_clean.NumberOfCores.describe()['25%']
print ("Number Of Cores IQR : ", NumberOfCores_IQR)

# Inter Quartile Range of PrimaryCamereaPixels
PrimaryCamereaPixels_IQR = data_clean.PrimaryCamereaPixels.describe()['75%'] - data_clean.PrimaryCamereaPixels.describe()['25%']
print ("Primary Camerea Pixels IQR : ", PrimaryCamereaPixels_IQR)

# Inter Quartile Range of ScreenHeight
ScreenHeight_IQR = data_clean.ScreenHeight.describe()['75%'] - data_clean.ScreenHeight.describe()['25%']
print ("Screen Height  IQR : ", ScreenHeight_IQR)

# Inter Quartile Range of ScreenWidth
ScreenWidth_IQR =data_clean.ScreenWidth.describe()['75%'] - data_clean.ScreenWidth.describe()['25%']
print ("Screen Width  IQR : ", ScreenWidth_IQR)

# Inter Quartile Range of TalkTime
TalkTime_IQR = data_clean.TalkTime.describe()['75%'] - data_clean.TalkTime.describe()['25%']
print ("Talk Time  IQR : ", TalkTime_IQR)

# %%
# Calculate correlation matrix for dataset
corr = data_clean.corr()
print(corr)

# %%
# Create heatmap to visualize correlation matrix
plt.figure(figsize=(15,10), layout='constrained',dpi=80)
sns.heatmap(corr, annot=True, cmap="Blues")

# Display heatmap
plt.show()

# we conclude that there is a positive relation between :
# 1-primary camera pixels and front camera pixels 
# 2-3G and 4G
# 3-price range and ram   // most important
# 4-pixel width & pixel height
# 5-screen width & screen height

# %%
#Visualization
# 1- Histogram
plt.hist(data_clean['ClockSpeed'])
plt.xlabel('Clock speed')
plt.ylabel('count')
plt.show

# significant difference in small clock speed values rather than larger ones
# right-skewed

# %%
plt.hist(data_clean['FrontCameraPixels'])
plt.xlabel('Front camera pixels')
plt.ylabel('count')
plt.show

#right-skewed  -> extreme values
#less front camera pixels, more count in data

# %%
plt.hist(x= data['InternalMemory'])
plt.xlabel('Internal memory')
plt.ylabel('count')
plt.show

# relatively close to each other

# %%
plt.hist(data_clean['MobileDepth'])
plt.xlabel('Mobile depth')
plt.ylabel('count')
plt.show

# significant difference in appearances of small mobile depth rather than larger mobile depths


# %%
plt.hist(data['MobileWeight'])
plt.xlabel('Mobile weight')
plt.ylabel('count')
plt.show

# relatively close to each other

# %%
plt.hist(data_clean['PixelHeight'])
plt.xlabel('Pixel height')
plt.ylabel('count')
plt.show

# right-skewed

# %%
plt.hist(data_clean['PrimaryCamereaPixels'])
plt.xlabel('Primary Camerea Pixels')
plt.ylabel('count')
plt.show

# significant difference in appearances of large primary camera pixels rather than smaller primary camera pixels


# %%
plt.hist(data_clean['Ram'])
plt.xlabel('Ram')
plt.ylabel('count')
plt.show

#relatively close to each other


# %%
plt.hist(data_clean['ScreenHeight'])
plt.xlabel('Screen height')
plt.ylabel('count')
plt.show

# significant difference between values


# %%
plt.hist(data['ScreenWidth'])
plt.xlabel('Screen width')
plt.ylabel('count')
plt.show

# right-skewed
# when screen width increases, count in data decreases
# with a slight drop in screen width from 7.5 to approximately 9.5


# %%
# 2- density plot
sns.displot(x = 'Ram', hue='PriceRange',data=data_clean, kind='kde')
# price range increase with the increase in Ram

# %%
sns.displot(x = 'DualSim', hue='PriceRange',data=data_clean, kind='kde')

# supporting dual sim increases price range


# %%
sns.displot(data_clean['FrontCameraPixels'],kind="kde")
#right-skewed distribution -> mean is greater than median due to the extreme values
# most of mobiles in our data have less front camera pixels


# %%
sns.displot(x = 'BatteryPower',hue='PriceRange',data = data_clean,kind="kde")

# small price range increase with less battery power and vise versa.


# %%
sns.displot(x = 'NumberOfCores', hue='PriceRange', data=data_clean,kind="kde")
# we can say that this is balanced distribution among this feature


# %%
# 3) countplot

## overall, data is balanced, except 3G attribute
sns.countplot(data=data_clean, x='Wifi')
plt.show

# 1-> yes, 0 -> no
# mobiles have wifi are greater than mobiles don't have with little amount


# %%
sns.countplot(data=data_clean, x='Bluetooth', hue='PriceRange')
plt.show
# mobiles don't have bluetooth are a little bit greater than mobiles have bluetooth
# mobiles don't have bluetooth have less price ranges.


# %%
sns.countplot(data=data_clean, x='three3G', hue='PriceRange')
plt.show

#There is a significant difference between having or not having 3G regarding to price range
# price ranges are balanced in each

# %%
sns.countplot(data=data_clean, x='DualSim', hue='PriceRange')
plt.show

# no specific observation

# %%
sns.countplot(data=data_clean, x='three3G', hue='four4G')
plt.show

#we conclude that large amount of mobiles that have 3G, also have 4G


# %%
sns.countplot(data=data_clean, x='NumberOfCores', hue='PriceRange')
plt.show

# over the data, mobiles with 4 cores found more than others, with price range 1 greater than other price ranges.


# %%
sns.countplot(data=data_clean, x='TouchScreen')
plt.show

# almost balanced (almost equally distributed)

# %%
sns.countplot(data=data_clean, x='PriceRange')
plt.show
# equal distributed.


# %%
data_clean.groupby('PriceRange').size().plot(kind='pie', autopct = '%.2f')

#ensure that price ranges are distributed equally.


# %%
data_clean.groupby('four4G').size().plot(kind='pie', autopct = '%.2f')

# mobiles with 4G are found a little bit more than mobiles without 4G


# %%
# 4- scatter plot
plt.scatter(x = data_clean['PriceRange'], y = data_clean['Ram'],marker='^')
plt.xlabel("Price Range")
plt.ylabel("Ram")
plt.title("Ram vs. price ranges")

# larger ram, larger price range

# %%
plt.scatter(x = data_clean['PrimaryCamereaPixels'], y = data_clean['FrontCameraPixels'], c='purple')
plt.xlabel("PrimaryCamereaPixels")
plt.ylabel("FrontCameraPixels")
plt.title("Primary Camera Pixels vs. Front Camera Pixels")

#as the primary camera pixels increase, the front camera pixels increase


# %%
plt.scatter(x = data_clean['BatteryPower'], y = data_clean['TalkTime'])
plt.xlabel("BatteryPower")
plt.ylabel("TalkTime")
plt.title("BatteryPower vs. TalkTime")

# no specific relation

# %%
#split data
x=data_clean.drop('PriceRange',axis=1)
y=data_clean['PriceRange']

# we are Classification problem because we are trying to predict categorical values


# %%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3,random_state=0)

from sklearn.metrics import accuracy_score, precision_score, recall_score




# %%
# random forest
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
pred = model.predict(x_test)

# printing
print('Accuracy : ', accuracy_score(y_test,pred))
print('Precision : ', precision_score(y_test, pred, average="macro"))
print('Recall : ', recall_score(y_test, pred, average="macro"))


# %%
#knn
from sklearn import neighbors

model= neighbors.KNeighborsClassifier()

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

# printing
print('Accuracy : ', accuracy_score(y_test,y_pred))
print('Precision : ', precision_score(y_test, y_pred, average="macro"))
print('Recall : ', recall_score(y_test,y_pred, average="macro"))



# %%
# decision tree
from sklearn.tree import DecisionTreeClassifier


model=DecisionTreeClassifier()

model.fit(x_train, y_train)

predictions=model.predict(x_test)

# printing
print('Accuracy : ', accuracy_score(y_test,predictions))
print('Precision : ', precision_score(y_test, predictions, average="macro"))
print('Recall : ', recall_score(y_test,predictions, average="macro"))




