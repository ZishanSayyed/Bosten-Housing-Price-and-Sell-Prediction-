                        # predicting house will be sold within 3 months or not 

data=read.csv('C:/Users/ADMIN/OneDrive/Documents/ML/machine learning models/House_sold.csv',header = TRUE)
str(data)

summary(data)
                        # Some EDA
#conclusion :
# 1-as we can observe an outlier or skewness in crime_rate,n_hot_rooms , rainfall 
#2-there are some missing values in n_hos_beds

#lets represent it graphically and understand it better 
hist(data$crime_rate)
names(data)
#relation of crime_rate,n_hot_rooms, rainfall with price
pairs(Sold~price+crime_rate+n_hot_rooms+rainfall,data=data)


#from the graph we can say that price & crime_rate has some relationship but its not linear 
#n_hot_room and rainfall has outliers 

#seeing categorical variables 
barplot(table(data$airport))
barplot(table(data$waterbody))
barplot(table(data$bus_ter))  #as we see we have only yes data so we can remove it 
barplot(table(data$airport))

#observation
# there are some missing values in n_hos_beds
# n_hot_room and rainfall has outliers
# price & crime_rate has some relationship but its not linear
# bus terminal has no use 


#handling outlier 

quantile(data$n_hot_rooms,0.99)
upper_v=3*quantile(data$n_hot_rooms,0.99)
data$n_hot_rooms[data$n_hot_rooms> upper_v]=upper_v
summary(data$n_hot_rooms)


quantile(data$rainfall,0.01)
lower_v=0.3*quantile(data$rainfall,0.01)
data$rainfall[data$rainfall<lower_v]=lower_v
summary(data$rainfall)



#handeling missing values
mean(data$n_hos_beds,na.rm = TRUE)
which(is.na(data$n_hos_beds))
data$n_hos_beds[is.na(data$n_hos_beds)]=mean(data$n_hos_beds,na.rm = TRUE)
(is.na(data$n_hos_beds))

# price & crime_rate has some relationship but its not linear

plot(data$price,data$crime_rate)
data$crime_rate=log(1+data$crime_rate)     
plot(data$price,data$crime_rate)  #now we can see a bit linear relation between the variables

#data cleaning 
#in data there are 4 dist we replace it by its avg 
data$avg_dist=(data$dist1+data$dist2+data$dist3+data$dist4)/4

#removing useless columns 
data= data[,-7:-10] 
data=data[,c(-14)]

#converting categorical var into numerical form to see their relation with price

library(dummies)
data=dummy.data.frame(data)

#again removing extra columns 
data=data[,c(-9,-15)]
#we removed the column airportNo and waterbodiesNOne

#Now our data is ready to future predection 

                   #logistic regression
glm_fit=glm(Sold~price ,data=data,family = binomial)  #singel var 
summary(glm_fit)

glm_fit2=glm(Sold~. ,data=data,family = binomial)   #multi var 
summary(glm_fit2)


#seeing prob of house to be sold within 3 months
glm_prob=predict(glm_fit2,type = "response")
glm_prob[1:10]

glm_pred=rep("No",506)
glm_pred[glm_prob>0.5]="Yes"


#creating confusision metrix 
table(glm_pred,data$Sold)
acc_with_logistic=(196+154)/506

               #linear discrimnet analysis (LDA)
library("MASS")
lda_fit=lda(Sold~.,data = data)
lda_fit
lda_pred=predict(lda_fit,data)

#to see posterier prob'
lda_pred$posterior

#creating confusision metrix 
table(lda_pred$class,data$Sold)
acc_with_LDA=(192+151)/506


           #k Nearest Neighbors(KNN)
library(caTools)
set.seed(0)
split=sample.split(data,SplitRatio = 0.8)
Train_set=subset(data,split== TRUE)
test_set=subset(data,split==FALSE)

train_fit=glm(Sold~.,data=Train_set,family = binomial)
test_prob=predict(train_fit,test_set,type = "response")

test_pred=rep('NO',113)
test_pred[test_prob>0.5]="Yes"

#creating confusision metrix 
table(test_pred,test_set$Sold)
acc_with_tt=(37+32)/113

library(class)

trainX=Train_set[,-17]   #removing Sold variable 
testX=test_set[,-17]
trainY=Train_set$Sold
testY=test_set$Sold

trainX_s=scale(trainX)
testX_s=scale(testX)

knn_pred=knn(trainX_s,testX_s,trainY,k=4)
#creating confusision metrix 
table(knn_pred,testY)
acc_with_knn=(35+29)/113


#best model 
max(acc_with_knn,acc_with_LDA,acc_with_logistic,acc_with_tt)

#we get best model with Logistic Regression 

