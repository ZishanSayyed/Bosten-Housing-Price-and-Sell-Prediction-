                                                         # Bostan housing Data EDA
df=read.csv('C:/Users/ADMIN/OneDrive/Documents/ML/machine learning models/House_Price.csv')
str(df)


#to get basic understanding of data (how data is distributed)
summary(df)

#conclusion :
# 1-as we can observe an outlier or skewness in crime_rate,n_hot_rooms , rainfall 
#2-there are some missing values in n_hos_beds

#lets represent it graphically and understand it better 
hist(df$crime_rate)
names(df)
#relation of crime_rate,n_hot_rooms, rainfall with price
pairs(~price+crime_rate+n_hot_rooms+rainfall,data=df)


#from the graph we can say that price & crime_rate has some relationship but its not linear 
#n_hot_room and rainfall has outliers 

#seeing categorical variables 
barplot(table(df$airport))
barplot(table(df$waterbody))
barplot(table(df$bus_ter))  #as we see we have only yes data so we can remove it 
barplot(table(df$airport))

#observation
# there are some missing values in n_hos_beds
# n_hot_room and rainfall has outliers
# price & crime_rate has some relationship but its not linear
# bus terminal has no use 


#handling outlier 

quantile(df$n_hot_rooms,0.99)
upper_v=3*quantile(df$n_hot_rooms,0.99)
df$n_hot_rooms[df$n_hot_rooms> upper_v]=upper_v
summary(df$n_hot_rooms)


quantile(df$rainfall,0.01)
lower_v=0.3*quantile(df$rainfall,0.01)

df$rainfall[df$rainfall<lower_v]=lower_v
summary(df$rainfall)



#handeling missing values
mean(df$n_hos_beds,na.rm = TRUE)
which(is.na(df$n_hos_beds))
df$n_hos_beds[is.na(df$n_hos_beds)]=mean(df$n_hos_beds,na.rm = TRUE)
(is.na(df$n_hos_beds))

# price & crime_rate has some relationship but its not linear

plot(df$price,df$crime_rate)
df$crime_rate=log(1+df$crime_rate)     
plot(df$price,df$crime_rate)  #now we can see a bit linear relation between the variables

#data cleaning 
#in data there are 4 dist we replace it by its avg 
df$avg_dist=(df$dist1+df$dist2+df$dist3+df$dist4)/4

#removing useless columns 
df= df[,-7:-10] 
df=df[,c(-14)]


#converting categorical var into numerical form to see their relation with price

library(dummies)
df=dummy.data.frame(df)

#again removing extra columns 
df=df[,c(-9,-15)]
#we removed the column airportNo and waterbodiesNOne

#correlation metrix in r 
round(cor(df),2)

#as we can say that room_num  has most positive response with price of house 
#and rainfall,n_hot_rooms,waterbodies has not much impact on price of house 
#air_qual and parks has 0.92% correlated that can coz over-fitting in the model 

                  #fitting linear Regression MOdels

simple_model=lm(price~room_num,data = df)
summary(simple_model)

#interpretation : room_num is significant impact on price by 90% individually  
plot(df$room_num,df$price)
abline(simple_model)


multi_model=lm(price~.,data = df)
summary(multi_model)
# * are saying variables as significant and other has not much effect on model 
# R-seq saying 72% variance can be satisfied 



                        #Training and Testing model 

library(caTools)
set.seed(0)
split=sample.split(df,SplitRatio = 0.8)
tain_set=subset(df,split==TRUE)
test_set=subset(df,split==FALSE)

lm_a=lm(price~.,data = tain_set)
#training model 
train_a=predict(lm_a,tain_set)
#predicting test values also 
test_a=predict(lm_a,test_set)


#lets check the model performance 
mean((tain_set$price - train_a)^2)
mean((test_set$price - test_a)^2)

#as we see the model does not perform well on test data 
#to improve the performance of the model we use subset selection technique 
library(leaps)

lm_best=regsubsets(price~.,data = df,nvmax =15 )
summary(lm_best)

#seeing which model is best we see their R-sq values
summary(lm_best)$adjr2
which.max(summary(lm_best)$adjr2)   #has best model among all other models 
coef(lm_best,9)     # to see model coefficient 


                       #forward selection technique
lm_forward=regsubsets(price~.,data = df,nvmax =15 , method = "forward")
summary(lm_forward)
summary(lm_forward)$adjr2
which.max(summary(lm_forward)$adjr2)
coef(lm_forward,9)
      
                      #backward selection technique
lm_back=regsubsets(price~.,data = df,nvmax =15 , method = "backward")
summary(lm_back)
summary(lm_back)$adjr2
which.max(summary(lm_back)$adjr2)
coef(lm_back,9)


                   #reg regression
library(glmnet)
#segregating data into dependent and independent variable

x=model.matrix(price~.,data = df)[,-1] #all variable other than price are indepandent variable 
y=df$price                             #dependent variable 

# we create different values of lambda and find for which lambda error is min 
grid=10^seq(10,-2,length=100)
grid
lm_ridge=glmnet(x,y,alpha = 0,lambda = grid)  # 0 for reg regression 
summary(lm_ridge)

#finding best lambda 
cv_fit=cv.glmnet(x,y,alpha=0,lambda = grid)
plot(cv_fit)

#optimum lambda
opt_lambda=cv_fit$lambda.min
opt_lambda

y_a=predict(lm_ridge,s=opt_lambda,newx = x)

#checking Rsq value
tss=sum((y-mean(y))^2)
rss=sum((y_a-y)^2)
Rsq=1-rss/tss    #model is covering 72% variance of the data 

                             #Lasso 
library(glmnet)
#segregating data into dependent and independent variable

x=model.matrix(price~.,data = df)[,-1] #all variable other than price are indepandent variable 
y=df$price                             #dependent variable 

# we create different values of lambda and find for which lambda error is min 
grid=10^seq(10,-2,length=100)
grid
lm_laso=glmnet(x,y,alpha = 1,lambda = grid)  # 1 for lasso 
summary(lm_laso)

#finding best lambda 
cv_fit=cv.glmnet(x,y,alpha=1,lambda = grid)
plot(cv_fit)

#optimum lambda
opt_lambda=cv_fit$lambda.min
opt_lambda

y_a=predict(lm_laso,s=opt_lambda,newx = x)

#checking Rsq value
tss=sum((y-mean(y))^2)
rss=sum((y_a-y)^2)
Rsq=1-rss/tss    
