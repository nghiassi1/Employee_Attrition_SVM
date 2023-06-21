####Loaded packages
library(caret) 
library(class) 
library(dplyr) 
library(pROC) 

###loading data
employee = read.csv("C:/Users/Stacie/Downloads/EmployeeData.csv", stringsAsFactors = TRUE)
summary(employee)

###removing unnecssary data, and updating types of data
###(cateogorical: gender, martial status, business travel; and hot helpful: employeeID, Standard hours)
finalemployeedata =
  employee %>% select(-c(BusinessTravel, EmployeeID, Gender, MaritalStatus, StandardHours))

finalemployeedata$Attrition = as.factor(finalemployeedata$Attrition)

####splitting data 
set.seed(123)  
index = sample(nrow(finalemployeedata),0.7*nrow(finalemployeedata)) 

train_data = finalemployeedata[index, ]
test_data = finalemployeedata[-index, ]

###checking probabilities of split data 

prop.table(table(train_data$Attrition))
prop.table(table(test_data$Attrition))

####updating missing categories (total working years, job satisfaction, environment satisfaction,
####number companies worked) with means

train_data[is.na(train_data$TotalWorkingYears),'TotalWorkingYears'] = mean(train_data$TotalWorkingYears,na.rm=TRUE) 
test_data[is.na(test_data$TotalWorkingYears),'TotalWorkingYears'] = mean(train_data$TotalWorkingYears,na.rm=TRUE)

train_data[is.na(train_data$JobSatisfaction),'JobSatisfaction'] = mean(train_data$JobSatisfaction,na.rm=TRUE) 
test_data[is.na(test_data$JobSatisfaction),'JobSatisfaction'] = mean(train_data$JobSatisfaction,na.rm=TRUE)

train_data[is.na(train_data$EnvironmentSatisfaction),'EnvironmentSatisfaction'] = mean(train_data$EnvironmentSatisfaction,na.rm=TRUE) 
test_data[is.na(test_data$EnvironmentSatisfaction),'EnvironmentSatisfaction'] = mean(train_data$EnvironmentSatisfaction,na.rm=TRUE)

train_data[is.na(train_data$NumCompaniesWorked),'NumCompaniesWorked'] = mean(train_data$NumCompaniesWorked,na.rm=TRUE) 
test_data[is.na(test_data$NumCompaniesWorked),'NumCompaniesWorked'] = mean(train_data$NumCompaniesWorked,na.rm=TRUE)

###final check to ensure no missing data

sum(is.na(train_data))
sum(is.na(test_data))

####scaling data 
train_x = scale(train_data[, -2]) 
test_x = scale(test_data[,-2],center = apply(train_data[,-2],2,mean),
               scale = apply(train_data[,-2],2,sd))


###target variable 
train_y = train_data$Attrition 
test_y = test_data$Attrition 


####KNN model 
k = sqrt(nrow(train_x)) 
k

####k is 55.56078. so baseline of 56 will be used.

set.seed(123)
model_knn = knn(train=train_x, test=test_x, cl=train_y, k=56)
model_knn 

confusionMatrix(data = model_knn, 
                reference = test_y,
                positive = "Yes")

output = matrix(ncol=2, nrow=50)

for (k_val in 1:50){
  set.seed(123)
  temp_pred = knn(train = train_x
                  , test = test_x
                  , cl = train_y
                  , k = k_val)
  temp_eval = confusionMatrix(table(as.factor(temp_pred), as.factor(test_y))) 
  temp_acc = temp_eval$overall[1]
  output[k_val, ] = c(k_val, temp_acc) 
}

output = as.data.frame(output)
names(output) = c("K_value", "Accuracy")

ggplot(data=output, aes(x=K_value, y=Accuracy, group=1)) +
  geom_line(color="red")+
  geom_point()+
  theme_bw()


###first model
set.seed(123)
model_1 = knn(train=train_x, test=test_x, cl=train_y, k=7)


set.seed(123)
model_1_probs = attributes(knn(train=train_x, test=test_x, cl=train_y, k=7, prob=TRUE))$prob


confusionMatrix(data = model_1, 
                reference = test_y, 
                positive = "Yes")

###checking for alternative K values 
ctrl = trainControl(method="cv",number=10) 

knn_cv = train(
  Attrition ~ ., data = train_data,
  method = "knn", trControl = ctrl, 
  preProcess = c("center","scale"), tuneLength = 20)

knn_cv

plot (knn_cv)

###finding most important variables 
plot(varImp(knn_cv), 5) 



###confirming the model 
model_2 = predict(knn_cv, test_data, type="raw")


model_2_probs = predict(knn_cv, test_data, type="prob")[,2]

confusionMatrix(data = model_2, 
                reference = test_y, 
                positive = "Yes")


plot.roc(test_y,model_1_probs,legacy.axes=T)
plot.roc(test_y,model_2_probs,add=TRUE,col="red",lty=2)
legend("bottomright",legend=c("KNN","KNN.CV"),
       col=c("black","red"),lty=c(1,2),cex=0.75)


###ROC Curves
auc(test_y,model_1_probs)
auc(test_y,model_2_probs)