#install.packages('rstan')
#install.packages('ROCR')
library(rstan); library(ROCR);
rm(list=ls())

# read in the fully connected responses and remove null answers
open_answers_fully_connected <- read.csv('bert_full_connected_responses.csv')
df <- na.omit(open_answers_fully_connected$cleaned_answer_text)


# Merge any Covariates ---------------------------------------------------------------
# ------------------------------------------------------------------------------------
### This is where any predictions or other covariates should be loaded and merged 
### onto the open_answers dataframe using problem_log_id
### Any such features to be used as covariates in the model must be defined in the
### covariates list

nwords <- read.csv('nword_model.csv')
names(nwords)[names(nwords)=='id'] <- 'problem_log_id'
open_answers_fully_connected <- merge(open_answers_fully_connected,nwords,all=TRUE)

open_answers_fully_connected$nwords <- ifelse(is.na(open_answers_fully_connected$nwords),0,
                                              log(open_answers_fully_connected$nwords))
# define list of columns to use as covariates
# p0, p1, p2, p3 and p4 here are the one hot encoding for prediction of scores
covariates <- c('nwords', 'p0', 'p1', 'p2', 'p3', 'p4')

### end of covariate merging
# ------------------------------------------------------------------------------------
# re-number the students and questions for the rasch model
  open_answers_fully_connected <- open_answers_fully_connected %>%
  mutate(student = as.numeric(as.factor(student_user_id)),
         question = as.numeric(as.factor(problem_id)))



# Run model ---------------------------------------------------------------

colAveragebyClass <- function(x, classes=NA) {
  # if (is.na(classes)) classes <= unique(x[,1])
  #x <- p
  #classes <- c(1,2,3,4,5)
  y <- data.frame(matrix(nrow=ncol(x),ncol=length(classes)+1))
  names(y) <- c('column', classes)
  
  for (i in 1:ncol(x)) {
    #for (i in 1:100) {  
    #i = 1
    #j = j
    y$column[i] <- names(x)[i]
    cnt <- plyr::count(x[,i]) %>% filter(x %in% classes) %>% mutate(freq=freq/sum(freq,na.rm=TRUE))
    for (j in classes) {
      c <- cnt %>% dplyr::filter(x==j) 
      y[i,as.character(names(y))==as.character(j)] <- ifelse(is.na(c$freq[1]),0,c$freq[1])
    }
  }
  return (y)
}


ordered_rasch <- stan_model("ordered_rasch.stan")
ordered_rasch_ft <- stan_model('ordered_rasch_withfeatures.stan')


splits <- open_answers_fully_connected$folds                                   

predicted_p <- list()
predicted_p_cov <- list()
for(s in 1:10) {
  data_list <- list(N = nrow(open_answers_fully_connected[splits != s,]),
                    S = length(unique(open_answers_fully_connected$student)),
                    Q = length(unique(open_answers_fully_connected$question)),
                    C = length(unique(open_answers_fully_connected$grade)),
                    N_OOS = nrow(open_answers_fully_connected[splits == s,]),
                    student = open_answers_fully_connected$student[splits != s],
                    question = open_answers_fully_connected$question[splits != s],
                    grade = open_answers_fully_connected$grade[splits != s],
                    student_oos = open_answers_fully_connected$student[splits == s],
                    question_oos = open_answers_fully_connected$question[splits == s])
  
  data_list_cov <- append(data_list,list(P = length(covariates),
                                         X = as.matrix(open_answers_fully_connected[splits != s,covariates]),
                                         X_OOS = as.matrix(open_answers_fully_connected[splits == s,covariates])))
  
  mcmc_draws <- vb(ordered_rasch, data = data_list, seed=s)
  p <- colAveragebyClass(as.data.frame(mcmc_draws, par = "y_pred"),c(1,2,3,4,5))
  p$label <- open_answers_fully_connected$grade[splits == s]
  predicted_p[[s]] <- p
  
  mcmc_draws <- vb(ordered_rasch_ft, data = data_list_cov, seed=s)
  p <- colAveragebyClass(as.data.frame(mcmc_draws, par = "y_pred"),c(1,2,3,4,5))
  p$label <- open_answers_fully_connected$grade[splits == s]
  predicted_p_cov[[s]] <- p
}


# Evaluate Model (No Covariates) ---------------------------------------------------------------

OOS_PROBS <- bind_rows(predicted_p)
OOS_PROBS <- OOS_PROBS %>% mutate(lb_1 = as.integer(label==1),
                                  lb_2 = as.integer(label==2),
                                  lb_3 = as.integer(label==3),
                                  lb_4 = as.integer(label==4),
                                  lb_5 = as.integer(label==5))
OOS_PROBS$pred_class <- sapply(as.data.frame(t(OOS_PROBS[,2:6])),which.max)

hist(OOS_PROBS$label)

auc_0 <- 0
for (i in 1:5) {
  p <- prediction(predictions=OOS_PROBS[,1+i],labels=OOS_PROBS[,7+i])
  auc_0  <- auc + performance(p,'auc')@y.values[[1]]
}

rmse_0 <- mean(sqrt((OOS_PROBS$label-OOS_PROBS$pred_class)^2))
kpa_0 <- cohen.kappa(OOS_PROBS[,c(7,13)])


# Evaluate Model (W/ Covariates) ---------------------------------------------------------------

OOS_PROBS <- bind_rows(predicted_p_cov)
OOS_PROBS <- OOS_PROBS %>% mutate(lb_1 = as.integer(label==1),
                                  lb_2 = as.integer(label==2),
                                  lb_3 = as.integer(label==3),
                                  lb_4 = as.integer(label==4),
                                  lb_5 = as.integer(label==5))
OOS_PROBS$pred_class <- sapply(as.data.frame(t(OOS_PROBS[,2:6])),which.max)

hist(OOS_PROBS$label)

auc <- 0
for (i in 1:5) {
  p <- prediction(predictions=OOS_PROBS[,1+i],labels=OOS_PROBS[,7+i])
  auc  <- auc + performance(p,'auc')@y.values[[1]]
}

rmse <- mean(sqrt((OOS_PROBS$label-OOS_PROBS$pred_class)^2))
kpa <- cohen.kappa(OOS_PROBS[,c(7,13)])

print('======================================')
print('Rasch Model (No Covariates):')
print(paste('AUC:',round(auc_0/5.,3),sep=' '))
print(paste('RMSE:',round(rmse_0,3),sep=' '))
print(paste('KAPPA:',round(kpa_0$kappa,3),sep=' '))
print('======================================')
print('-----------------------------------')
print('======================================')
print('Rasch Model (With Covariates):')
print(paste('AUC:',round(auc/5.,3),sep=' '))
print(paste('RMSE:',round(rmse,3),sep=' '))
print(paste('KAPPA:',round(kpa$kappa,3),sep=' '))
print('======================================')