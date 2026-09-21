
# Load necessary libraries
library(caret)
library(randomForest)
library(pROC)
library(boot)
library(stringr)
library(data.table)
library(glue)
library(matrixTests)
library(matrixStats)
library(dplyr)

# Load the data
cat("Test\tAUC\tCI95\tSe\tCI95\tSp\tCI95\tACC\tCI95\n",file="chrY/ROC-results-loo-svm.txt")
cancers <- c("lung","pancreatic", "colorectal")

# Filter datasets


for (cancer in cancers){

  data_e <- read.csv("chrY_figs_svg/scalar_ratios_short_long_ratio_147bp_cristiano.csv", header = TRUE, row.names = 1, sep=";")

  target_variable <- c(rep("healthy", nrow(data_e[data_e$group=="healthy",])))

  target_variable <- c(target_variable, rep("cancer", nrow(data_e[data_e$group==cancer,])))
  #Dataset avec tous les samples heatlhy et cancer en lignes et tous les end motifs en colonne
  data_t <- data_e[data_e$group%in%c("healthy",cancer),]
  # Create target variable based on sample names

  target_variable <- factor(target_variable, levels = c("healthy","cancer"))

  # Ensure the data is in the correct format
  data_t <- as.data.frame(data_t)
  data_t$group <- target_variable
  # Combine the data and target variable into one data frame

  train_control <- trainControl(method = "LOOCV",sampling ="smote",classProbs = TRUE,summaryFunction = twoClassSummary , savePredictions = TRUE)

  # Train the random forest model using stratified k-fold cross-validation
  rf_model <- train(group ~ ., data = data_t, method = "svmLinear", trControl = train_control, metric = "ROC", importance = TRUE)

  # Get cross-validation results
  loo_res <- rf_model$pred

  dist.to.diag <- function(c){
    x0 <- c$x0
    y0 <- c$y0
    sqrt((x0-(1+x0-y0)*0.5)**2 + (y0-(1-x0+y0)*0.5)**2)
  }

  #For specificity at 0.95
  roc.stat2 <- function(orig,indices){
    c <- orig[sort(indices),]
    x0 <- c$x0
    i <- which.min(abs(x0 - (0.95)))
    c(se=c$y0[i],sp=c$x0[i])
  }
  #capture.output(rf_model, file = glue("chrY/{cancer}-svm-predictions.txt"), append = TRUE)
  write.csv(loo_res, file = glue("chrY/{cancer}-svm-predictions.csv"), row.names = FALSE)
  
  oneRow <- function(name,R=1000){
    auc.roc <- roc(response = factor(loo_res$obs, levels = c("healthy", "cancer")),
                   predictor = loo_res$cancer,
                   ci = TRUE, of = "auc")
    curve <- data.frame(x0=auc.roc$specificities,y0=auc.roc$sensitivities)
    boot.auc <- boot(data=curve,statistic=roc.stat2,R=R,sim="balanced")
    ci.se <- boot.ci(boot.auc,type="basic",index=1)
    ci.sp <- boot.ci(boot.auc,type="basic",index=2)
    est <- c(auc.roc$ci[2],auc.roc$ci[1],auc.roc$ci[3],
             ci.se$t0,ci.se$basic[,4:5],
             ci.sp$t0,ci.sp$basic[,4:5])
    accs <- with(loo_res, pred == obs)
    mean_acc <- mean(accs)
    ci_acc <- quantile(boot(accs, function(x, i) mean(x[i]), R = 1000)$t, probs = c(0.025, 0.975))
    cat(sprintf("Accuracy: %.3f [95%% CI: %.3f - %.3f]\n", mean_acc, ci_acc[1], ci_acc[2]))
    sprintf("%s\t%.3f\t[%.3f; %.3f]\t%.3f\t[%.3f; %.3f]\t%.3f\t[%.3f; %.3f]\t%.3f\t[%.3f; %.3f]\n",
            name,est[1],est[2],est[3],est[4],est[5],est[6],est[7],est[8],est[9], mean_acc, ci_acc[1], ci_acc[2])
  }

  #Enregistrer performance

  cat(oneRow(glue("healthy-{cancer}")),append=T,file="chrY/ROC-results-loo-svm.txt")
}
# Read predictions back (use read.csv so headers parse into distinct columns)
svm.pred.crc  <- read.csv("chrY/colorectal-svm-predictions.csv", header = TRUE)
svm.pred.panc <- read.csv("chrY/pancreatic-svm-predictions.csv", header = TRUE)
svm.pred.lung <- read.csv("chrY/lung-svm-predictions.csv", header = TRUE)

# Verify the column names exist:
# colnames(svm.pred.crc) should be: c("pred", "obs", "healthy", "cancer", "rowIndex", ...)

# Generate ROC objects:
# Note: 'obs' contains the true labels ("healthy" vs "cancer")
# 'cancer' contains the predicted probabilities for the cancer class.

roc.crc  <- roc(response  = factor(svm.pred.crc$obs, levels = c("healthy", "cancer")), 
                predictor = svm.pred.crc$cancer, 
                ci = TRUE, of = "auc")

roc.panc <- roc(response  = factor(svm.pred.panc$obs, levels = c("healthy", "cancer")), 
                predictor = svm.pred.panc$cancer, 
                ci = TRUE, of = "auc")

roc.lung <- roc(response  = factor(svm.pred.lung$obs, levels = c("healthy", "cancer")), 
                predictor = svm.pred.lung$cancer, 
                ci = TRUE, of = "auc")

roc.list <- list("colorectal" = roc.crc, "pancreatic" = roc.panc, "lung" = roc.lung)
roc.list <- list("colorectal" = roc.crc, "pancreatic" = roc.panc, "lung" = roc.lung)

# Calculate CI ranges across specificities
ci.list <- lapply(roc.list, ci.se, specificities = seq(0, 1, length.out = 50))

dat.ci.list <- lapply(ci.list, function(ciobj) {
  data.frame(x = as.numeric(rownames(ciobj)), lower = ciobj[, 1], upper = ciobj[, 3])
})

# Plot using ggplot2 / ggroc
pointsize <- 7
p <- ggroc(roc.list, linewidth = 0.5) +
  theme(
    legend.key = element_blank(),
    legend.key.size = unit(0.2, "cm"),
    legend.position = "right",
    legend.box = "horizontal",
    axis.text.x = element_text(colour = "black", size = pointsize, angle = 90, vjust = 0.3, hjust = 1),
    axis.text.y = element_text(colour = "black", size = pointsize),
    axis.title = element_text(color = "black", size = pointsize),
    legend.text = element_text(size = pointsize - 1, colour = "black"),
    legend.title = element_text(size = pointsize),
    panel.background = element_blank(),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.5),
    panel.grid.major = element_line(colour = "grey95")
  ) +
  geom_abline(slope = 1, intercept = 1, linetype = "dashed", alpha = 0.7, color = "grey") +
  coord_equal()

for (i in seq_along(roc.list)) {
  p <- p + geom_ribbon(
    data = dat.ci.list[[i]], 
    aes(x = x, ymin = lower, ymax = upper),
    fill = i + 1, alpha = 0.2, inherit.aes = FALSE
  )
}
p