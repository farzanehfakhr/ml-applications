
# Student Performance & Behavior Analysis in 


# Load libraries 
library(tidyverse)
library(broom)
library(corrplot)
install.packages("ggplot2")
install.packages("corrplot")
library(ggplot2)

# 2. Read raw data 

# Read the CSV file 
#data <- read.csv("/Users/farzanehfakhr/Documents/cs555/project/Students_Grading_Dataset_Biased.csv", stringsAsFactors = TRUE)
data <- read.csv("/Users/farzanehfakhr/Documents/cs555/project/Students_Grading_Dataset.csv", stringsAsFactors = TRUE)

head(data)
str(data)
summary(data)


# 4. Data cleaning 
# Replace spaces and special characters with underscores
names(data) <- make.names(names(data), unique = TRUE)
data$Stress_Level..1.10. <- as.numeric(data$Stress_Level..1.10.)

  # rename for convenience
  data <- data %>%
  rename(
    Stress_Level = Stress_Level..1.10.,
    Attendance = Attendance....,  # renaming for easier reference
   
    #Study_Hours = Study_Hours_per_Week,
    #Parent_Ed = Parent_Education_Level,
    #Total = Total_Score
  )
colnames(data)

# Drop rows with missing key analysis variables
df_clean <- data %>%
  drop_na(Attendance, Study_Hours_per_Week, Total_Score, Stress_Level, Parent_Education_Level)
scores_pop <- df_clean $Total_Score
scores_pop
# Random sampling -----------------------------------------------------
set.seed(42)
df_sample <- df_clean %>% sample_n(1000)

# Write cleaned sample to CSV ----------------------------------------
write_csv(df_sample, "/Users/farzanehfakhr/Documents/cs555/project/Students_Grading_Dataset_Cleaned.csv")
data_cleaned <- read.csv("/Users/farzanehfakhr/Documents/cs555/project/Students_Grading_Dataset_Cleaned.csv", stringsAsFactors = TRUE)

summary(data_cleaned)
summary(select(data_cleaned, Total_Score, Attendance, Study_Hours_per_Week, Stress_Level,Parent_Education_Level,Internet_Access_at_Home))
#
n_reps <- 1000
n_samp <- 1000

# 4. Generate sampling distribution of the mean
sample_means <- replicate(n_reps,
                          mean(sample(scores_pop, size = n_samp, replace = FALSE))
)

# 5. Histogram
hist(sample_means,
     breaks = 30,
     main   = "Sampling Distribution of the Mean\n(n = 1000)",
     xlab   = "Sample Mean of Total_Score")

# 6. Q-Q plot
qqnorm(sample_means,
       main = "Q-Q Plot of Sample Means (n = 1000)")
qqline(sample_means)


# 
ggplot(data_cleaned, aes(x = Study_Hours_per_Week, y = Total_Score)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(
    title = "Study_Hours_per_Week vs Total Score",
    x = "Study_Hours_per_Week",
    y = "Total Score"
  ) 

hist(data_cleaned$Total_Score, 
     # breaks = seq(min(data_cleaned$Attendance) - 0.5, max(data_cleaned$Attendance) + 0.5, by = 1),
     main = "Histogram of Total Scores",
     xlab = "Scores",
     ylab = "Frequency",
     col = "lightblue", 
     border = "black")


#  Research Question 1: Correlation ------------------------------------
cor_test <- cor.test(data_cleaned$Attendance, data_cleaned$Total_Score)
print(cor_test)


ggplot(data_cleaned, aes(x = Attendance, y = Total_Score)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(
    title = "Attendance vs Total Score",
    x = "Attendance",
    y = "Total Score"
  ) 
# cor for nummeric variables
cor(select(data_cleaned, Total_Score, Attendance, Study_Hours_per_Week, Stress_Level), use = "complete.obs")

# Pairwise scatterplots
pairs(~ Total_Score + Attendance + Study_Hours_per_Week + Stress_Level+Parent_Education_Level+Internet_Access_at_Home, data = data_cleaned)

# Correlation heatmap
library(corrplot)
corrplot(cor(select(data_cleaned, Total_Score, Attendance, Study_Hours_per_Week, Stress_Level)),
         method = "color", addCoef.col = "black", tl.cex = 0.8)


#Research Question 2: Parental Education ANOVA 

# Ensure Parent_Education_Level is a factor
data_cleaned$Parent_Education_Level <- as.factor(data_cleaned$Parent_Education_Level)

# One-way ANOVA
anova_model <- aov(Total_Score ~ Parent_Education_Level, data = data_cleaned)
summary(anova_model)


# Post‐hoc Tukey if desired:
tukey <- TukeyHSD(anova_model)
print(tukey)

# Boxplot by Parent Education Level
ggplot(data_cleaned, aes(x = Parent_Education_Level, y = Total_Score)) +
  geom_boxplot(fill = "skyblue") +
  theme_minimal() +
  labs(title = "Total Score by Parent Education Level") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))



# Research Question 3: Study Hours Effect
# Fit the simple linear regression model
lm_hours <- lm(Total_Score ~ Study_Hours_per_Week, data = data_cleaned)
print(summary(lm_hours))

# Residual diagnostics
plot(lm_hours$fitted.values, lm_hours$residuals,
     xlab = "Fitted Values", ylab = "Residuals", main = "Residuals vs Fitted")
abline(h = 0, col = "red", lty = 2)

# QQ plot
qqnorm(lm_hours$residuals)
qqline(lm_hours$residuals, col = "blue")

#  Research Question 4: Multiple Regression 
# Encode Internet access as binary
data_cleaned <- data_cleaned %>%
mutate(Internet_Binary = if_else(Internet_Access_at_Home == "Yes", 1, 0))

# Convert Parent_Education_Level to factor
data_cleaned$Parent_Education_Level <- as.factor(data_cleaned$Parent_Education_Level)
# Fit the model
model <- lm(Total_Score ~ Attendance + Study_Hours_per_Week + Stress_Level + Internet_Binary+Parent_Education_Level,
            data = data_cleaned)


summary(model)

# ANCOVA is integrated in the multiple regression using a categorical covariate
# Use anova to test significance of parent education level after adjusting for other variables
anova(model)

# Residuals vs Fitted
plot(model$fitted.values, model$residuals,
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Residuals vs Fitted", pch = 20, col = "blue")
abline(h = 0, col = "red", lty = 2)

# QQ Plot for normality of residuals
qqnorm(model$residuals)
qqline(model$residuals, col = "red")


# Plots 
# Scatter: Attendance vs Total_Score
ggplot(data_cleaned, aes(x = Attendance, y = Total_Score)) +
  geom_point(alpha = 0.5) +
  labs(title = "Attendance vs Total Score",
       x = "Attendance....",
       y = "Total Score") 

# Boxplot: Total_Score by Internet Access
ggplot(data_cleaned, aes(x = Internet_Access_at_Home, y = Total_Score)) +
  geom_boxplot() +
  labs(title = "Total Score by Internet Access at Home",
       x = "Internet Access at Home",
       y = "Total Score") +
  theme_minimal() 



