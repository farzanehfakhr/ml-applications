# Student Performance Prediction

**How Study Effort and Life Factors Shape Student Performance**

Term project for METCS555 (Boston University) analyzing what drives academic performance among university students, using regression and statistical testing in R.

## Research Question

To what extent do class attendance, study hours per week, stress level, and parent education level predict a student's overall academic performance (Total Score)?

## Scenario

A university wants to understand which behavioral and socioeconomic factors contribute most to student success, in order to inform policy decisions such as attendance requirements or targeted academic support. The dataset includes demographic attributes (age, gender, family income), academic performance metrics (test scores, assignment grades), behavioral patterns (study hours, attendance), and lifestyle factors (stress levels, sleep habits).

## Approach

- Cleaned and prepared the student grading dataset in R (`analysis.R`), including handling a deliberately biased version of the dataset to test the sensitivity of the analysis to sampling/measurement issues
- Explored relationships between attendance, study hours, stress level, and parent education level against Total Score
- Applied regression modeling and hypothesis testing to quantify which factors were statistically significant predictors of performance

## Files

| File | Description |
|---|---|
| `Term_Project_Writeup.docx` | Full write-up: research question, methodology, results, and conclusions |
| `analysis.R` | R script for data cleaning, exploration, and statistical modeling |
| `data/Students_Grading_Dataset.csv` | Original dataset |
| `data/Students_Grading_Dataset_Biased.csv` | Deliberately biased variant used to test robustness of findings |
| `data/Students_Grading_Dataset_Cleaned.csv` | Cleaned dataset used for final analysis |

## Tools

R, tidyverse, broom, corrplot, ggplot2

---
*Completed as part of graduate coursework (METCS555, Statistics) at Boston University.*
