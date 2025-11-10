# Databricks notebook source
clientid = dbutils.secrets.get(scope="team4scope", key="clientid")
tenantid = dbutils.secrets.get(scope="team4scope", key="tenantid")
secretid = dbutils.secrets.get(scope="team4scope", key="secretid")
spark.conf.set("fs.azure.account.auth.type.loanpredictionadls.dfs.core.windows.net", "OAuth")
spark.conf.set("fs.azure.account.oauth.provider.type.loanpredictionadls.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
spark.conf.set("fs.azure.account.oauth2.client.id.loanpredictionadls.dfs.core.windows.net", clientid)
spark.conf.set("fs.azure.account.oauth2.client.secret.loanpredictionadls.dfs.core.windows.net", secretid)
spark.conf.set("fs.azure.account.oauth2.client.endpoint.loanpredictionadls.dfs.core.windows.net", f"https://login.microsoftonline.com/{tenantid}/oauth2/token")

# COMMAND ----------

from pyspark.sql.functions import *

# Load loan data
loan_df = spark.read.csv("abfss://datasets@loanpredictionadls.dfs.core.windows.net/silver/loan_pbi.csv", header=True, inferSchema=True)

# Query for total loan disbursed, average loan term, and total default count
loan_performance = loan_df.agg(
    sum("PrincipalAmount").alias("TotalLoanDisbursed"),
    avg("TermMonths").alias("AverageLoanTerm"),
    count(col("DefaultFlag")).alias("TotalDefaults")
)

loan_performance.show()


# COMMAND ----------

from pyspark.sql.functions import *

# Query for default rates by loan type
loan_types_default_rate = loan_df.groupBy("LoanType").agg(
    sum(when(col("DefaultFlag") == 1, 1).otherwise(0)).alias("DefaultCount"),
    count("LoanID").alias("TotalLoans")
)

loan_types_default_rate = loan_types_default_rate.withColumn(
    "DefaultRate", col("DefaultCount") / col("TotalLoans")
)

loan_types_default_rate.show()


# COMMAND ----------

from pyspark.sql.functions import *

# -------------------------------------------------------
# 1️⃣ Read the datasets
# -------------------------------------------------------
loan_data = spark.read.csv(
    "abfss://datasets@loanpredictionadls.dfs.core.windows.net/silver/loan_pbi.csv",
    header=True,
    inferSchema=True
)

customer_data = spark.read.csv(
    "abfss://datasets@loanpredictionadls.dfs.core.windows.net/silver/customers_pbi.csv",
    header=True,
    inferSchema=True
)

# -------------------------------------------------------
# 2️⃣ Join both DataFrames on CustomerID (common key)
# -------------------------------------------------------
loan_df = loan_data.join(customer_data, on="CustomerID", how="inner")

# -------------------------------------------------------
# 3️⃣ Derive Credit Score Category
# -------------------------------------------------------
loan_df = loan_df.withColumn(
    "CreditScoreCategory",
    when(col("CreditScore") < 400, "Poor")
    .when(col("CreditScore") < 600, "Average")
    .when(col("CreditScore") < 800, "Good")
    .otherwise("Excellent")
)

# -------------------------------------------------------
# 4️⃣ Calculate Default Rate by CreditScoreCategory
# -------------------------------------------------------
credit_score_default = (
    loan_df.groupBy("CreditScoreCategory")
    .agg(
        sum(when(col("DefaultFlag") == 1, 1).otherwise(0)).alias("DefaultCount"),
        count("LoanID").alias("TotalLoans")
    )
    .withColumn("DefaultRate", col("DefaultCount") / col("TotalLoans"))
)

display(credit_score_default)
display(loan_df)


# COMMAND ----------

from pyspark.sql import functions as F

# Load customer data
customer_df = spark.read.csv(
    "abfss://datasets@loanpredictionadls.dfs.core.windows.net/silver/customers_pbi.csv",
    header=True,
    inferSchema=True
)

# Alias input DFs
l = loan_df.alias("l")
c = customer_df.alias("c")

# Join on CustomerID
loan_customer_df = (
    l.join(c, F.col("l.CustomerID") == F.col("c.CustomerID"), "inner")
)

# Use the CUSTOMER credit score for categorization (choose c.CreditScore explicitly)
loan_df = loan_df.withColumn(
    "CreditScoreCategory",
    F.when(F.col("c.CreditScore") < 400, "Poor")
     .when(F.col("c.CreditScore") < 600, "Average")
     .when(F.col("c.CreditScore") < 800, "Good")
     .otherwise("Excellent")
)

# (Optional) Prune columns to avoid name collisions later
# loan_customer = loan_customer_df.select(
#     F.col("l.LoanID"),
#     F.col("l.CustomerID"),
#     F.col("l.DefaultFlag"),
#     F.col("c.CreditScore"),
#     F.col("CreditScoreCategory")
# # )

# Aggregate: default probability by credit score category
credit_score_default = (
    loan_df
    .groupBy("CreditScoreCategory")
    .agg(
        F.sum(F.when(F.col("DefaultFlag") == 1, 1).otherwise(0)).alias("DefaultCount"),
        F.countDistinct("LoanID").alias("TotalLoans")
    )
    .withColumn("DefaultRate", F.col("DefaultCount") / F.col("TotalLoans"))
    .orderBy("CreditScoreCategory")
)

display(credit_score_default)

# COMMAND ----------

from pyspark.sql.functions import avg

employment_income_loan = (
    loan_df.groupBy("EmploymentStatus").agg(
        avg("PrincipalAmount").alias("AverageLoanSize"),
        avg("AnnualIncome").alias("AverageIncome")
    )
)

display(employment_income_loan)

# COMMAND ----------

# Query for loan default rate by age group
age_groups_default = loan_df.withColumn(
    "AgeGroup", 
    when(col("Age") < 30, "Young")
    .when(col("Age") < 50, "Middle-Aged")
    .otherwise("Older")
).groupBy("AgeGroup").agg(
    sum(when(col("DefaultFlag") == 1, 1).otherwise(0)).alias("DefaultCount"),
    count("LoanID").alias("TotalLoans")
)

age_groups_default = age_groups_default.withColumn(
    "DefaultRate", col("DefaultCount") / col("TotalLoans")
)

age_groups_default.show()


# COMMAND ----------

# Query for loan performance by region
region_performance = loan_df.groupBy("Region").agg(
    sum("PrincipalAmount").alias("TotalLoanDisbursed"),
    count("CustomerID").alias("ActiveCustomers"),
    sum(when(col("DefaultFlag") == 1, 1).otherwise(0)).alias("TotalDefaults")
)

region_performance.show()


# COMMAND ----------

# Define income categories
loan_df = loan_df.withColumn(
    "IncomeLevel",
    when(col("AnnualIncome") < 500000, "Low")
    .when(col("AnnualIncome") < 1000000, "Medium")
    .otherwise("High")
)

# Query for default rate by income level
income_default_rate = loan_df.groupBy("IncomeLevel").agg(
    sum(when(col("DefaultFlag") == 1, 1).otherwise(0)).alias("DefaultCount"),
    count("LoanID").alias("TotalLoans")
)

income_default_rate = income_default_rate.withColumn(
    "DefaultRate", col("DefaultCount") / col("TotalLoans")
)

income_default_rate.show()


# COMMAND ----------

# Calculate DTI ratio and analyze default rate
loan_df = loan_df.withColumn(
    "DTIRatio", col("EMIAmount") / col("AnnualIncome")
)

dti_default_rate = loan_df.groupBy("DTIRatio").agg(
    sum(when(col("DefaultFlag") == 1, 1).otherwise(0)).alias("DefaultCount"),
    count("LoanID").alias("TotalLoans")
)

dti_default_rate = dti_default_rate.withColumn(
    "DefaultRate", col("DefaultCount") / col("TotalLoans")
)

dti_default_rate.show()


# COMMAND ----------

from pyspark.sql.functions import col, when, avg, count

# Load the customer data
customer_df = spark.read.csv("abfss://datasets@loanpredictionadls.dfs.core.windows.net/silver/customers_pbi.csv", header=True, inferSchema=True)
loan_df = spark.read.csv("abfss://datasets@loanpredictionadls.dfs.core.windows.net/silver/loan_pbi.csv", header=True, inferSchema=True)

# Join loan data with customer data on CustomerID to access the AnnualIncome
loan_customer_df = loan_df.join(customer_df, on="CustomerID", how="inner")

# Filter out rows with null or zero values in EMIAmount or AnnualIncome
loan_customer_df = loan_customer_df.filter(
    (col("EMIAmount").isNotNull()) & (col("EMIAmount") > 0) & 
    (col("AnnualIncome").isNotNull()) & (col("AnnualIncome") > 0)
)

# Calculate the Debt-to-Income (DTI) ratio: EMI / AnnualIncome
loan_customer_df = loan_customer_df.withColumn(
    "DTIRatio", col("EMIAmount") / col("AnnualIncome")
)

# Query to get the average DTI ratio and the relationship with defaults
dti_default_analysis = loan_customer_df.groupBy("DTIRatio").agg(
    avg("DTIRatio").alias("AverageDTIRatio"),
    sum(when(col("DefaultFlag") == 1, 1).otherwise(0)).alias("DefaultCount"),
    count("LoanID").alias("TotalLoans")
)

# Calculate the default rate for each DTI ratio
dti_default_analysis = dti_default_analysis.withColumn(
    "DefaultRate", col("DefaultCount") / col("TotalLoans")
)

# Show the results, this output will be useful for Power BI visualization
dti_default_analysis.show()


# COMMAND ----------

from pyspark.sql.functions import col, sum, when

# Load loan data (assuming loan_data is the DataFrame containing the loan dataset)
loan_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("abfss://datasets@loanpredictionadls.dfs.core.windows.net/silver/loan_pbi.csv")

# Assuming no RecoveryAmount column, calculate recovery rate as a proxy (e.g., ratio of PrincipalAmount that is recovered)
loan_data_recovery = loan_df.withColumn(
    "RecoveryRate", when(col("DefaultFlag") == 1, col("PrincipalAmount") * 0.45).otherwise(0)  # Assuming 45% recovery for defaults
)

# Now, aggregate and calculate the average recovery rate for each LoanType
recovery_rate_analysis = loan_data_recovery.groupBy("LoanType").agg(
    sum("RecoveryRate").alias("TotalRecovered"),
    sum("PrincipalAmount").alias("TotalDisbursed")
)

# Calculate the recovery rate as a ratio
recovery_rate_analysis = recovery_rate_analysis.withColumn(
    "RecoveryRatePercentage", col("TotalRecovered") / col("TotalDisbursed") * 100
)

# Show the results for Power BI visualization
recovery_rate_analysis.show()


# COMMAND ----------

# Analyze loan performance across geographic regions
region_performance = loan_customer_df.groupBy("Region").agg(
    sum("PrincipalAmount").alias("TotalLoanDisbursed"),
    count("CustomerID").alias("ActiveCustomers"),
    sum(when(col("DefaultFlag") == 1, 1).otherwise(0)).alias("TotalDefaults"),
    (sum(when(col("DefaultFlag") == 1, 1).otherwise(0)) / count("LoanID")).alias("DefaultRate")
)

display(region_performance)

# COMMAND ----------

# Analyze default rate by customer income level to assess repayment behavior
income_default_rate = loan_customer_df.groupBy("LoanType").agg(
    sum(when(col("DefaultFlag") == 1, 1).otherwise(0)).alias("DefaultCount"),
    count("LoanID").alias("TotalLoans")
).withColumn(
    "DefaultRate", col("DefaultCount") / col("TotalLoans")
)

display(income_default_rate)

# COMMAND ----------

# Analyze default rate and average loan size by EmploymentStatus for actionable insights
from pyspark.sql.functions import col, when, avg, count, sum

employment_insight = loan_customer_df.groupBy("EmploymentStatus").agg(
    avg("PrincipalAmount").alias("AverageLoanSize"),
    avg("AnnualIncome").alias("AverageIncome"),
    count("LoanID").alias("TotalLoans")
)

display(employment_insight)

# COMMAND ----------

from pyspark.sql.functions import col, avg

# Calculate average loan-to-income ratio by EmploymentStatus
employment_loan_income_ratio = loan_customer_df.withColumn(
    "LoanToIncomeRatio", col("PrincipalAmount") / col("AnnualIncome")
).groupBy("EmploymentStatus").agg(
    avg("LoanToIncomeRatio").alias("AvgLoanToIncomeRatio"),
    avg("PrincipalAmount").alias("AverageLoanSize"),
    avg("AnnualIncome").alias("AverageIncome")
)

display(employment_loan_income_ratio)