library("mlogit", "gmnl", "stargazer")
# DATA NOT INCLUDED (PRIVACY)!
df <- read.csv("survey_long.csv")

df$income <- factor(df$income, ordered=FALSE)

df$school_degree <- factor(df$school_degree, ordered=FALSE)
df$degree <- factor(df$degree, ordered=FALSE)
df$household_income <- factor(df$household_income, ordered=FALSE)

df$parking_strategy <- factor(df$parking_strategy, ordered=FALSE, labels=c('goal','garage','cruising', 'other'))


df$travel_mode_of_choice <- factor(df$travel_mode_of_choice, ordered=FALSE, labels=c('car','bike','public', 'foot', 'other'))



df$gender <- factor(df$gender, ordered=FALSE, labels=c('male','female','other'))
df$foreign_born_parents <- factor(df$foreign_born_parents, ordered=FALSE)
df$time <- factor(df$time, ordered=FALSE)
df$purpose <- factor(df$purpose, ordered=FALSE)


df <- within(df, time <- relevel(time, ref = "morgens"))



df$choiceid <- 1:nrow(df)

new_data <- mlogit.data(
  df,
  choice = "choice",
  shape = "wide",
  varying = 19:28,
  sep = "_",
  id.var = "id"
)


inspect <- data.frame(new_data)

# Simple model with Main Effects

simple_mx_model <- mlogit(choice ~ access + egress + search + type +  fee | age + gender,
                   rpar=c(access='n', egress='n', search='n', fee='n', typegarage = 'n', "1:age"="n", "1:genderfemale"="n"), 
                   R = 100, 
                   panel=TRUE,
                   halton = NA,
                   print.level = 2,
                   new_data)

print(summary(simple_mx_model))





# Complete Model


mx_model <- mlogit(choice ~ access + access:parking_strategy +  access:time + access:purpose + egress + egress:parking_strategy + egress:purpose + egress:time + search + search:parking_strategy + search:purpose + search:time + type + type:parking_strategy  + type:purpose + type:time +  fee + fee:purpose + fee:parking_strategy +  fee:time +  fee:household_income | age + gender,
                   rpar=c(access='n', egress='n', search='n', fee='n', typegarage = 'n', "1:age"="n", "1:genderfemale"="n"),
                   R = 100, 
                   panel=TRUE,
                   halton = NA,
                   print.level = 2,
                   new_data)

print(summary(mx_model))

print(coef(mx_model) / -coef(mx_model)["fee"])


saveRDS(mx_model, "logit_model_parking_choice.rds")
1