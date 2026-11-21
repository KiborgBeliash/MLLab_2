df_clean = df.copy()

df_clean.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

df_clean["Embarked"].fillna(df_clean["Embarked"].mode()[0], inplace=True)

age_median = df_clean.groupby("Sex")["Age"].transform("median")
df_clean["Age"].fillna(age_median, inplace=True)

le_sex = LabelEncoder()
df_clean["Sex"] = le_sex.fit_transform(df_clean["Sex"])

embarked_dummies = pd.get_dummies(df_clean["Embarked"], prefix="Embarked", drop_first=True)
df_clean = pd.concat([df_clean.drop(columns=["Embarked"]), embarked_dummies], axis=1)

print("After preprocessing shape:", df_clean.shape)
print("\nПропуски после обработки:\n", df_clean.isna().sum())
print("\nHead:\n", df_clean.head())