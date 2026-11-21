X = df_clean.drop(columns=["Survived"])
y = df_clean["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("Train:", X_train.shape[0], "Test:", X_test.shape[0])
print("Class balance train:", y_train.value_counts(normalize=True).to_dict())
print("Class balance test:", y_test.value_counts(normalize=True).to_dict())