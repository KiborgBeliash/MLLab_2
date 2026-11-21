dt = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=5)
dt.fit(X_train, y_train)
y_pred_dt_train = dt.predict(X_train)
y_pred_dt = dt.predict(X_test)
results["Decision Tree"] = {
    "Accuracy": accuracy_score(y_test, y_pred_dt),
    "Precision": precision_score(y_test, y_pred_dt),
    "Recall": recall_score(y_test, y_pred_dt),
    "F1": f1_score(y_test, y_pred_dt)
}
print("DT train acc:", round(accuracy_score(y_train, y_pred_dt_train),4))
print("DT test acc:", round(results["Decision Tree"]["Accuracy"],4))
print("DT metrics:", results["Decision Tree"])
dt_imp = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nDT importance:\n", dt_imp)