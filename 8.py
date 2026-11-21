rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results["Random Forest"] = {
    "Accuracy": accuracy_score(y_test, y_pred_rf),
    "Precision": precision_score(y_test, y_pred_rf),
    "Recall": recall_score(y_test, y_pred_rf),
    "F1": f1_score(y_test, y_pred_rf)
}
print("RF metrics:", results["Random Forest"])
rf_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nRF importance:\n", rf_imp)