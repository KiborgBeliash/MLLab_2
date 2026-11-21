gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3,
                                 subsample=0.8, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
results["Gradient Boosting"] = {
    "Accuracy": accuracy_score(y_test, y_pred_gb),
    "Precision": precision_score(y_test, y_pred_gb),
    "Recall": recall_score(y_test, y_pred_gb),
    "F1": f1_score(y_test, y_pred_gb)
}
print("GB metrics:", results["Gradient Boosting"])
gb_imp = pd.Series(gb.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nGB importance:\n", gb_imp)