scaler = StandardScaler().fit(X_train)
Xtr_s, Xte_s = scaler.transform(X_train), scaler.transform(X_test)

svm = SVC(kernel="rbf", random_state=42)
svm.fit(Xtr_s, y_train)
y_pred_svm = svm.predict(Xte_s)
results["SVM"] = {
    "Accuracy": accuracy_score(y_test, y_pred_svm),
    "Precision": precision_score(y_test, y_pred_svm),
    "Recall": recall_score(y_test, y_pred_svm),
    "F1": f1_score(y_test, y_pred_svm)
}
print("SVM metrics:", results["SVM"])