knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(Xtr_s, y_train)
y_pred_knn = knn.predict(Xte_s)
results["KNN"] = {
    "Accuracy": accuracy_score(y_test, y_pred_knn),
    "Precision": precision_score(y_test, y_pred_knn),
    "Recall": recall_score(y_test, y_pred_knn),
    "F1": f1_score(y_test, y_pred_knn)
}
print("KNN metrics:", results["KNN"])