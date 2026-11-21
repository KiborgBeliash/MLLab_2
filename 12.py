summary = pd.DataFrame(results).T.sort_values("F1", ascending=False)
print("\nИТОГОВАЯ ТАБЛИЦА:\n", summary.round(4))

pred_map = {"Dummy":y_pred_dummy, "Decision Tree":y_pred_dt, "Random Forest":y_pred_rf,
            "Gradient Boosting":y_pred_gb, "SVM":y_pred_svm, "KNN":y_pred_knn}

fig, axes = plt.subplots(2, 3, figsize=(15,10))

for i, (name, preds) in enumerate(pred_map.items()):
    cm = confusion_matrix(y_test, preds)
    ax = axes[i//3][i%3]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(name)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row,col]), ha="center", va="center", fontsize=14)

plt.tight_layout(); plt.show()