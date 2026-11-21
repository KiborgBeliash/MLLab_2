best = summary.index[0]
y_pred_best = pred_map[best]
cm_best = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm_best.ravel()

err_df = pd.DataFrame({"true":y_test.to_numpy(),"pred":y_pred_best})
err_df["correct"] = (err_df["true"]==err_df["pred"]).astype(int)
err_df = pd.concat([err_df, X_test.reset_index(drop=True)], axis=1)

wrong_fn = err_df[(err_df["true"]==1)&(err_df["pred"]==0)]
wrong_fp = err_df[(err_df["true"]==0)&(err_df["pred"]==1)]

print(f"Лучшая модель: {best}")
print(f"CM: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"\nFalse Negatives (выжил, но модель сказала погиб): {len(wrong_fn)} шт")
print(wrong_fn.head(10)[["true","pred","Sex","Age","Pclass","Fare"]])
print(f"\nFalse Positives (погиб, но модель сказала выжил): {len(wrong_fp)} шт")
print(wrong_fp.head(10)[["true","pred","Sex","Age","Pclass","Fare"]])

plt.figure(figsize=(6,4))
plt.bar(["FN (missed)", "FP (false alarm)"], [len(wrong_fn), len(wrong_fp)], color=["red","orange"], edgecolor="black")
plt.title(f"Типы ошибок: {best}"); plt.ylabel("Count")
plt.tight_layout(); plt.show()