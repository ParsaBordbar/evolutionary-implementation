from data import load_data, standardize_train_test, stratified_split
from es_component import evaluate_final, train_es
from plot import plot_training_curves


if __name__ == '__main__':
    df = load_data('./Heart Disease dataset.csv')
    print(df.columns.tolist())
    # Convert UCI labels to binary (0 = no disease, 1 = disease)
    df['num'] = (df['num'] > 0).astype(int)

    X_train, X_test, y_train, y_test = stratified_split(df, target_col='num')
    X_train, X_test, _, _ = standardize_train_test(X_train, X_test)

    population, best_losses, mean_losses, train_acc = train_es(
        X_train, y_train,
        mu=30,
        lambd=210,
        generations=100,
        lambda_reg=0.01
    )

    best_individual = max(population, key=lambda ind: ind.fitness)
    acc, precision, recall, f1, cm = evaluate_final(best_individual, X_test, y_test)

    print('Test Accuracy:', acc)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)
    print('Confusion Matrix:\n', cm)

    plot_training_curves(best_losses, mean_losses, train_acc)