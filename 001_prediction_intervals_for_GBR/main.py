import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Example: [scikit-learn](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html)""")
    return


@app.cell
def _():
    import numpy as np
    from sklearn.model_selection import train_test_split

    def f(x):
        return x * np.sin(x)

    rng = np.random.RandomState(42)
    X = np.atleast_2d(rng.uniform(0, 10.0, size=1000)).T
    expected_y = f(X).ravel()

    sigma = 0.5 + X.ravel() / 10
    noise = rng.lognormal(sigma=sigma) - np.exp(sigma**2 / 2)
    y = expected_y + noise

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    return


@app.cell
def _():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_pinball_loss, mean_squared_error
    return


if __name__ == "__main__":
    app.run()
