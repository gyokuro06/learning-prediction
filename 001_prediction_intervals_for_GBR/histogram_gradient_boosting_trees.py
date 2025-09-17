import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Features in Histogram Gradient Boosting Trees""")
    return


@app.cell
def _():
    from sklearn.datasets import fetch_openml

    electricity = fetch_openml(name="electricity", version=1, as_frame=True, parser="pandas")
    df = electricity.frame
    df
    return (df,)


@app.cell
def _(df):
    df["transfer"][:17_760].unique()
    return


@app.cell
def _(df, pl):
    exlude_stepwise_df = df.iloc[17_760:]
    X = exlude_stepwise_df.drop(columns=["transfer", "class"])
    y = exlude_stepwise_df["transfer"]

    fig = pl
    return


if __name__ == "__main__":
    app.run()
