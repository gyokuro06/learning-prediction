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
def _(df):
    import plotly.express as px
    import plotly.graph_objects as go

    exclude_stepwise_df = df.iloc[17_760:]
    X = exclude_stepwise_df.drop(columns=["transfer", "class"])
    y = exclude_stepwise_df["transfer"]

    # 曜日番号→英語名のマッピング（必要に応じて調整）
    day_map = {1: "Sun", 2: "Mon", 3: "Tue", 4: "Wed", 5: "Thu", 6: "Fri", 7: "Sat"}

    # groupbyで曜日・periodごとにtransferの平均値を計算
    grouped = exclude_stepwise_df.groupby(["day", "period"])["transfer"].mean().reset_index()

    # Plotlyで曜日ごとに線を描画
    fig = go.Figure()

    for day_num, day_name in day_map.items():
        day_data = grouped[grouped["day"] == day_num]
        fig.add_trace(go.Scatter(
            x=day_data["period"],
            y=day_data["transfer"],
            mode="lines+markers",
            name=day_name
        ))

    fig.update_layout(
        title="Hourly energy transfer for different days of the week (Plotly)",
        xaxis_title="Normalized time of the day",
        yaxis_title="Normalized energy transfer",
        legend_title="Day of week",
        width=1000,
        height=600
    )

    fig.show()
    return


if __name__ == "__main__":
    app.run()
